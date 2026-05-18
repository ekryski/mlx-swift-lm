// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// Flat batched KV cache: single pre-allocated tensor for B requests.
///
/// Instead of B separate StandardKVCache objects, stores all K/V in
/// `[B, kv_heads, max_seq, head_dim]`. Cache update and attention
/// are single batched operations — no per-request loops.
public class BatchedKVCache {
    public let maxBatch: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxSeq: Int

    /// Current offset per request (how many tokens cached)
    public var offsets: [Int]

    /// K cache: [B, kv_heads, max_seq, head_dim]
    public var keys: MLXArray
    /// V cache: [B, kv_heads, max_seq, head_dim]
    public var values: MLXArray

    /// Active request count (first `active` slots are in use)
    public var active: Int = 0

    /// Cached mask — invalidated on cache update, shared across layers
    private var _cachedMask: MLXArray?
    private var _cachedMaxOff: Int = 0

    public init(maxBatch: Int, kvHeads: Int, headDim: Int, maxSeq: Int = 2048,
                dtype: DType = .bfloat16) {
        self.maxBatch = maxBatch
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.offsets = Array(repeating: 0, count: maxBatch)

        self.keys = MLXArray.zeros([maxBatch, kvHeads, maxSeq, headDim], dtype: dtype)
        self.values = MLXArray.zeros([maxBatch, kvHeads, maxSeq, headDim], dtype: dtype)
        eval(self.keys, self.values)
    }

    /// Add a new request. Returns batch slot index.
    public func addRequest() -> Int {
        let slot = active
        active += 1
        offsets[slot] = 0
        return slot
    }

    /// Set offset for a slot (after prefill)
    public func setOffset(_ slot: Int, _ offset: Int) {
        offsets[slot] = offset
    }

    /// Batched cache update: write new K/V for all active requests.
    /// newK, newV: [B_active, kv_heads, 1, head_dim]
    ///
    /// Fast path: when all requests have the same offset (common during
    /// continuous decode), uses single slice assignment — no loop.
    public func update(newKeys: MLXArray, newValues: MLXArray) {
        let B = active
        guard B > 0 else { return }

        let allSameOffset = offsets[0..<B].allSatisfy { $0 == offsets[0] }

        _cachedMask = nil  // invalidate mask cache

        if allSameOffset {
            let off = offsets[0]
            keys[..<B, 0..., off, 0...] = newKeys[0..., 0..., 0, 0...]
            values[..<B, 0..., off, 0...] = newValues[0..., 0..., 0, 0...]
            for i in 0..<B { offsets[i] = off + 1 }
        } else {
            // Different offsets — per-request write
            for i in 0..<B {
                let off = offsets[i]
                keys[i, 0..., off, 0...] = newKeys[i, 0..., 0, 0...]
                values[i, 0..., off, 0...] = newValues[i, 0..., 0, 0...]
                offsets[i] = off + 1
            }
        }
    }

    /// Prefill-aware variant: writes an L-token chunk for all active
    /// requests starting at each slot's current offset, advancing each
    /// slot by L.
    ///
    /// - Parameters:
    ///   - newKeys: `[B_active, kv_heads, L, head_dim]`
    ///   - newValues: `[B_active, kv_heads, L, head_dim]`
    ///
    /// Fast path: when all requests share the same offset (rectangular
    /// prefill), uses single slice assignment. The default `update()`
    /// hard-wires L=1 (decode step); this method is the L>1 sibling.
    public func updateChunk(newKeys: MLXArray, newValues: MLXArray) {
        let B = active
        guard B > 0 else { return }
        precondition(newKeys.shape.count == 4,
            "expected [B, kv_heads, L, head_dim], got \(newKeys.shape)")
        precondition(newValues.shape.count == 4,
            "expected [B, kv_heads, L, head_dim], got \(newValues.shape)")
        precondition(newKeys.dim(0) == B,
            "newKeys B=\(newKeys.dim(0)) must equal active=\(B)")
        precondition(newValues.dim(0) == B, "newValues B mismatch")
        let L = newKeys.dim(2)
        precondition(newValues.dim(2) == L, "newKeys/newValues L mismatch")
        precondition(L >= 1, "L must be >= 1")

        _cachedMask = nil

        let allSameOffset = offsets[0..<B].allSatisfy { $0 == offsets[0] }
        if allSameOffset {
            let off = offsets[0]
            precondition(off + L <= maxSeq,
                "BatchedKVCache.updateChunk: off+L=\(off + L) exceeds maxSeq=\(maxSeq)")
            keys[..<B, 0..., off ..< (off + L), 0...] = newKeys
            values[..<B, 0..., off ..< (off + L), 0...] = newValues
            for i in 0..<B { offsets[i] = off + L }
        } else {
            for i in 0..<B {
                let off = offsets[i]
                precondition(off + L <= maxSeq,
                    "BatchedKVCache.updateChunk: slot \(i) off+L=\(off + L) exceeds maxSeq=\(maxSeq)")
                keys[i, 0..., off ..< (off + L), 0...] = newKeys[i, 0..., 0..., 0...]
                values[i, 0..., off ..< (off + L), 0...] = newValues[i, 0..., 0..., 0...]
                offsets[i] = off + L
            }
        }
    }

    /// Get cached K/V for all active requests up to their offsets.
    /// Returns (K, V, mask) for batched SDPA.
    /// K: [B, kv_heads, max_offset, head_dim]
    /// mask: [B, 1, 1, max_offset] with -inf for positions beyond each request's offset
    public func getCachedWithMask() -> (MLXArray, MLXArray, MLXArray) {
        let B = active
        let maxOff = offsets[0..<B].max() ?? 0

        let k = keys[..<B, 0..., ..<maxOff, 0...]
        let v = values[..<B, 0..., ..<maxOff, 0...]

        // Build mask: [B, 1, 1, maxOff] — fully vectorized, no loop
        let cacheDtype = k.dtype
        let positions = MLXArray(0..<maxOff).reshaped(1, maxOff)  // [1, maxOff]
        let offsetsArr = MLXArray(offsets[0..<B]).reshaped(B, 1)  // [B, 1]
        let valid = positions .< offsetsArr  // [B, maxOff] broadcast
        let mask = MLX.where(valid,
                             MLXArray(Float(0)).asType(cacheDtype),
                             MLXArray(Float(-1e9)).asType(cacheDtype))
            .reshaped(B, 1, 1, maxOff)

        return (k, v, mask)
    }

    /// Reset all slots
    public func reset() {
        active = 0
        for i in 0..<maxBatch { offsets[i] = 0 }
    }
}
