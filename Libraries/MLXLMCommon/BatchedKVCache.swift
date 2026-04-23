// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// Flat batched KV cache: single pre-allocated tensor for B requests.
///
/// Instead of B separate KVCacheSimple objects, stores all K/V in
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

        if allSameOffset {
            // Single batched write — no per-request loop
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

    /// Get cached K/V for all active requests up to their offsets.
    /// Returns (K, V, mask) for batched SDPA.
    /// K: [B, kv_heads, max_offset, head_dim]
    /// mask: [B, 1, 1, max_offset] with -inf for positions beyond each request's offset
    public func getCachedWithMask() -> (MLXArray, MLXArray, MLXArray) {
        let B = active
        let maxOff = offsets[0..<B].max() ?? 0

        let k = keys[..<B, 0..., ..<maxOff, 0...]
        let v = values[..<B, 0..., ..<maxOff, 0...]

        // Build mask: [B, 1, 1, maxOff] matching cache dtype
        let cacheDtype = k.dtype
        let positions = MLXArray(0..<maxOff)
        var maskRows = [MLXArray]()
        for i in 0..<B {
            let valid = positions .< MLXArray(offsets[i])
            let row = MLX.where(valid,
                                MLXArray(Float(0)).asType(cacheDtype),
                                MLXArray(Float(-1e9)).asType(cacheDtype))
            maskRows.append(row)
        }
        let mask = MLX.stacked(maskRows, axis: 0)
            .reshaped(B, 1, 1, maxOff)

        return (k, v, mask)
    }

    /// Reset all slots
    public func reset() {
        active = 0
        for i in 0..<maxBatch { offsets[i] = 0 }
    }
}
