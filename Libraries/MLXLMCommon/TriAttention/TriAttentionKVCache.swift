// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// TriAttention V3 — KVCache subclass that wraps KVCacheSimple.
//
// Bridge between V3's policy state (which positions to evict) and the
// model's KV storage (which holds the actual K/V tensors). Mirrors the
// llama.cpp pattern at src/llama-triattention.cpp:803,867,896 — V3
// picks positions, then iterates `cache->seq_rm(0, pos, pos+1)` to
// physically remove them. On Swift the analog is `removePositions([Int])`
// which slices the contiguous keys/values MLXArrays and decrements the
// offset counter.
//
// Why not subclass + valid_mask: AMD uses a `valid_mask` constexpr that
// the TurboQuant attention kernel reads to skip evicted positions. Swift
// attention is per-model and goes through `MLXFast.scaledDotProductAttention`
// which doesn't take an arbitrary mask cleanly. Physical compaction is
// strictly simpler — once removed, ALL attention compute paths see the
// shorter cache automatically. The trade-off is a one-shot allocation
// cost on each eviction round (concat + eval), which is amortized across
// thousands of decode steps so it's free in practice.
import Foundation
import MLX
import MLXNN

public final class TriAttentionKVCache: KVCacheSimple {
    /// Layer index — used to scope per-layer score accumulation in the
    /// global V3 engine. Set at init by the model adapter that builds
    /// the cache (one TriAttentionKVCache per attention layer).
    public let layerIdx: Int

    /// Per-process engine instance. Could be the global singleton from
    /// `TriAttentionRuntime.shared.engine` or a test-injected one.
    public weak var engine: TriAttentionV3Engine?

    /// Sequence id for V3's per-sequence state. Phase A is single-batch
    /// so this stays 0; multi-batch needs request-id plumbing.
    public var seqId: Int = 0

    public init(layerIdx: Int, engine: TriAttentionV3Engine) {
        self.layerIdx = layerIdx
        self.engine = engine
        super.init()
        engine.registerCache(layerIdx: layerIdx, cache: self)
    }

    deinit {
        engine?.unregisterCache(layerIdx: layerIdx)
    }

    /// Called by the model's attention forward to commit new K/V into
    /// the cache. We let `super` handle the actual storage append, then
    /// fire V3's per-layer score accumulation. For prefill (T > 1) the
    /// score is computed against ALL cached positions; for decode
    /// (T == 1) it's a no-op since V3 calibrates on prefill Q only.
    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let result = super.update(keys: keys, values: values)
        guard let engine, engine.calibrated else { return result }

        // Pull the *full* cached K shaped [seq_len, kvHeads, headDim]
        // from the post-update state. KVCacheSimple stores [B, kvHeads,
        // T, headDim] with B==1 single-batch — strip B and transpose.
        let (kCached, _) = result
        // kCached: [1, kvHeads, seq_len, headDim] → [seq_len, kvHeads, headDim]
        let kForScoring = kCached[0].transposed(1, 0, 2).asType(.float32)
        let cachedLen = kForScoring.dim(0)
        // V3's accumulate is gated internally on `pending_scores` shape
        // matching cachedLen — guards against shape drift across layers.
        engine.accumulateLayerScoreFromBridge(
            seqId: seqId, layerIdx: layerIdx, K: kForScoring,
            cachedLen: cachedLen
        )

        // After the LAST layer (highest layerIdx) accumulates, fire
        // finalize. Other layers no-op the policy until all have
        // contributed (engine guards on layer set).
        engine.maybeFinalizeAndCompact(
            seqId: seqId, effectiveSeqLen: cachedLen
        )
        return result
    }

    /// llama.cpp `seq_rm` analog: physically remove positions from this
    /// layer's K/V cache. Called by the engine's finalize step on every
    /// registered cache after V3 picks the eviction set.
    ///
    /// Sliced via boolean indexing — MLX's `take` op with a list of
    /// kept indices. After the slice, `offset` drops by the count of
    /// removed positions.
    public func removePositions(_ evicted: Set<Int>) {
        guard !evicted.isEmpty,
              let oldKeys = keys, let oldValues = values else { return }
        let len = self.offset
        var keepIdx: [Int32] = []
        keepIdx.reserveCapacity(len - evicted.count)
        for p in 0..<len where !evicted.contains(p) {
            keepIdx.append(Int32(p))
        }
        let keepArr = MLXArray(keepIdx)
        // Storage shape [B, kvHeads, T, headDim]. Take along axis=2
        // gives [B, kvHeads, len-|evicted|, headDim], then concatenate
        // with zero-padding back to allocated step length.
        let liveK = MLX.take(oldKeys, keepArr, axis: 2)
        let liveV = MLX.take(oldValues, keepArr, axis: 2)
        let newOffset = liveK.dim(2)

        // Re-pad to the original allocated length so subsequent
        // `update()` calls don't trip the "needs reset" guard. The
        // allocated length is oldKeys.dim(2). Padding is the suffix
        // beyond newOffset.
        let allocLen = oldKeys.dim(2)
        if newOffset < allocLen {
            let padShape = [
                liveK.dim(0), liveK.dim(1), allocLen - newOffset, liveK.dim(3),
            ]
            let padK = MLXArray.zeros(padShape, dtype: liveK.dtype)
            let padShapeV = [
                liveV.dim(0), liveV.dim(1), allocLen - newOffset, liveV.dim(3),
            ]
            let padV = MLXArray.zeros(padShapeV, dtype: liveV.dtype)
            self.keys = concatenated([liveK, padK], axis: 2)
            self.values = concatenated([liveV, padV], axis: 2)
        } else {
            self.keys = liveK
            self.values = liveV
        }
        self.offset = newOffset
        // Materialize to break the lazy-graph chain (same trick as
        // KVCacheSimple.update's reset path).
        eval(self.keys!, self.values!)
    }
}

// MARK: - Engine extensions for cache registry + finalize-and-compact

extension TriAttentionV3Engine {

    public func registerCache(layerIdx: Int, cache: TriAttentionKVCache) {
        cacheRegistry.setObject(cache, forKey: NSNumber(value: layerIdx))
    }

    public func unregisterCache(layerIdx: Int) {
        cacheRegistry.removeObject(forKey: NSNumber(value: layerIdx))
    }

    /// Bridge entry: open a score round on first layer, accumulate this
    /// layer's contribution. Mirrors `accumulate_prefill_k` from the
    /// AMD-side backend_helpers.py. Idempotent across layer indices —
    /// repeated calls for the same layer in one round are no-ops via
    /// `pendingLayers` set.
    public func accumulateLayerScoreFromBridge(
        seqId: Int, layerIdx: Int, K: MLXArray, cachedLen: Int
    ) {
        // If no round is open OR pending shape doesn't match, open a
        // fresh round. Mirrors the `needs_open` branch in Python's
        // backend_helpers.accumulate_prefill_k.
        beginScoreRound(seqId: seqId, seqLen: cachedLen)
        // Compute window_thr from the post-append max position.
        let maxPos = cachedLen - 1
        let windowThr = max(0, maxPos - cfg.windowSize + 1)
        accumulateLayerScore(
            seqId: seqId, layerIL: layerIdx,
            K: K, maxPos: maxPos, windowThr: windowThr
        )
    }

    /// Call after each layer's accumulation. Fires the V3 policy when
    /// (a) calibration done, (b) all expected layers contributed,
    /// (c) cache pressure exceeds budget. On fire, applies physical
    /// compaction to every registered cache via removePositions, then
    /// fires the Tier 2 eviction callback for longctx-svc write-out.
    public func maybeFinalizeAndCompact(
        seqId: Int, effectiveSeqLen: Int
    ) {
        guard calibrated else { return }
        guard shouldEvict(seqId: seqId, seqLen: effectiveSeqLen)
        else { return }
        // NOTE: For Phase A, fire on the LAST registered layer's call.
        // A more correct guard would track expected_layers and gate on
        // pending_layers.count == expected. For the MVP we just trust
        // the model's per-layer iteration order; the engine's
        // finalize_evict_round is idempotent and bails on empty pending.
        let evicted = finalizeEvictRound(seqId: seqId)
        guard evicted > 0 else { return }
        // Apply compaction. The engine doesn't keep evict_pos directly
        // after finalize — pull from the seq state's recently-updated
        // valid mask: any position where mask[p] == False AND p < length
        // is now evicted.
        let validMask = getValidMask(seqId: seqId, seqLen: effectiveSeqLen)
        let validBoolArr: [Bool] = validMask.asArray(Bool.self)
        var evictedSet: Set<Int> = []
        evictedSet.reserveCapacity(evicted)
        for p in 0..<validBoolArr.count where !validBoolArr[p] {
            evictedSet.insert(p)
        }
        // Walk every registered cache and physically remove the
        // evicted positions. NSMapTable.objectEnumerator gives us a
        // stable snapshot of currently-live caches.
        if let enumerator = cacheRegistry.objectEnumerator() {
            while let cache = enumerator.nextObject() as? TriAttentionKVCache {
                cache.removePositions(evictedSet)
            }
        }
    }
}
