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

    /// Per-request engine instance shared by every TriAttentionKVCache in
    /// this cache array. The engine keeps weak refs back to caches, so this
    /// strong reference does not create a cycle.
    public let engine: TriAttentionV3Engine

    /// Sequence id for V3's per-sequence state. Phase A is single-batch
    /// so this stays 0; multi-batch needs request-id plumbing.
    public var seqId: Int = 0

    /// Absolute RoPE position for the next appended token. This intentionally
    /// differs from `offset` after physical compaction: `offset` is compacted
    /// storage length, while `logicalOffset` is the model's original token
    /// position stream.
    public private(set) var logicalOffset: Int = 0

    /// Maps compacted storage slot -> original logical token position.
    /// Needed because V3 evicts by original position, while this cache stores
    /// survivors densely after compaction.
    private var positionIds: [Int] = []

    public init(layerIdx: Int, engine: TriAttentionV3Engine) {
        self.layerIdx = layerIdx
        self.engine = engine
        super.init()
        engine.registerCache(layerIdx: layerIdx, cache: self)
    }

    deinit {
        engine.unregisterCache(layerIdx: layerIdx)
    }

    /// Called by the model's attention forward to commit new K/V into
    /// the cache. We let `super` handle the actual storage append, then
    /// fire V3's per-layer score accumulation. For prefill (T > 1) the
    /// score is computed against ALL cached positions; for decode
    /// (T == 1) it's a no-op since V3 calibrates on prefill Q only.
    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let nNew = keys.dim(2)
        let firstLogicalPos = logicalOffset
        let result = super.update(keys: keys, values: values)
        for p in firstLogicalPos..<(firstLogicalPos + nNew) {
            positionIds.append(p)
        }
        logicalOffset += nNew

        guard engine.calibrated else { return result }

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
    /// The input set contains ORIGINAL logical positions. We translate
    /// through `positionIds` to compacted storage indices so repeated
    /// eviction rounds still remove the intended cells after prior
    /// compactions. `offset` drops to the dense storage length, while
    /// `logicalOffset` is intentionally left unchanged.
    /// Map storage-space indices (what the engine emits in evictPos
    /// after compaction has happened on prior rounds) back to original
    /// token positions. Used by the rescue handler so the eviction
    /// callback decodes via _PROMPT_TOKEN_IDS at the right offsets.
    /// Returns nil when positionIds isn't tracking (ie cache hasn't
    /// been touched yet).
    public func translateStorageIndicesToOriginal(
        _ storageIndices: [Int]
    ) -> [Int]? {
        guard !positionIds.isEmpty else { return nil }
        var result: [Int] = []
        result.reserveCapacity(storageIndices.count)
        for idx in storageIndices {
            guard idx >= 0, idx < positionIds.count else { continue }
            result.append(positionIds[idx])
        }
        return result
    }

    public func removePositions(_ evicted: Set<Int>) {
        guard !evicted.isEmpty,
              let oldKeys = keys, let oldValues = values else { return }
        let len = self.offset
        if positionIds.count != len {
            positionIds = Array(0..<len)
        }
        var keepStorageIdx: [Int] = []
        var keepPositions: [Int] = []
        keepStorageIdx.reserveCapacity(len)
        keepPositions.reserveCapacity(len)
        for (storageIdx, logicalPos) in positionIds.enumerated()
            where !evicted.contains(logicalPos)
        {
            keepStorageIdx.append(storageIdx)
            keepPositions.append(logicalPos)
        }
        // Storage shape [B, kvHeads, T, headDim]. Build the surviving
        // slice as a list of single-position slices then concat — more
        // robust than MLX.take with axis (which had layout issues in
        // testing). Each kept[i] grabs [B, kvHeads, 1, headDim].
        let liveK: MLXArray
        let liveV: MLXArray
        if keepStorageIdx.isEmpty {
            // All positions evicted (degenerate case) — keep cache structure
            // but reset offset to 0.
            self.offset = 0
            self.positionIds = []
            return
        }
        var kSlices: [MLXArray] = []
        var vSlices: [MLXArray] = []
        kSlices.reserveCapacity(keepStorageIdx.count)
        vSlices.reserveCapacity(keepStorageIdx.count)
        for p in keepStorageIdx {
            kSlices.append(oldKeys[.ellipsis, p..<(p + 1), 0...])
            vSlices.append(oldValues[.ellipsis, p..<(p + 1), 0...])
        }
        liveK = concatenated(kSlices, axis: 2)
        liveV = concatenated(vSlices, axis: 2)
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
        self.positionIds = keepPositions
        // Compression telemetry: every round logs nBefore→nKept (=
        // physical KV cells alive after compaction) so a caller can
        // measure savings %. Gated on VLLM_TRIATT_COMPRESSION_LOG=1
        // (default off — keeps decode quiet). Aggregate counters are
        // also exposed via `compressionStats`.
        Self.recordCompactionRound(
            nBefore: len, nEvicted: evicted.count, nKept: newOffset,
        )
        // Materialize to break the lazy-graph chain (same trick as
        // KVCacheSimple.update's reset path).
        eval(self.keys!, self.values!)
    }

    /// Process-wide rolling telemetry: total seen / evicted / kept across
    /// all rounds + caches. Exposed for harness / smoke / dashboard.
    public struct CompressionStats: Sendable {
        public var rounds: Int = 0
        public var totalBefore: Int = 0
        public var totalEvicted: Int = 0
        public var totalKept: Int = 0
        public var savingsPct: Double {
            guard totalBefore > 0 else { return 0.0 }
            return 100.0 * Double(totalEvicted) / Double(totalBefore)
        }
    }

    nonisolated(unsafe) private static let _statsLock = NSLock()
    nonisolated(unsafe) private static var _stats = CompressionStats()

    /// Read the current compression stats (thread-safe snapshot).
    public static var compressionStats: CompressionStats {
        _statsLock.lock(); defer { _statsLock.unlock() }
        return _stats
    }

    /// Reset stats. Call at the start of a new benchmark / smoke run.
    public static func resetCompressionStats() {
        _statsLock.lock(); defer { _statsLock.unlock() }
        _stats = CompressionStats()
    }

    private static func recordCompactionRound(
        nBefore: Int, nEvicted: Int, nKept: Int,
    ) {
        _statsLock.lock()
        _stats.rounds += 1
        _stats.totalBefore += nBefore
        _stats.totalEvicted += nEvicted
        _stats.totalKept += nKept
        let snapshot = _stats
        _statsLock.unlock()
        if ProcessInfo.processInfo
            .environment["VLLM_TRIATT_COMPRESSION_LOG"] == "1" {
            let pct = nBefore > 0
                ? 100.0 * Double(nEvicted) / Double(nBefore) : 0.0
            print("[V3-compaction] round=\(snapshot.rounds) "
                  + "before=\(nBefore) evicted=\(nEvicted) "
                  + "kept=\(nKept) saved=\(String(format: "%.1f%%", pct))")
        }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = super.trim(n)
        if trimmed > 0 && positionIds.count >= trimmed {
            positionIds.removeLast(trimmed)
            logicalOffset = max(0, logicalOffset - trimmed)
        }
        return trimmed
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
        // Only open a fresh score round when there isn't one open OR
        // the pending shape doesn't match. Mirrors the `needs_open`
        // branch in Python's backend_helpers.accumulate_prefill_k.
        // Unconditionally opening would reset pendingLayers on every
        // layer's call, defeating the expected_layers gate.
        if pendingScoreLen(seqId: seqId) != cachedLen {
            beginScoreRound(seqId: seqId, seqLen: cachedLen)
        }
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
        // expected_layers gate: hold finalize until all attention layers
        // have contributed to the pending score buffer. Mirrors the AMD
        // Python `cfg.expected_layers` knob. Default derives from
        // nLayers - boundarySkip; override via cfg.expectedLayers when
        // the runtime intentionally leaves boundary layers outside the
        // V3 hook (e.g. fp16 boundary skip on the TQ side). Without
        // this gate, synthetic single-cache tests fire eviction on the
        // first cache.update; real models with all-layers iteration
        // would fire correctly anyway, but we want parity.
        let expected = max(
            1, cfg.expectedLayers ?? (nLayers - cfg.boundarySkip)
        )
        if pendingLayersCount(seqId: seqId) < expected { return }
        let evicted = finalizeEvictRound(seqId: seqId)
        guard evicted > 0 else { return }
        // Apply compaction. The engine doesn't keep evict_pos directly
        // after finalize — pull from the seq state's recently-updated
        // valid mask: any position where mask[p] == False AND p < length
        // is now evicted. These are STORAGE indices (engine scores
        // against the cache's compacted K), not original token
        // positions.
        let validMask = getValidMask(seqId: seqId, seqLen: effectiveSeqLen)
        let validBoolArr: [Bool] = validMask.asArray(Bool.self)
        var evictedStorageIdx: [Int] = []
        evictedStorageIdx.reserveCapacity(evicted)
        for p in 0..<validBoolArr.count where !validBoolArr[p] {
            evictedStorageIdx.append(p)
        }
        // Translate storage-indices → ORIGINAL TOKEN POSITIONS via any
        // registered cache's positionIds map. After compaction the two
        // diverge; the rescue text-decode in TriAttentionRescue indexes
        // _PROMPT_TOKEN_IDS by original position, so we MUST translate
        // before the eviction callback fires. positionIds is identical
        // across all per-layer caches for this seq (same prefill token
        // stream), so picking any one is fine. Without this fix, the
        // first eviction round decodes correctly (storage_idx ==
        // original_pos before any compaction) but subsequent rounds
        // decode the wrong text — the bug the audit flagged.
        var evictedOriginal: [Int] = evictedStorageIdx
        if let enumerator = cacheRegistry.objectEnumerator(),
           let firstCache = enumerator.nextObject() as? TriAttentionKVCache,
           let translated = firstCache.translateStorageIndicesToOriginal(
            evictedStorageIdx
           )
        {
            evictedOriginal = translated
        }
        // Replay through the rescue callback now that positions are in
        // the original-token space (engine.finalizeEvictRound already
        // fired the callback with storage indices — we override here
        // with the corrected indices for the rescue handler).
        if let cb = evictionCallback {
            cb(seqId, evictedOriginal, evictedOriginal.count)
        }
        // Walk every registered cache and physically remove the
        // evicted positions. removePositions accepts the ORIGINAL
        // positions and translates internally via positionIds. Storage
        // compaction happens here — after this, positionIds shrinks
        // to surviving entries.
        let evictedSet = Set(evictedOriginal)
        if let enumerator = cacheRegistry.objectEnumerator() {
            while let cache = enumerator.nextObject() as? TriAttentionKVCache {
                cache.removePositions(evictedSet)
            }
        }
    }
}
