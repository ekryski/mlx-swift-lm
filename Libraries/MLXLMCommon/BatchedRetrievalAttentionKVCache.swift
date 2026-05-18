// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedRetrievalAttentionKVCache — sparse-aware decode wrapper for the
// flat batched KV cache.
//
// Composition: an inner BatchedKVCache holds the rectangular fp16 K/V.
// The batched selector index sits alongside. At decode L=1:
//   1. cache.update writes new K/V into the rectangular buffer.
//   2. updateIndex folds the new K (post-RoPE) into per-slot block features.
//   3. sparseAttend projects per-(B, KV) Q -> top-K -> emits a per-slot
//      fp16 mask + dispatches MLXFast.SDPA on the sdpa_vector_2pass path.
//
// Assumes rectangular T (all slots at the same offset) at decode time.
// Continuous-batching ragged-T is a future concern.

import Foundation
import MLX

public final class BatchedRetrievalAttentionKVCache {

    /// Inner batched K/V cache. We hold a strong ref but don't own
    /// lifecycle exclusively — the model's per-layer cache list owns it.
    public let inner: BatchedKVCache

    /// Per-(B, KV-head) selector index.
    public let index: BatchedRetrievalAttentionIndexB

    public let raConfig: RetrievalAttentionConfig
    public let layerIdx: Int
    public let totalLayers: Int
    public let ropeBase: Float

    /// Cached `[B, nKVH]` int32 array of representative Q-head indices per
    /// KV group. Built lazily on first sparseAttend.
    private var cachedHeadIdx: MLXArray?

    /// Env-gated kernel selection. Default is the batched mask kernel
    /// (`fp16 additive mask + MLXFast SDPA` on the sdpa_vector_2pass path
    /// — this hits the tuned Metal fast path on M-series GPUs).
    ///
    /// Values:
    ///   "mask"  (default) — batched fp16 mask + MLXFast SDPA
    ///   "loop"            — Swift loop over per-slot single-batch mask + SDPA
    ///   "group"           — fused sparse SDPA group kernel (gather-based)
    ///   "gather"          — compose-gather (per-slot K/V materialize then dense SDPA)
    public static let envBatchedKernel: String = {
        ProcessInfo.processInfo.environment["VSM_SPARSE_BATCHED_KERNEL"]
            ?? "mask"
    }()

    /// Env override for the prefill sparse path. When set to `1`, the
    /// per-family `fullyBatchedSparsePrefill` hook engages the
    /// `prefillSparseAttend` branch for L>1 chunks (subject to the
    /// `sparsePrefillMinContext` threshold + sparse-eligible layer band).
    /// Default off — opt-in. Mirrors the decode-side knob in the bridge.
    public static let envSparsePrefillEnabled: Bool = {
        ProcessInfo.processInfo.environment["VSM_SPARSE_PREFILL"] == "1"
    }()

    public var isSparseEligible: Bool {
        raConfig.isSparseLayer(layerIdx: layerIdx, totalLayers: totalLayers)
    }

    public init(
        inner: BatchedKVCache,
        B: Int,
        nKVHeads: Int,
        dHead: Int,
        layerIdx: Int,
        totalLayers: Int,
        raConfig: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        ropeBase: Float = 10_000.0
    ) {
        self.inner = inner
        self.layerIdx = layerIdx
        self.totalLayers = totalLayers
        self.raConfig = raConfig
        self.ropeBase = ropeBase
        self.index = BatchedRetrievalAttentionIndexB(
            config: raConfig, B: B, dHead: dHead, nKVHeads: nKVHeads,
            ropeBase: ropeBase, layerIdx: layerIdx)
    }

    /// Apply a new K chunk to the selector index. Caller has ALREADY
    /// written K/V into the inner cache; this only updates selector state.
    /// Decode-step (L=1) is the common path; prefill chunks (L>1) also.
    public func updateIndex(newKeys: MLXArray) {
        guard isSparseEligible else { return }
        // Skip index updates on decode tokens — they live in the sliding
        // window so block-feature membership doesn't help selection.
        let L = newKeys.dim(2)
        if L > 1 {
            index.update(newKeys: newKeys.asType(.float32))
        }
        // (no L==1 update — sliding window covers it)
    }

    /// Sparse attend at decode step (L=1).
    ///
    /// - Parameters:
    ///   - queries: `[B, nQH, 1, D]` post-RoPE Q (already broadcasted with
    ///     GQA group expansion).
    ///   - scale: SDPA scale (1/sqrt(D)).
    /// - Returns: `[B, nQH, 1, D]` attention output.
    ///
    /// Dispatches by `VSM_SPARSE_BATCHED_KERNEL` env (see static doc).
    public func sparseAttend(
        queries: MLXArray, scale: Float
    ) -> MLXArray {
        precondition(queries.shape.count == 4, "queries must be [B, nQH, 1, D]")
        precondition(queries.dim(2) == 1, "decode-step L=1 only")
        let B = queries.dim(0)
        let nQH = queries.dim(1)
        precondition(B == index.B, "B mismatch")

        // Pull rectangular K/V at the current offset. All slots assumed
        // to share offset (rectangular T) — caller responsible if not.
        let off = inner.offsets[0]
        let cachedK = inner.keys[..<B, 0..., ..<off, 0...]
        let cachedV = inner.values[..<B, 0..., ..<off, 0...]
        let nKVH = cachedK.dim(1)
        let T = cachedK.dim(2)
        precondition(nQH % nKVH == 0)
        let groupSize = nQH / nKVH

        // Build rep-per-group Q indices once.
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            eval(cachedHeadIdx!)
        }
        let qSqueezed = queries[0..., 0..., 0, 0...]                         // [B, nQH, D]
        let qRep = qSqueezed.take(cachedHeadIdx!, axis: 1).asType(.float32)  // [B, nKVH, D]
        let projQ = index.projectQueriesBatched(qRep)                        // [B, nKVH, contentDim]

        let kernelChoice = BatchedRetrievalAttentionKVCache.envBatchedKernel
        switch kernelChoice {
        case "group":
            let (gather, _) = index.perKVHeadGatherBatched(projectedQ: projQ, seqLen: T)
            return retrievalAttentionGroupSparseSDPA(
                queries: queries,
                keys: cachedK,
                values: cachedV,
                perKVHeadGather: gather,
                scale: scale
            )

        case "loop":
            return sparseAttendLoopMask(
                queries: queries, projQ: projQ, K: cachedK, V: cachedV,
                T: T, scale: scale)

        case "gather":
            return sparseAttendComposeGather(
                queries: queries, projQ: projQ, K: cachedK, V: cachedV,
                T: T, scale: scale)

        default:  // "mask" + anything else
            return sparseAttendBatchedMask(
                queries: queries, projQ: projQ, K: cachedK, V: cachedV,
                T: T, scale: scale)
        }
    }

    /// Default — one batched mask-kernel launch + MLXFast SDPA on the
    /// sdpa_vector_2pass kernel variant. Mask is fp16 additive.
    private func sparseAttendBatchedMask(
        queries: MLXArray, projQ: MLXArray,
        K: MLXArray, V: MLXArray, T: Int, scale: Float
    ) -> MLXArray {
        // Per-slot per-KV-head top-K block starts. Both shapes [B, nKVH, K].
        let fineStarts = index.topKFineBlockStarts(projectedQ: projQ)
        let coarseStarts = index.topKCoarseBlockStarts(projectedQ: projQ)
        let mask = retrievalAttentionBuildMaskFusedBatched(
            fineStarts: fineStarts,
            coarseStarts: coarseStarts,
            T: T,
            staticInit: raConfig.staticInit,
            slidingWindow: raConfig.slidingWindow,
            fineBS: raConfig.fineBlockSize,
            coarseBS: raConfig.coarseBlockSize,
            outputDtype: K.dtype
        )
        // mask is [B, 1, 1, T] — broadcasts over nQH and L=1.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: K, values: V,
            scale: scale, mask: .array(mask)
        )
    }

    /// Fallback: B Swift-side launches of the single-batch mask kernel +
    /// per-slot MLXFast SDPA. Only for A/B sanity; expected slower at B>1.
    private func sparseAttendLoopMask(
        queries: MLXArray, projQ: MLXArray,
        K: MLXArray, V: MLXArray, T: Int, scale: Float
    ) -> MLXArray {
        let B = queries.dim(0)
        let fineStarts = index.topKFineBlockStarts(projectedQ: projQ)
        let coarseStarts = index.topKCoarseBlockStarts(projectedQ: projQ)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for b in 0..<B {
            let fineB = fineStarts[b, 0..., 0...]    // [nKVH, K_fine]
            let coarseB = coarseStarts[b, 0..., 0...]
            let mask = retrievalAttentionBuildMaskFused(
                fineStarts: fineB,
                coarseStarts: coarseB,
                T: T,
                staticInit: raConfig.staticInit,
                slidingWindow: raConfig.slidingWindow,
                fineBS: raConfig.fineBlockSize,
                coarseBS: raConfig.coarseBlockSize,
                outputDtype: K.dtype
            )  // [1, 1, 1, T]
            let qSlot = queries[b ..< (b + 1), 0..., 0..., 0...]  // [1, nQH, 1, D]
            let kSlot = K[b ..< (b + 1), 0..., 0..., 0...]
            let vSlot = V[b ..< (b + 1), 0..., 0..., 0...]
            let oSlot = MLXFast.scaledDotProductAttention(
                queries: qSlot, keys: kSlot, values: vSlot,
                scale: scale, mask: .array(mask)
            )
            outputs.append(oSlot)
        }
        return concatenated(outputs, axis: 0)
    }

    /// Compose-gather: build per-(B, nKVH) gather index `[B, nKVH, K_padded]`,
    /// take K/V along axis 2 to materialize `[B, nKVH, K_padded, D]` slabs,
    /// then call dense MLXFast SDPA with `mask: .none` on the small slab.
    /// GQA broadcast (nKVH → nQH) is handled internally by MLXFast.
    private func sparseAttendComposeGather(
        queries: MLXArray, projQ: MLXArray,
        K: MLXArray, V: MLXArray, T: Int, scale: Float
    ) -> MLXArray {
        let (gather, kPadded) = index.perKVHeadGatherBatched(
            projectedQ: projQ, seqLen: T)
        precondition(gather.shape.count == 3, "gather must be [B, nKVH, K_padded]")
        precondition(gather.dim(0) == queries.dim(0), "gather B mismatch")
        let gExp = gather.expandedDimensions(axis: 3)        // [B, nKVH, K_padded, 1]
        let kSmall = takeAlong(K, gExp, axis: 2)             // [B, nKVH, K_padded, D]
        let vSmall = takeAlong(V, gExp, axis: 2)             // [B, nKVH, K_padded, D]
        _ = kPadded
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: kSmall, values: vSmall,
            scale: scale, mask: .none
        )
    }

    // MARK: - F-83 sparse PREFILL (chunked attention, L > 1)

    /// Sparse attend for a chunk of L queries (L > 1) — the prefill path.
    ///
    /// Called from the model's `fullyBatchedSparsePrefill` hook when:
    ///   - L > 1 (prefill chunk)
    ///   - the layer is sparse-eligible
    ///   - `raConfig.sparsePrefillEnabled` is true (or env override set)
    ///   - the prior cache length exceeds `sparsePrefillMinContext`
    ///
    /// Pipeline (per slot):
    ///   1. Project chunk Q via the L-aware selector → union top-K
    ///      fine + coarse block starts (max-pooled across L queries
    ///      per KV head — NSA GQA pattern).
    ///   2. Combine with static prefix + sliding window, clip to prior
    ///      cache length, dedupe per slot. Duplicates in SDPA's K dim
    ///      silently double-count rows through softmax — dedupe is
    ///      load-bearing for correctness.
    ///   3. Gather prior K/V at those positions; concat with the chunk's
    ///      own K/V tail; run ONE MLXFast.SDPA with a `[L, P+L]` mask:
    ///      prior cols = 0 (attend, all positions < chunk_start so
    ///      causally safe), chunk cols = lower-triangular causal.
    ///
    /// Caller has ALREADY written the chunk's K/V into `inner` and
    /// updated the selector index via `updateIndex(newKeys:)`. The
    /// cache offsets already reflect the chunk being written —
    /// `inner.offsets[i]` is the post-chunk length for slot i.
    ///
    /// - Parameters:
    ///   - queries: `[B, nQH, L, D]` post-RoPE queries.
    ///   - scale: SDPA scale (typically `1/sqrt(D)`).
    /// - Returns: `[B, nQH, L, D]` attention output.
    public func prefillSparseAttend(
        queries: MLXArray, scale: Float
    ) -> MLXArray {
        precondition(queries.shape.count == 4,
            "queries must be [B, nQH, L, D]")
        let B = queries.dim(0)
        let nQH = queries.dim(1)
        let L = queries.dim(2)
        precondition(L > 1,
            "prefillSparseAttend requires L > 1; use sparseAttend at decode")
        precondition(B == index.B, "B mismatch")

        // Pull rectangular cache state — same assumption as `sparseAttend`.
        let off = inner.offsets[0]
        precondition(off >= L,
            "inner offset (\(off)) must include the chunk (L=\(L)) — caller " +
            "must write K/V before invoking prefillSparseAttend")
        let cachedK = inner.keys[..<B, 0..., ..<off, 0...]
        let cachedV = inner.values[..<B, 0..., ..<off, 0...]
        let nKVH = cachedK.dim(1)
        let priorLen = off - L

        // Fast path: nothing to gather. Run dense causal chunk-against-chunk.
        // Caller normally avoids this via the `sparsePrefillMinContext` gate.
        if priorLen <= 0 {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedK, values: cachedV,
                scale: scale, mask: .causal)
        }

        precondition(nQH % nKVH == 0,
            "Q heads (\(nQH)) must be a multiple of KV heads (\(nKVH))")
        let groupSize = nQH / nKVH

        // Build rep-per-group Q indices (lazy + reused across calls — same
        // pattern as `sparseAttend`).
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            eval(cachedHeadIdx!)
        }

        // [B, nQH, L, D] → take rep-per-KV-group → [B, nKVH, L, D]
        let qStacked = queries.take(cachedHeadIdx!, axis: 1).asType(.float32)
        let qProjL = index.projectQueriesBatchedL(qStacked)
        let fineStarts = index.topKFineBlockStartsUnionL(projectedQL: qProjL)
        let coarseStarts = index.topKCoarseBlockStartsUnionL(projectedQL: qProjL)

        // Static + sliding ranges are shared across slots (rectangular T).
        let staticEnd = min(raConfig.staticInit, priorLen)
        let slidingStart = max(0, priorLen - raConfig.slidingWindow)
        var staticSliding: [Int32] = []
        staticSliding.reserveCapacity(staticEnd + (priorLen - slidingStart))
        if staticEnd > 0 {
            for p in 0..<staticEnd { staticSliding.append(Int32(p)) }
        }
        if slidingStart < priorLen {
            for p in slidingStart..<priorLen { staticSliding.append(Int32(p)) }
        }

        let fineBS = raConfig.fineBlockSize
        let coarseBS = raConfig.coarseBlockSize
        let kFine = fineStarts.dim(2)
        let kCoarse = coarseStarts.dim(2)

        // Pull selector picks to CPU once per layer per chunk. CPU dedupe
        // + sort is on the order of (nKVH × kFine × fineBS) ints per slot
        // — tens of thousands per layer per chunk, negligible vs gather +
        // SDPA wall-clock at long context.
        let fineCpu = fineStarts.asArray(Int32.self)        // [B*nKVH*kFine]
        let coarseCpu = raConfig.coarseRescueEnabled
            ? coarseStarts.asArray(Int32.self) : [Int32]()  // [B*nKVH*kCoarse]
        let outputDtype = queries.dtype

        // Within-chunk causal mask is shared across slots.
        let neginf: Float = -.infinity
        let iRow = MLXArray(0..<Int32(L)).reshaped(L, 1)
        let iCol = MLXArray(0..<Int32(L)).reshaped(1, L)
        let chunkMaskTemplate = MLX.where(
            iCol .<= iRow,
            MLXArray(Float(0)),
            MLXArray(neginf)
        ).asType(outputDtype)  // [L, L]

        var perSlotOutputs: [MLXArray] = []
        perSlotOutputs.reserveCapacity(B)
        for slot in 0..<B {
            var union = Set<Int32>(staticSliding)
            // Cross-head fine union for this slot.
            let fineBase = slot * nKVH * kFine
            for h in 0..<nKVH {
                let hBase = fineBase + h * kFine
                for kk in 0..<kFine {
                    let start = fineCpu[hBase + kk]
                    if start < 0 { continue }
                    let startI = Int(start)
                    let endExc = min(startI + fineBS, priorLen)
                    if endExc <= 0 { continue }
                    let s = max(0, startI)
                    for p in s..<endExc { union.insert(Int32(p)) }
                }
            }
            if raConfig.coarseRescueEnabled {
                let coarseBase = slot * nKVH * kCoarse
                for h in 0..<nKVH {
                    let hBase = coarseBase + h * kCoarse
                    for kk in 0..<kCoarse {
                        let start = coarseCpu[hBase + kk]
                        if start < 0 { continue }
                        let startI = Int(start)
                        let endExc = min(startI + coarseBS, priorLen)
                        if endExc <= 0 { continue }
                        let s = max(0, startI)
                        for p in s..<endExc { union.insert(Int32(p)) }
                    }
                }
            }

            // Sort + materialize positions tensor.
            let positions = union.sorted()
            let P = positions.count
            let posArr = MLXArray(positions)

            // Gather prior K/V at `positions`; concat with chunk's own.
            let priorK = cachedK[slot ..< (slot + 1), 0..., ..<priorLen, 0...]
            let priorV = cachedV[slot ..< (slot + 1), 0..., ..<priorLen, 0...]
            let gK = priorK.take(posArr, axis: 2)            // [1, nKVH, P, D]
            let gV = priorV.take(posArr, axis: 2)
            let chunkK = cachedK[slot ..< (slot + 1), 0..., priorLen..., 0...]
            let chunkV = cachedV[slot ..< (slot + 1), 0..., priorLen..., 0...]
            let combinedK = concatenated([gK, chunkK], axis: 2)  // [1, nKVH, P+L, D]
            let combinedV = concatenated([gV, chunkV], axis: 2)

            // Combined mask [1, 1, L, P+L]: prior all-zero, chunk causal.
            let priorMask = MLXArray.zeros([L, P], dtype: outputDtype)
            let combinedMask = concatenated(
                [priorMask, chunkMaskTemplate], axis: 1
            ).reshaped(1, 1, L, P + L)
            let qSlot = queries[slot ..< (slot + 1), 0..., 0..., 0...]
            let oSlot = MLXFast.scaledDotProductAttention(
                queries: qSlot, keys: combinedK, values: combinedV,
                scale: scale, mask: .array(combinedMask)
            )
            perSlotOutputs.append(oSlot)
        }
        return concatenated(perSlotOutputs, axis: 0)
    }
}
