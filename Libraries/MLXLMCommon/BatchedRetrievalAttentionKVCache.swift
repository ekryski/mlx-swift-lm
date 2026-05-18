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
}
