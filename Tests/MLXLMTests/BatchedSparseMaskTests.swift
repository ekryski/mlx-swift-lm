// SPDX-License-Identifier: Apache-2.0
// Batched sparse mask kernel + sparseAttend path-equivalence tests.
//
// Asserts: per-slot batched mask matches per-slot single-batch mask;
// mask-kernel path matches loop path numerically; compose-gather path
// matches the group-SDPA kernel path numerically (same selector scope).

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Batched sparse mask kernel + path equivalence")
struct BatchedSparseMaskTests {

    /// Populating the selector index from a chunk of K should let the
    /// top-K return non-trivial block starts. Picking blocks 0..k-1 by
    /// index order (the placeholder pattern) would mean
    /// `topKFineBlockStarts` returns [0, fineBS, 2*fineBS, ...] regardless
    /// of Q. After real population, picked starts should differ across
    /// slots with different K.
    @Test func selectorPicksDifferentBlocksPerSlotAfterPopulation() {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 64
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.fineTopK = 4
        cfg.staticInit = 64
        cfg.slidingWindow = 128
        cfg.denseFirstN = 0
        cfg.denseLastN = 0

        let B = 2
        let nKVH = 4
        let dHead = 64
        let T = 2048   // 32 fine blocks (way more than fineTopK=4)

        // DIFFERENT K per slot so the optimal blocks differ.
        let k0 = MLXRandom.normal([1, nKVH, T, dHead], key: MLXRandom.key(41)).asType(.float32)
        let k1 = MLXRandom.normal([1, nKVH, T, dHead], key: MLXRandom.key(42)).asType(.float32)
        let kAll = concatenated([k0, k1], axis: 0)
        let vAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(43)).asType(.float32)

        let cache = BatchedKVCache(
            maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
            dtype: .float32)
        for _ in 0..<B { _ = cache.addRequest() }
        cache.keys[..<B, 0..., ..<T, 0...] = kAll
        cache.values[..<B, 0..., ..<T, 0...] = vAll
        for i in 0..<B { cache.offsets[i] = T }

        let raCache = BatchedRetrievalAttentionKVCache(
            inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
            layerIdx: 5, totalLayers: 10, raConfig: cfg)
        raCache.index.update(newKeys: kAll)

        let qRep = MLXRandom.normal([B, nKVH, dHead], key: MLXRandom.key(44)).asType(.float32)
        let projQ = raCache.index.projectQueriesBatched(qRep)
        let fineStarts = raCache.index.topKFineBlockStarts(projectedQ: projQ)
        eval(fineStarts)
        #expect(fineStarts.shape == [B, nKVH, cfg.fineTopK])

        // Across SOME (slot, head) pairs the picked starts must NOT be
        // [0, 64, 128, 192] (the placeholder index-order pattern).
        var pickedDifferentFromIndexOrder = false
        let starts = fineStarts.asArray(Int32.self)
        let perHead = cfg.fineTopK
        outer: for b in 0..<B {
            for h in 0..<nKVH {
                let base = b * nKVH * perHead + h * perHead
                let slot = Array(starts[base ..< base + perHead]).sorted()
                let placeholder = (0..<perHead).map { Int32($0 * cfg.fineBlockSize) }
                if slot != placeholder {
                    pickedDifferentFromIndexOrder = true
                    break outer
                }
            }
        }
        #expect(pickedDifferentFromIndexOrder,
            "after selector population, selector must pick non-placeholder blocks")
    }

    /// The batched build-mask kernel must emit a mask whose per-slot
    /// slices equal the single-batch mask for the same inputs.
    @Test func batchedMaskMatchesPerSlotLoop() {
        let B = 3
        let nKVH = 4
        let kFine = 2
        let kCoarse = 2
        let T = 512
        let staticInit = 64
        let slidingWindow = 128
        let fineBS = 32
        let coarseBS = 128

        let fineStarts = MLXArray(
            stride(from: Int32(0), to: Int32(B * nKVH * kFine), by: 1).map { $0 % Int32(T / fineBS) * Int32(fineBS) }
        ).reshaped(B, nKVH, kFine)
        let coarseStarts = MLXArray(
            stride(from: Int32(0), to: Int32(B * nKVH * kCoarse), by: 1).map { $0 % Int32(T / coarseBS) * Int32(coarseBS) }
        ).reshaped(B, nKVH, kCoarse)

        // Batched mask in one launch.
        let maskBatched = retrievalAttentionBuildMaskFusedBatched(
            fineStarts: fineStarts, coarseStarts: coarseStarts,
            T: T, staticInit: staticInit, slidingWindow: slidingWindow,
            fineBS: fineBS, coarseBS: coarseBS, outputDtype: .float32
        )
        #expect(maskBatched.shape == [B, 1, 1, T], "batched mask shape")

        // Per-slot single-batch masks.
        for b in 0..<B {
            let fineB = fineStarts[b, 0..., 0...]
            let coarseB = coarseStarts[b, 0..., 0...]
            let maskSingle = retrievalAttentionBuildMaskFused(
                fineStarts: fineB, coarseStarts: coarseB,
                T: T, staticInit: staticInit, slidingWindow: slidingWindow,
                fineBS: fineBS, coarseBS: coarseBS, outputDtype: .float32
            )
            #expect(maskSingle.shape == [1, 1, 1, T])
            let slotBatched = maskBatched[b ..< (b + 1), 0..., 0..., 0...]
            let diff = (slotBatched - maskSingle).abs().max().item(Float.self)
            #expect(diff == 0.0, "slot \(b) batched-vs-loop mask diff = \(diff)")
        }
    }

    /// Batched mask path and Swift-loop single-batch mask path must
    /// produce numerically close outputs for the same inputs. The selector
    /// picks the SAME blocks for both paths (same projQ, same features)
    /// so the gathered K/V support is identical.
    @Test func batchedMaskAndLoopAgreeOnOutput() {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 32
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.fineTopK = 4
        cfg.staticInit = 64
        cfg.slidingWindow = 128
        cfg.denseFirstN = 0
        cfg.denseLastN = 0

        let B = 3
        let nKVH = 4
        let nQH = 8     // groupSize = 2
        let dHead = 64
        let T = 512

        let kAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(11)).asType(.float32)
        let vAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(12)).asType(.float32)
        let Q = MLXRandom.normal([B, nQH, 1, dHead], key: MLXRandom.key(13)).asType(.float32)

        let scale = pow(Float(dHead), -0.5)

        // Helper to build a fresh cache + raCache with these tensors.
        func makeRACache() -> BatchedRetrievalAttentionKVCache {
            let cache = BatchedKVCache(
                maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
                dtype: .float32)
            for _ in 0..<B { _ = cache.addRequest() }
            cache.keys[..<B, 0..., ..<T, 0...] = kAll
            cache.values[..<B, 0..., ..<T, 0...] = vAll
            for i in 0..<B { cache.offsets[i] = T }
            let raCache = BatchedRetrievalAttentionKVCache(
                inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
                layerIdx: 5, totalLayers: 10, raConfig: cfg)
            raCache.updateIndex(newKeys: kAll)
            return raCache
        }

        // Path A — batched mask kernel (default).
        let raA = makeRACache()
        let outA: MLXArray = {
            let qSqueezed = Q[0..., 0..., 0, 0...]
            let groupSize = nQH / nKVH
            let headIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            let qRep = qSqueezed.take(headIdx, axis: 1).asType(.float32)
            let projQ = raA.index.projectQueriesBatched(qRep)
            let fineStarts = raA.index.topKFineBlockStarts(projectedQ: projQ)
            let coarseStarts = raA.index.topKCoarseBlockStarts(projectedQ: projQ)
            let mask = retrievalAttentionBuildMaskFusedBatched(
                fineStarts: fineStarts, coarseStarts: coarseStarts,
                T: T, staticInit: cfg.staticInit, slidingWindow: cfg.slidingWindow,
                fineBS: cfg.fineBlockSize, coarseBS: cfg.coarseBlockSize,
                outputDtype: kAll.dtype
            )
            return MLXFast.scaledDotProductAttention(
                queries: Q, keys: kAll, values: vAll,
                scale: scale, mask: .array(mask))
        }()
        eval(outA)

        // Path B — Swift loop over per-slot mask kernel + SDPA.
        let raB = makeRACache()
        let outB: MLXArray = {
            let qSqueezed = Q[0..., 0..., 0, 0...]
            let groupSize = nQH / nKVH
            let headIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            let qRep = qSqueezed.take(headIdx, axis: 1).asType(.float32)
            let projQ = raB.index.projectQueriesBatched(qRep)
            let fineStarts = raB.index.topKFineBlockStarts(projectedQ: projQ)
            let coarseStarts = raB.index.topKCoarseBlockStarts(projectedQ: projQ)
            var outs: [MLXArray] = []
            for b in 0..<B {
                let fineB = fineStarts[b, 0..., 0...]
                let coarseB = coarseStarts[b, 0..., 0...]
                let mask = retrievalAttentionBuildMaskFused(
                    fineStarts: fineB, coarseStarts: coarseB,
                    T: T, staticInit: cfg.staticInit, slidingWindow: cfg.slidingWindow,
                    fineBS: cfg.fineBlockSize, coarseBS: cfg.coarseBlockSize,
                    outputDtype: kAll.dtype
                )
                let q = Q[b ..< (b + 1), 0..., 0..., 0...]
                let k = kAll[b ..< (b + 1), 0..., 0..., 0...]
                let v = vAll[b ..< (b + 1), 0..., 0..., 0...]
                outs.append(MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v,
                    scale: scale, mask: .array(mask)))
            }
            return concatenated(outs, axis: 0)
        }()
        eval(outB)

        // Same mask, same SDPA — should be numerically identical.
        let diffAB = (outA - outB).abs().max().item(Float.self)
        #expect(diffAB < 1e-4, "batched-mask vs loop-mask diff=\(diffAB)")
    }

    /// Compose-gather (per-slot K/V materialize then dense SDPA) should
    /// match the group-SDPA kernel numerically: same selector, same
    /// gathered K/V scope, just different SDPA primitives.
    @Test func composeGatherMatchesGroupSDPA() {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 32
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.fineTopK = 4
        cfg.staticInit = 64
        cfg.slidingWindow = 128
        cfg.denseFirstN = 0
        cfg.denseLastN = 0

        let B = 2
        let nKVH = 4
        let nQH = 8     // groupSize = 2
        let dHead = 64
        let T = 512

        let kAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(21)).asType(.float32)
        let vAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(22)).asType(.float32)
        let Q = MLXRandom.normal([B, nQH, 1, dHead], key: MLXRandom.key(23)).asType(.float32)

        let scale = pow(Float(dHead), -0.5)

        func makeRACache() -> BatchedRetrievalAttentionKVCache {
            let cache = BatchedKVCache(
                maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
                dtype: .float32)
            for _ in 0..<B { _ = cache.addRequest() }
            cache.keys[..<B, 0..., ..<T, 0...] = kAll
            cache.values[..<B, 0..., ..<T, 0...] = vAll
            for i in 0..<B { cache.offsets[i] = T }
            let raCache = BatchedRetrievalAttentionKVCache(
                inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
                layerIdx: 5, totalLayers: 10, raConfig: cfg)
            raCache.updateIndex(newKeys: kAll)
            return raCache
        }

        // Common Q-rep + projQ setup (mirrors sparseAttend internals).
        let qSqueezed = Q[0..., 0..., 0, 0...]
        let groupSize = nQH / nKVH
        let headIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
        let qRep = qSqueezed.take(headIdx, axis: 1).asType(.float32)

        // Reference path: group-SDPA custom kernel.
        let raRef = makeRACache()
        let projQRef = raRef.index.projectQueriesBatched(qRep)
        let (gatherRef, _) = raRef.index.perKVHeadGatherBatched(
            projectedQ: projQRef, seqLen: T)
        let outRef = retrievalAttentionGroupSparseSDPA(
            queries: Q,
            keys: kAll,
            values: vAll,
            perKVHeadGather: gatherRef,
            scale: scale)
        eval(outRef)

        // Compose-gather path: same selector, MLXFast SDPA on gathered slab.
        let raCG = makeRACache()
        let projQCG = raCG.index.projectQueriesBatched(qRep)
        let (gather, _) = raCG.index.perKVHeadGatherBatched(
            projectedQ: projQCG, seqLen: T)
        let gExp = gather.expandedDimensions(axis: 3)
        let kSmall = takeAlong(kAll, gExp, axis: 2)
        let vSmall = takeAlong(vAll, gExp, axis: 2)
        let outCG = MLXFast.scaledDotProductAttention(
            queries: Q, keys: kSmall, values: vSmall,
            scale: scale, mask: .none)
        eval(outCG)

        // Both paths attend the SAME per-(B, nKVH) positions, so the only
        // numerical difference is the SDPA primitive. Tight cosine.
        for b in 0..<B {
            for h in 0..<nQH {
                let a = outRef[b, h, 0, 0...].asArray(Float.self)
                let c = outCG[b, h, 0, 0...].asArray(Float.self)
                var dot: Float = 0
                var na: Float = 0
                var nc: Float = 0
                for i in 0..<a.count {
                    dot += a[i] * c[i]
                    na += a[i] * a[i]
                    nc += c[i] * c[i]
                }
                let cos = dot / (sqrt(na) * sqrt(nc) + 1e-9)
                #expect(cos >= 0.99,
                    "compose-gather vs group-SDPA cosine = \(cos) at (b=\(b), h=\(h))")
            }
        }
    }

    /// Compose-gather output should be a reasonable approximation to
    /// dense at high K_padded coverage. At small T with kFine=8, fineBS=32,
    /// static=64, sliding=128 the gather covers ~448 of 512 positions = 88%
    /// → cosine vs dense should be well above 0.4 — guards against total
    /// wrong-axis bugs (slot crossover etc).
    @Test func composeGatherSanityVsDense() {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 32
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.fineTopK = 8     // 8 * 32 = 256 fine positions
        cfg.staticInit = 64
        cfg.slidingWindow = 128
        cfg.denseFirstN = 0
        cfg.denseLastN = 0

        let B = 2
        let nKVH = 4
        let nQH = 8
        let dHead = 64
        let T = 512

        let kAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(31)).asType(.float32)
        let vAll = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(32)).asType(.float32)
        let Q = MLXRandom.normal([B, nQH, 1, dHead], key: MLXRandom.key(33)).asType(.float32)
        let scale = pow(Float(dHead), -0.5)

        // Dense reference.
        let outDense = MLXFast.scaledDotProductAttention(
            queries: Q, keys: kAll, values: vAll,
            scale: scale, mask: .none)
        eval(outDense)

        // Compose-gather.
        let cache = BatchedKVCache(
            maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
            dtype: .float32)
        for _ in 0..<B { _ = cache.addRequest() }
        cache.keys[..<B, 0..., ..<T, 0...] = kAll
        cache.values[..<B, 0..., ..<T, 0...] = vAll
        for i in 0..<B { cache.offsets[i] = T }
        let raCache = BatchedRetrievalAttentionKVCache(
            inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
            layerIdx: 5, totalLayers: 10, raConfig: cfg)
        raCache.updateIndex(newKeys: kAll)

        let qSqueezed = Q[0..., 0..., 0, 0...]
        let groupSize = nQH / nKVH
        let headIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
        let qRep = qSqueezed.take(headIdx, axis: 1).asType(.float32)
        let projQ = raCache.index.projectQueriesBatched(qRep)
        let (gather, _) = raCache.index.perKVHeadGatherBatched(
            projectedQ: projQ, seqLen: T)
        let gExp = gather.expandedDimensions(axis: 3)
        let kSmall = takeAlong(kAll, gExp, axis: 2)
        let vSmall = takeAlong(vAll, gExp, axis: 2)
        let outCG = MLXFast.scaledDotProductAttention(
            queries: Q, keys: kSmall, values: vSmall,
            scale: scale, mask: .none)
        eval(outCG)

        var minCos: Float = 1.0
        for b in 0..<B {
            for h in 0..<nQH {
                let a = outDense[b, h, 0, 0...].asArray(Float.self)
                let c = outCG[b, h, 0, 0...].asArray(Float.self)
                var dot: Float = 0
                var na: Float = 0
                var nc: Float = 0
                for i in 0..<a.count {
                    dot += a[i] * c[i]
                    na += a[i] * a[i]
                    nc += c[i] * c[i]
                }
                let cos = dot / (sqrt(na) * sqrt(nc) + 1e-9)
                if cos < minCos { minCos = cos }
            }
        }
        #expect(minCos >= 0.4,
            "compose-gather vs dense min cosine = \(minCos) — too low, possible slot crossover")
    }
}
