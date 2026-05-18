// SPDX-License-Identifier: Apache-2.0
// Sparse PREFILL kernel correctness tests for BatchedRetrievalAttentionKVCache.
//
// Asserts:
//   - At small prior cache (< sparsePrefillMinContext / fully covered by
//     static+sliding) sparse prefill output matches dense causal SDPA
//     bit-for-bit (cosine ≥ 0.999). Validates gather + concat + mask
//     wiring without quality risk.
//   - Per-slot output differs when Q differs across slots (proves per-slot
//     selector + gather flows through, not a constant).
//   - BatchedKVCache.updateChunk correctly writes L>1 K/V at slot offsets.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Batched sparse PREFILL (L > 1)")
struct BatchedSparsePrefillTests {

    /// Build a `RetrievalAttentionConfig` tuned for prefill smoke tests:
    /// small block sizes so a synthetic prior fits in a few blocks,
    /// sparse-everywhere (no dense bands).
    private func makeCfg(
        fineBS: Int = 32,
        fineTopK: Int = 4,
        staticInit: Int = 32,
        slidingWindow: Int = 64,
        sparsePrefillMinContext: Int = 0
    ) -> RetrievalAttentionConfig {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = fineBS
        cfg.fineTopK = fineTopK
        cfg.adaptiveTopK = false
        cfg.coarseRescueEnabled = false
        cfg.staticInit = staticInit
        cfg.slidingWindow = slidingWindow
        cfg.denseFirstN = 0
        cfg.denseLastN = 0
        cfg.sparsePrefillEnabled = true
        cfg.sparsePrefillMinContext = sparsePrefillMinContext
        return cfg
    }

    /// Plumbing test: at tiny prior (≤ staticInit + slidingWindow) the
    /// union of static + sliding covers all prior positions, so sparse
    /// output MUST be bit-identical to dense causal. Validates gather +
    /// concat + mask wiring without quality risk.
    @Test func prefillSparseAttendEqualsDenseAtSmallPrior() {
        let cfg = makeCfg()
        let B = 2
        let nKVH = 2
        let nQH = 4     // groupSize = 2
        let dHead = 32
        let prior = 32 + 64  // = staticInit + slidingWindow → fully covered
        let L = 4
        let totalT = prior + L
        let maxSeq = totalT + 16

        // Inner cache.
        let inner = BatchedKVCache(
            maxBatch: B, kvHeads: nKVH, headDim: dHead,
            maxSeq: maxSeq, dtype: .float32)
        for _ in 0..<B { _ = inner.addRequest() }

        // Prefill the prior portion (post-RoPE-equivalent fake K/V).
        MLXRandom.seed(0xF830AAA)
        let priorK = MLXRandom.normal([B, nKVH, prior, dHead]).asType(.float32)
        let priorV = MLXRandom.normal([B, nKVH, prior, dHead]).asType(.float32)
        inner.updateChunk(newKeys: priorK, newValues: priorV)
        #expect(inner.offsets[0] == prior, "prior write should advance offset to \(prior)")
        #expect(inner.offsets[1] == prior, "slot 1 offset should also be \(prior)")

        let raCache = BatchedRetrievalAttentionKVCache(
            inner: inner, B: B, nKVHeads: nKVH, dHead: dHead,
            layerIdx: 5, totalLayers: 10, raConfig: cfg)
        // Feed selector with the prior chunk (block features built).
        raCache.updateIndex(newKeys: priorK)

        // Now write the prefill chunk K/V (the queries' own K/V) into cache.
        let chunkK = MLXRandom.normal([B, nKVH, L, dHead]).asType(.float32)
        let chunkV = MLXRandom.normal([B, nKVH, L, dHead]).asType(.float32)
        inner.updateChunk(newKeys: chunkK, newValues: chunkV)
        #expect(inner.offsets[0] == totalT, "chunk write should advance offset to \(totalT)")
        raCache.updateIndex(newKeys: chunkK)

        let chunkQ = MLXRandom.normal([B, nQH, L, dHead]).asType(.float32)
        let scale = 1.0 / sqrt(Float(dHead))

        // Dense reference: full causal SDPA over all (prior + chunk).
        let allK = inner.keys[..<B, 0..., ..<totalT, 0...]
        let allV = inner.values[..<B, 0..., ..<totalT, 0...]
        // Dense mask: chunk queries can attend to all prior + lower-tri
        // within chunk. The dense fall-through in the model uses
        // getCachedWithMask + causal; we replicate that here so the
        // comparison is apples-to-apples.
        let iRow = MLXArray(0..<Int32(L)).reshaped(L, 1)
        let iCol = MLXArray(0..<Int32(L)).reshaped(1, L)
        let neginf: Float = -.infinity
        let chunkCausal = MLX.where(
            iCol .<= iRow,
            MLXArray(Float(0)),
            MLXArray(neginf)
        ).asType(allK.dtype)
        let priorMask = MLXArray.zeros([L, prior], dtype: allK.dtype)
        let combinedMask = concatenated([priorMask, chunkCausal], axis: 1)
            .reshaped(1, 1, L, totalT)
        let denseOut = MLXFast.scaledDotProductAttention(
            queries: chunkQ, keys: allK, values: allV,
            scale: scale, mask: .array(combinedMask)
        )

        let sparseOut = raCache.prefillSparseAttend(
            queries: chunkQ, scale: scale)
        eval(denseOut, sparseOut)

        #expect(denseOut.shape == sparseOut.shape,
            "shape mismatch dense \(denseOut.shape) vs sparse \(sparseOut.shape)")
        let cosineVal = cosine(denseOut, sparseOut)
        // staticInit + slidingWindow fully cover the prior — sparse should
        // pick the SAME positions as dense (after dedupe). Allow tiny
        // float drift from gather reorder.
        #expect(cosineVal >= 0.999,
            "small-prior sparse should match dense; got cosine=\(cosineVal)")
    }

    /// Per-slot sanity: different Q per slot → different sparse output.
    /// Prevents "constant fallback" regressions where the per-slot
    /// gather/selector silently collapses across the batch.
    @Test func prefillSparseAttendPerSlotOutputsDiffer() {
        let cfg = makeCfg(
            fineBS: 32, fineTopK: 4,
            staticInit: 32, slidingWindow: 64
        )
        let B = 2
        let nKVH = 2
        let nQH = 4
        let dHead = 32
        let prior = 32 + 64       // fully-covered prior (same as plumbing)
        let L = 4
        let totalT = prior + L
        let maxSeq = totalT + 16

        let inner = BatchedKVCache(
            maxBatch: B, kvHeads: nKVH, headDim: dHead,
            maxSeq: maxSeq, dtype: .float32)
        for _ in 0..<B { _ = inner.addRequest() }

        // SAME K/V across both slots → output difference must come from Q.
        MLXRandom.seed(0xF830BBB)
        let oneKp = MLXRandom.normal([1, nKVH, prior, dHead]).asType(.float32)
        let oneVp = MLXRandom.normal([1, nKVH, prior, dHead]).asType(.float32)
        let kPrior = concatenated([oneKp, oneKp], axis: 0)
        let vPrior = concatenated([oneVp, oneVp], axis: 0)
        inner.updateChunk(newKeys: kPrior, newValues: vPrior)

        let raCache = BatchedRetrievalAttentionKVCache(
            inner: inner, B: B, nKVHeads: nKVH, dHead: dHead,
            layerIdx: 5, totalLayers: 10, raConfig: cfg)
        raCache.updateIndex(newKeys: kPrior)

        let oneKc = MLXRandom.normal([1, nKVH, L, dHead]).asType(.float32)
        let oneVc = MLXRandom.normal([1, nKVH, L, dHead]).asType(.float32)
        let kChunk = concatenated([oneKc, oneKc], axis: 0)
        let vChunk = concatenated([oneVc, oneVc], axis: 0)
        inner.updateChunk(newKeys: kChunk, newValues: vChunk)
        raCache.updateIndex(newKeys: kChunk)

        // DIFFERENT Q per slot.
        let q0 = MLXRandom.normal([1, nQH, L, dHead], key: MLXRandom.key(701))
            .asType(.float32)
        let q1 = MLXRandom.normal([1, nQH, L, dHead], key: MLXRandom.key(702))
            .asType(.float32)
        let Q = concatenated([q0, q1], axis: 0)
        let scale = 1.0 / sqrt(Float(dHead))
        let out = raCache.prefillSparseAttend(queries: Q, scale: scale)
        eval(out)
        #expect(out.shape == [B, nQH, L, dHead], "shape contract")
        let slot0 = out[0, 0..., 0..., 0...]
        let slot1 = out[1, 0..., 0..., 0...]
        let diff = (slot0 - slot1).abs().max().item(Float.self)
        #expect(diff > 1e-3,
            "expected per-slot outputs to differ (same K, different Q); got max-abs-diff=\(diff)")
    }

    /// BatchedKVCache.updateChunk smoke: writes L>1 contiguously at the
    /// per-slot offset and advances by L. Same-offset (rectangular) +
    /// ragged-offset paths both exercised.
    @Test func batchedKVCacheUpdateChunkRectangular() {
        let B = 2
        let kvH = 2
        let dHead = 16
        let maxSeq = 128
        let cache = BatchedKVCache(
            maxBatch: B, kvHeads: kvH, headDim: dHead,
            maxSeq: maxSeq, dtype: .float32)
        for _ in 0..<B { _ = cache.addRequest() }
        let L = 5
        MLXRandom.seed(0xC4C4E)
        let k0 = MLXRandom.normal([B, kvH, L, dHead]).asType(.float32)
        let v0 = MLXRandom.normal([B, kvH, L, dHead]).asType(.float32)
        cache.updateChunk(newKeys: k0, newValues: v0)
        #expect(cache.offsets[0] == L && cache.offsets[1] == L,
            "rectangular updateChunk advances all slots by L")
        // Verify k0 lives at [0..L] in the buffer.
        let written = cache.keys[..<B, 0..., ..<L, 0...]
        let delta = (written - k0).abs().max().item(Float.self)
        #expect(delta < 1e-6, "updateChunk write fidelity (got delta=\(delta))")

        // Second chunk lands at [L..2L].
        let k1 = MLXRandom.normal([B, kvH, L, dHead]).asType(.float32)
        let v1 = MLXRandom.normal([B, kvH, L, dHead]).asType(.float32)
        cache.updateChunk(newKeys: k1, newValues: v1)
        #expect(cache.offsets[0] == 2 * L && cache.offsets[1] == 2 * L,
            "second updateChunk advances to 2L")
        let written2 = cache.keys[..<B, 0..., L ..< (2 * L), 0...]
        let delta2 = (written2 - k1).abs().max().item(Float.self)
        #expect(delta2 < 1e-6, "second updateChunk write fidelity (got delta=\(delta2))")
    }

    @Test func batchedKVCacheUpdateChunkRagged() {
        let B = 2
        let kvH = 2
        let dHead = 16
        let maxSeq = 128
        let cache = BatchedKVCache(
            maxBatch: B, kvHeads: kvH, headDim: dHead,
            maxSeq: maxSeq, dtype: .float32)
        for _ in 0..<B { _ = cache.addRequest() }
        // Set slot 0 to offset 3, slot 1 to offset 7 — different starting
        // points. Then write an L=4 chunk to each.
        cache.setOffset(0, 3)
        cache.setOffset(1, 7)
        let L = 4
        MLXRandom.seed(0xC4F33)
        let k = MLXRandom.normal([B, kvH, L, dHead]).asType(.float32)
        let v = MLXRandom.normal([B, kvH, L, dHead]).asType(.float32)
        cache.updateChunk(newKeys: k, newValues: v)
        #expect(cache.offsets[0] == 3 + L, "slot 0 offset advances by L")
        #expect(cache.offsets[1] == 7 + L, "slot 1 offset advances by L")
        let w0 = cache.keys[0, 0..., 3 ..< (3 + L), 0...]
        let w1 = cache.keys[1, 0..., 7 ..< (7 + L), 0...]
        let d0 = (w0 - k[0, 0..., 0..., 0...]).abs().max().item(Float.self)
        let d1 = (w1 - k[1, 0..., 0..., 0...]).abs().max().item(Float.self)
        #expect(d0 < 1e-6 && d1 < 1e-6,
            "ragged updateChunk per-slot write fidelity (got d0=\(d0), d1=\(d1))")
    }

    // MARK: - helpers

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let aF = a.reshaped(a.size).asType(.float32)
        let bF = b.reshaped(b.size).asType(.float32)
        let dot = (aF * bF).sum().item(Float.self)
        let an = sqrt((aF * aF).sum()).item(Float.self)
        let bn = sqrt((bF * bF).sum()).item(Float.self)
        return dot / (an * bn + 1e-12)
    }
}
