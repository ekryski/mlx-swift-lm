// SPDX-License-Identifier: Apache-2.0
// Smoke tests for BatchedRetrievalAttentionKVCache + sparse decode kernel.
//
// Asserts: shape contract holds + per-slot outputs differ when Q differs
// across slots (proves the selector + sparseAttend path is per-slot real).

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("BatchedRetrievalAttentionKVCache (group-SDPA feeder)")
struct BatchedRetrievalAttentionKVCacheTests {

    /// Build a rectangular BatchedKVCache with T tokens of random fp32
    /// K/V written into the first T slots of each (B, nKVH).
    private func makeCacheWithT(
        B: Int, nKVH: Int, T: Int, dHead: Int, seed: UInt64 = 17
    ) -> BatchedKVCache {
        let cache = BatchedKVCache(
            maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
            dtype: .float32)
        for _ in 0..<B { _ = cache.addRequest() }
        // Fill with one big batched chunk by writing the K/V tensors
        // directly — bypasses the per-token update path.
        let kAll = MLXRandom.normal(
            [B, nKVH, T, dHead], key: MLXRandom.key(seed)
        ).asType(.float32)
        let vAll = MLXRandom.normal(
            [B, nKVH, T, dHead], key: MLXRandom.key(seed &+ 1)
        ).asType(.float32)
        cache.keys[..<B, 0..., ..<T, 0...] = kAll
        cache.values[..<B, 0..., ..<T, 0...] = vAll
        for i in 0..<B { cache.offsets[i] = T }
        return cache
    }

    @Test func b2SmokeShape() {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 64
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.fineTopK = 4
        cfg.staticInit = 128
        cfg.slidingWindow = 256
        cfg.denseFirstN = 0
        cfg.denseLastN = 0
        let B = 2
        let nKVH = 2
        let nQH = 4   // groupSize = 2
        let dHead = 64
        let T = 1024
        let raCache = BatchedRetrievalAttentionKVCache(
            inner: makeCacheWithT(B: B, nKVH: nKVH, T: T, dHead: dHead),
            B: B, nKVHeads: nKVH, dHead: dHead,
            layerIdx: 5, totalLayers: 10, raConfig: cfg)
        // Populate selector index via prefill update with same K layout.
        let kPrefill = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(101)).asType(.float32)
        raCache.updateIndex(newKeys: kPrefill)

        let Q = MLXRandom.normal([B, nQH, 1, dHead], key: MLXRandom.key(202))
        let out = raCache.sparseAttend(queries: Q, scale: pow(Float(dHead), -0.5))
        #expect(out.shape == [B, nQH, 1, dHead], "sparseAttend output shape contract")
        eval(out)
    }

    @Test func b2PerSlotOutputsDiffer() {
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 64
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.fineTopK = 4
        cfg.staticInit = 128
        cfg.slidingWindow = 256
        cfg.denseFirstN = 0
        cfg.denseLastN = 0
        let B = 2
        let nKVH = 2
        let nQH = 4
        let dHead = 64
        let T = 1024
        // SAME K/V across slots so the only differentiator is Q.
        let oneK = MLXRandom.normal([1, nKVH, T, dHead], key: MLXRandom.key(33)).asType(.float32)
        let oneV = MLXRandom.normal([1, nKVH, T, dHead], key: MLXRandom.key(34)).asType(.float32)
        let kAll = concatenated([oneK, oneK], axis: 0)
        let vAll = concatenated([oneV, oneV], axis: 0)
        let cache = BatchedKVCache(
            maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
            dtype: .float32)
        for _ in 0..<B { _ = cache.addRequest() }
        cache.keys[..<B, 0..., ..<T, 0...] = kAll
        cache.values[..<B, 0..., ..<T, 0...] = vAll
        for i in 0..<B { cache.offsets[i] = T }

        let raCache = BatchedRetrievalAttentionKVCache(
            inner: cache,
            B: B, nKVHeads: nKVH, dHead: dHead,
            layerIdx: 5, totalLayers: 10, raConfig: cfg)
        // Selector index sees the same K as the cache.
        raCache.updateIndex(newKeys: kAll)

        // DIFFERENT Q per slot → different gather → different output.
        let q0 = MLXRandom.normal([1, nQH, 1, dHead], key: MLXRandom.key(901))
        let q1 = MLXRandom.normal([1, nQH, 1, dHead], key: MLXRandom.key(902))
        let Q = concatenated([q0, q1], axis: 0)
        let out = raCache.sparseAttend(queries: Q, scale: pow(Float(dHead), -0.5))
        eval(out)
        // The two slot outputs MUST differ — same K/V, different Q.
        let slot0 = out[0, 0..., 0, 0...]
        let slot1 = out[1, 0..., 0, 0...]
        let diff = (slot0 - slot1).abs().max().item(Float.self)
        #expect(diff > 1e-3, "expected slot-0 and slot-1 outputs to differ (diff=\(diff))")
    }
}
