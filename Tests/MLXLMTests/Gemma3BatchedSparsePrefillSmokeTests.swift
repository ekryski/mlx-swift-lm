// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Gemma3 batched-sparse PREFILL path.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Gemma3 batched-sparse PREFILL synthetic smoke", .serialized)
struct Gemma3BatchedSparsePrefillSmokeTests {

    private func makeTinyConfig() -> Gemma3.TextConfiguration {
        Gemma3.TextConfiguration(
            modelType: "gemma3_text",
            hiddenSize: 128,
            hiddenLayers: 4,
            intermediateSize: 256,
            attentionHeads: 4,
            headDim: 32,
            rmsNormEps: 1e-5,
            vocabularySize: 256,
            kvHeads: 2,
            ropeTheta: 10_000.0,
            ropeLocalBaseFreq: 10_000.0,
            ropeTraditional: false,
            queryPreAttnScalar: 1.0,
            slidingWindow: 64,
            slidingWindowPattern: 1,
            maxPositionEmbeddings: 4096)
    }

    @Test func sparsePrefillForwardShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Gemma3TextModel(cfg)
        eval(model)

        let nLayers = 4
        let kvHeads = 2
        let headDim = 32
        let vocab = 256

        let B = 2
        let maxSeq = 512

        var raCfg = RetrievalAttentionConfig()
        raCfg.fineBlockSize = 32
        raCfg.coarseRescueEnabled = false
        raCfg.adaptiveTopK = false
        raCfg.fineTopK = 2
        raCfg.staticInit = 32
        raCfg.slidingWindow = 64
        raCfg.denseFirstN = 1
        raCfg.denseLastN = 1
        raCfg.sparseMinContext = 0
        raCfg.sparsePrefillEnabled = true
        raCfg.sparsePrefillMinContext = 0

        let raCaches: [BatchedRetrievalAttentionKVCache] = (0..<nLayers).map { layer in
            let inner = BatchedKVCache(
                maxBatch: B, kvHeads: kvHeads, headDim: headDim,
                maxSeq: maxSeq, dtype: .float32)
            for _ in 0..<B { _ = inner.addRequest() }
            return BatchedRetrievalAttentionKVCache(
                inner: inner, B: B, nKVHeads: kvHeads, dHead: headDim,
                layerIdx: layer, totalLayers: nLayers, raConfig: raCfg)
        }

        let priorT = 128
        for raCache in raCaches {
            let inner = raCache.inner
            let kFill = MLXRandom.normal(
                [B, kvHeads, priorT, headDim],
                key: MLXRandom.key(UInt64(raCache.layerIdx + 1))
            ).asType(.float32)
            let vFill = MLXRandom.normal(
                [B, kvHeads, priorT, headDim],
                key: MLXRandom.key(UInt64(raCache.layerIdx + 101))
            ).asType(.float32)
            inner.keys[..<B, 0..., ..<priorT, 0...] = kFill
            inner.values[..<B, 0..., ..<priorT, 0...] = vFill
            for i in 0..<B { inner.offsets[i] = priorT }
            raCache.updateIndex(newKeys: kFill)
        }

        let chunkL = 16
        let tokens = MLXRandom.randInt(
            low: 0, high: vocab,
            [B, chunkL]).asType(.int32)
        let out = model.model.fullyBatchedSparseForward(tokens, raCaches: raCaches)
        let logits = model.lmHead(out)
        eval(logits)
        #expect(logits.shape == [B, chunkL, vocab],
            "prefill forward output shape")
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite }, "prefill logits must be finite")
        for raCache in raCaches {
            #expect(raCache.inner.offsets[0] == priorT + chunkL)
            #expect(raCache.inner.offsets[1] == priorT + chunkL)
        }
    }
}
