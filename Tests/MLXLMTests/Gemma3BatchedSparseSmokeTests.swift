// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Gemma3 batched-sparse decode hook.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Gemma3 batched-sparse synthetic smoke")
struct Gemma3BatchedSparseSmokeTests {

    private func makeTinyConfig() -> Gemma3.TextConfiguration {
        // slidingWindowPattern=1 ⇒ every layer is global, which uses a single
        // shared K shape across layers. queryPreAttnScalar=1 keeps the
        // softmax scale at 1.0 (the synthetic test only cares about finiteness).
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

    @Test func sparseDecodeShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Gemma3TextModel(cfg)
        eval(model)

        let nLayers = 4
        let kvHeads = 2
        let headDim = 32
        let vocab = 256

        let B = 2
        let maxSeq = 256

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

        let raCaches: [BatchedRetrievalAttentionKVCache] = (0..<nLayers).map { layer in
            let inner = BatchedKVCache(
                maxBatch: B, kvHeads: kvHeads, headDim: headDim,
                maxSeq: maxSeq, dtype: .float32)
            for _ in 0..<B { _ = inner.addRequest() }
            return BatchedRetrievalAttentionKVCache(
                inner: inner, B: B, nKVHeads: kvHeads, dHead: headDim,
                layerIdx: layer, totalLayers: nLayers, raConfig: raCfg)
        }

        let T0 = 128
        for raCache in raCaches {
            let inner = raCache.inner
            let kFill = MLXRandom.normal(
                [B, kvHeads, T0, headDim],
                key: MLXRandom.key(UInt64(raCache.layerIdx + 1))
            ).asType(.float32)
            let vFill = MLXRandom.normal(
                [B, kvHeads, T0, headDim],
                key: MLXRandom.key(UInt64(raCache.layerIdx + 101))
            ).asType(.float32)
            inner.keys[..<B, 0..., ..<T0, 0...] = kFill
            inner.values[..<B, 0..., ..<T0, 0...] = vFill
            for i in 0..<B { inner.offsets[i] = T0 }
            raCache.updateIndex(newKeys: kFill)
        }

        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, raCaches: raCaches)
        eval(logits)
        #expect(logits.shape == [B, 1, vocab],
            "fullyBatchedSparseDecode output shape")
        let asArr = logits.asArray(Float.self)
        let allFinite = asArr.allSatisfy { $0.isFinite }
        #expect(allFinite, "logits must be finite")
    }
}
