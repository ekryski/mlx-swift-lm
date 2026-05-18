// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Mistral3 batched-sparse PREFILL path.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Mistral3 batched-sparse PREFILL synthetic smoke", .serialized)
struct Mistral3BatchedSparsePrefillSmokeTests {

    private func makeTinyConfig() -> Mistral3TextConfiguration {
        Mistral3TextConfiguration(
            hiddenSize: 128,
            hiddenLayers: 4,
            intermediateSize: 256,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 256,
            headDimensions: 32,
            maxPositionEmbeddings: 4096,
            kvHeads: 2,
            ropeTheta: 10_000,
            ropeParameters: nil,
            tieWordEmbeddings: true,
            layerTypes: ["full_attention", "full_attention", "full_attention", "full_attention"],
            slidingWindow: nil)
    }

    @Test func sparsePrefillForwardShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Mistral3TextModel(cfg)
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
        var out = model.model.fullyBatchedSparseForward(tokens, raCaches: raCaches)
        if let lmHead = model.lmHead {
            out = lmHead(out)
        } else {
            out = model.model.embedTokens.asLinear(out)
        }
        eval(out)
        #expect(out.shape == [B, chunkL, vocab],
            "prefill forward output shape")
        let asArr = out.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite }, "prefill logits must be finite")
        for raCache in raCaches {
            #expect(raCache.inner.offsets[0] == priorT + chunkL)
            #expect(raCache.inner.offsets[1] == priorT + chunkL)
        }
    }
}
