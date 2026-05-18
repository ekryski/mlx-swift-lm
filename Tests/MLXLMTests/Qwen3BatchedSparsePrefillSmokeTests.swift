// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Qwen3 batched-sparse PREFILL path. Builds a tiny
// random-weight Qwen3 model in memory, reserves B slots in per-layer
// BatchedRetrievalAttentionKVCache, primes a synthetic prior cache, then
// runs an L>1 prefill chunk through `fullyBatchedSparseForward` with
// `sparsePrefillEnabled=true`. Asserts output shape + finite logits.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen3 batched-sparse PREFILL synthetic smoke", .serialized)
struct Qwen3BatchedSparsePrefillSmokeTests {

    private func makeTinyConfig() -> Qwen3Configuration {
        let json = """
        {
          "hidden_size": 128,
          "num_hidden_layers": 4,
          "intermediate_size": 256,
          "num_attention_heads": 4,
          "rms_norm_eps": 1e-5,
          "vocab_size": 256,
          "num_key_value_heads": 2,
          "rope_theta": 10000.0,
          "head_dim": 32,
          "tie_word_embeddings": true
        }
        """.data(using: .utf8)!
        return try! JSONDecoder().decode(Qwen3Configuration.self, from: json)
    }

    @Test func sparsePrefillForwardShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Qwen3Model(cfg)
        eval(model)

        let cfgData = try! JSONEncoder().encode(cfg)
        let cfgDict = try! JSONSerialization.jsonObject(with: cfgData) as! [String: Any]
        let nLayers = cfgDict["num_hidden_layers"] as! Int
        let kvHeads = cfgDict["num_key_value_heads"] as! Int
        let headDim = cfgDict["head_dim"] as! Int
        let vocab = cfgDict["vocab_size"] as! Int

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

        // Prime a synthetic prior in each layer cache.
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

        // Run an L=16 prefill chunk through the model.
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
        let allFinite = asArr.allSatisfy { $0.isFinite }
        #expect(allFinite, "prefill logits must be finite")

        for raCache in raCaches {
            #expect(raCache.inner.offsets[0] == priorT + chunkL,
                "slot 0 offset should be priorT + chunkL")
            #expect(raCache.inner.offsets[1] == priorT + chunkL,
                "slot 1 offset should be priorT + chunkL")
        }
    }
}
