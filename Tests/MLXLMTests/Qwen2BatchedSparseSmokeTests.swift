// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Qwen2 batched-sparse decode hook. Builds a
// tiny random-weight Qwen2 model in memory (no checkpoint), reserves B
// slots in per-layer BatchedRetrievalAttentionKVCache, runs prefill + a
// few decode steps, and asserts output shape + finite logits.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen2 batched-sparse synthetic smoke")
struct Qwen2BatchedSparseSmokeTests {

    private func makeTinyConfig() -> Qwen2Configuration {
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
          "rope_traditional": false,
          "tie_word_embeddings": true
        }
        """.data(using: .utf8)!
        return try! JSONDecoder().decode(Qwen2Configuration.self, from: json)
    }

    @Test func sparseDecodeShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Qwen2Model(cfg)
        // Random weights — we only need shape + dtype propagation correct.
        eval(model)

        let B = 2
        let nLayers = cfg.hiddenLayers
        let kvHeads = cfg.kvHeads
        let headDim = cfg.hiddenSize / cfg.attentionHeads
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

        // Synthetic prefill: write 128 tokens of fake K/V into each layer
        // cache, advance per-slot offsets accordingly. Bypasses model
        // forward to keep the smoke focused on the sparse decode path.
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
            // Selector index: feed the prefill so block features exist for
            // sparse-eligible layers.
            raCache.updateIndex(newKeys: kFill)
        }

        // Decode a few tokens. Random next-token ids.
        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, raCaches: raCaches)
        eval(logits)
        #expect(logits.shape == [B, 1, cfg.vocabularySize],
            "fullyBatchedSparseDecode output shape")
        // Finite logits (no NaN/Inf leaking from the sparse kernel).
        let asArr = logits.asArray(Float.self)
        let allFinite = asArr.allSatisfy { $0.isFinite }
        #expect(allFinite, "logits must be finite")
    }
}
