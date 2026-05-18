// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Qwen3MoE batched-sparse decode hook. Tiny
// random-weight Qwen3MoE model in memory; B slots in per-layer
// BatchedRetrievalAttentionKVCache; synthetic prefill + one decode step.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen3MoE batched-sparse synthetic smoke")
struct Qwen3MoEBatchedSparseSmokeTests {

    /// Tiny config — uses a single dense MLP layer (mlp_only_layers covers all
    /// indices) to keep the synthetic smoke focused on the attention path,
    /// not the SwitchGLU expert routing.
    private func makeTinyConfig() -> Qwen3MoEConfiguration {
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
          "tie_word_embeddings": true,
          "num_experts": 0,
          "num_experts_per_tok": 1,
          "decoder_sparse_step": 1,
          "moe_intermediate_size": 64,
          "mlp_only_layers": [0, 1, 2, 3]
        }
        """.data(using: .utf8)!
        return try! JSONDecoder().decode(Qwen3MoEConfiguration.self, from: json)
    }

    @Test func sparseDecodeShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Qwen3MoEModel(cfg)
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
