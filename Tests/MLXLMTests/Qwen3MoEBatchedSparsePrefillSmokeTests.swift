// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Qwen3MoE batched-sparse PREFILL path.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen3MoE batched-sparse PREFILL synthetic smoke", .serialized)
struct Qwen3MoEBatchedSparsePrefillSmokeTests {

    /// Tiny config — all layers are mlp_only (matches decode smoke).
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

    @Test func sparsePrefillForwardShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Qwen3MoEModel(cfg)
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
