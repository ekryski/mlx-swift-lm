// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Gemma4 batched-sparse PREFILL path.
//
// Mirrors the decode smoke's per-layer cache build (donor + shared) but
// runs an L>1 prefill chunk through `fullyBatchedSparseForward`.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Gemma4 batched-sparse PREFILL synthetic smoke", .serialized)
struct Gemma4BatchedSparsePrefillSmokeTests {

    private static func makeRaConfig() -> RetrievalAttentionConfig {
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
        return raCfg
    }

    /// Build per-layer caches honoring `previousKVs` — donors get their own
    /// cache, shared layers reuse donor's instance.
    private static func buildRaCaches(
        model: Gemma4TextModel,
        B: Int, kvHeads: Int, headDim: Int, maxSeq: Int,
        prefillT0: Int, raCfg: RetrievalAttentionConfig
    ) -> [BatchedRetrievalAttentionKVCache] {
        let inner = model.model
        let nLayers = inner.layers.count
        var caches = [BatchedRetrievalAttentionKVCache?](repeating: nil, count: nLayers)
        for i in 0..<nLayers {
            let donor = inner.previousKVs[i]
            if donor == i {
                let innerCache = BatchedKVCache(
                    maxBatch: B, kvHeads: kvHeads, headDim: headDim,
                    maxSeq: maxSeq, dtype: .float32)
                for _ in 0..<B { _ = innerCache.addRequest() }
                let ra = BatchedRetrievalAttentionKVCache(
                    inner: innerCache, B: B, nKVHeads: kvHeads, dHead: headDim,
                    layerIdx: i, totalLayers: nLayers, raConfig: raCfg)
                let kFill = MLXRandom.normal(
                    [B, kvHeads, prefillT0, headDim],
                    key: MLXRandom.key(UInt64(i + 1))
                ).asType(.float32)
                let vFill = MLXRandom.normal(
                    [B, kvHeads, prefillT0, headDim],
                    key: MLXRandom.key(UInt64(i + 101))
                ).asType(.float32)
                innerCache.keys[..<B, 0..., ..<prefillT0, 0...] = kFill
                innerCache.values[..<B, 0..., ..<prefillT0, 0...] = vFill
                for s in 0..<B { innerCache.offsets[s] = prefillT0 }
                ra.updateIndex(newKeys: kFill)
                caches[i] = ra
            }
        }
        for i in 0..<nLayers {
            if caches[i] == nil {
                let donor = inner.previousKVs[i]
                caches[i] = caches[donor]
            }
        }
        return caches.compactMap { $0 }
    }

    @Test func sparsePrefillForwardShapesAndFiniteLogits() {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 32,
            "global_head_dim": 32,
            "rms_norm_eps": 1e-5,
            "vocab_size": 256,
            "num_key_value_heads": 2,
            "sliding_window": 64,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "tie_word_embeddings": true,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0
        }
        """
        let cfg = try! JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
        let model = Gemma4TextModel(cfg)
        eval(model)

        let raCfg = Self.makeRaConfig()
        let B = 2
        let priorT = 128
        let raCaches = Self.buildRaCaches(
            model: model, B: B, kvHeads: 2, headDim: 32, maxSeq: 512,
            prefillT0: priorT, raCfg: raCfg)

        let chunkL = 16
        let tokens = MLXRandom.randInt(
            low: 0, high: 256, [B, chunkL]).asType(.int32)
        let out = model.model.fullyBatchedSparseForward(tokens, raCaches: raCaches)
        let logits: MLXArray
        if cfg.tieWordEmbeddings {
            logits = model.model.embedTokens.asLinear(out)
        } else {
            logits = model.lmHead!(out)
        }
        eval(logits)
        #expect(logits.shape == [B, chunkL, 256],
            "prefill forward output shape")
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite }, "prefill logits must be finite")
    }
}
