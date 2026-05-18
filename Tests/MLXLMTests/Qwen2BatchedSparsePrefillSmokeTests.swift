// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Qwen2 batched-sparse PREFILL path. Builds a tiny
// random-weight Qwen2 model in memory, reserves B slots in per-layer
// BatchedRetrievalAttentionKVCache, primes a synthetic prior cache, then
// runs an L>1 prefill chunk through `fullyBatchedSparseForward` with
// `sparsePrefillEnabled=true`. Asserts output shape + finite logits.
//
// The smoke does NOT check numerical equivalence vs dense at the model
// scale — that's covered by kernel-level tests in BatchedSparsePrefillTests.
// This test just verifies the model-level wire-up (Attention →
// DecoderLayer → ModelInner → Qwen2Model) compiles and runs end-to-end.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen2 batched-sparse PREFILL synthetic smoke")
struct Qwen2BatchedSparsePrefillSmokeTests {

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

    @Test func sparsePrefillForwardShapesAndFiniteLogits() {
        let cfg = makeTinyConfig()
        let model = Qwen2Model(cfg)
        eval(model)

        let B = 2
        let nLayers = cfg.hiddenLayers
        let kvHeads = cfg.kvHeads
        let headDim = cfg.hiddenSize / cfg.attentionHeads
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
        raCfg.sparsePrefillMinContext = 0  // engage on this synthetic test

        let raCaches: [BatchedRetrievalAttentionKVCache] = (0..<nLayers).map {
            layer in
            let inner = BatchedKVCache(
                maxBatch: B, kvHeads: kvHeads, headDim: headDim,
                maxSeq: maxSeq, dtype: .float32)
            for _ in 0..<B { _ = inner.addRequest() }
            return BatchedRetrievalAttentionKVCache(
                inner: inner, B: B, nKVHeads: kvHeads, dHead: headDim,
                layerIdx: layer, totalLayers: nLayers, raConfig: raCfg)
        }

        // Synthetic prior: write 128 tokens of fake K/V into each layer
        // cache, prime the selector. Bypasses prior-prefill model forward.
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

        // Now feed an L=16 prefill chunk through the model. This exercises
        // the L>1 branch in `fullyBatchedSparseForward` which routes to
        // `cache.updateChunk` + `raCache.prefillSparseAttend` (for the
        // sparse-band layers; first + last layers stay dense).
        let chunkL = 16
        let tokens = MLXRandom.randInt(
            low: 0, high: cfg.vocabularySize,
            [B, chunkL]).asType(.int32)
        // Call the chunk forward directly via the per-family hook. The
        // top-level `fullyBatchedSparseDecode` wraps decode-step semantics;
        // for prefill we run `model.fullyBatchedSparseForward` then apply
        // lmHead/tied embedding manually.
        var out = model.model.fullyBatchedSparseForward(tokens, raCaches: raCaches)
        if let lmHead = model.lmHead {
            out = lmHead(out)
        } else {
            out = model.model.embedTokens.asLinear(out)
        }
        eval(out)
        #expect(out.shape == [B, chunkL, cfg.vocabularySize],
            "prefill forward output shape")
        let asArr = out.asArray(Float.self)
        let allFinite = asArr.allSatisfy { $0.isFinite }
        #expect(allFinite, "prefill logits must be finite")

        // Offsets should have advanced from priorT to priorT + chunkL.
        for raCache in raCaches {
            #expect(raCache.inner.offsets[0] == priorT + chunkL,
                "slot 0 offset should be priorT + chunkL")
            #expect(raCache.inner.offsets[1] == priorT + chunkL,
                "slot 1 offset should be priorT + chunkL")
        }
    }
}
