// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Gemma4 batched-sparse decode hook.
//
// Coverage:
//   - Plain dense FFN, no KV sharing (homogeneous sliding layers) — baseline
//     that mirrors the original smoke before KV-shared + MoE support landed.
//   - KV-shared layers (num_kv_shared_layers > 0): trailing layers reuse
//     the donor's `BatchedRetrievalAttentionKVCache`. Mirrors the dense
//     `previousKVs[j] != j` path.
//   - MoE FFN (enable_moe_block + experts/router): exercised through the
//     sparse decode path; only the attention step routes through sparse,
//     expert routing is orthogonal.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

// `.serialized` — each test compiles Gemma4SharedMLP via MLX `compile()`,
// and parallel test execution can deadlock on the compile mutex when
// three @Test functions race the same shared compiled closure. Running
// them in series (~50ms each) avoids the contention.
@Suite("Gemma4 batched-sparse synthetic smoke", .serialized)
struct Gemma4BatchedSparseSmokeTests {

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
        return raCfg
    }

    /// Build per-layer `BatchedRetrievalAttentionKVCache` list, pointing
    /// shared layers at their donors' cache instance — mirrors how
    /// `Gemma4ModelInner.previousKVs` keys shared layers off the donor.
    /// Pre-fills each donor's cache with synthetic K/V at offset T0.
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
                // Donor: allocate its own inner BatchedKVCache + RA wrapper.
                let innerCache = BatchedKVCache(
                    maxBatch: B, kvHeads: kvHeads, headDim: headDim,
                    maxSeq: maxSeq, dtype: .float32)
                for _ in 0..<B { _ = innerCache.addRequest() }
                let ra = BatchedRetrievalAttentionKVCache(
                    inner: innerCache, B: B, nKVHeads: kvHeads, dHead: headDim,
                    layerIdx: i, totalLayers: nLayers, raConfig: raCfg)

                // Pre-fill K/V for the donor cache.
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
        // Second pass: shared layers point at donor's cache instance.
        for i in 0..<nLayers {
            if caches[i] == nil {
                let donor = inner.previousKVs[i]
                precondition(caches[donor] != nil,
                    "donor cache must be built before shared layer")
                caches[i] = caches[donor]
            }
        }
        return caches.compactMap { $0 }
    }

    @Test func sparseDecodeShapesAndFiniteLogits() {
        // Plain dense, no KV-shared, no MoE — original baseline.
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
        let raCaches = Self.buildRaCaches(
            model: model, B: B, kvHeads: 2, headDim: 32, maxSeq: 256,
            prefillT0: 128, raCfg: raCfg)

        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, raCaches: raCaches)
        eval(logits)
        #expect(logits.shape == [B, 1, 256],
            "fullyBatchedSparseDecode output shape")
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite }, "logits must be finite")
    }

    @Test func sparseDecodeWithKvSharedLayers() {
        // 6 layers, all sliding, num_kv_shared_layers=4 ⇒ layers 0,1 are
        // donors; layers 2,3,4,5 reuse them (matching layer_types).
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 128,
            "num_hidden_layers": 6,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "head_dim": 32,
            "global_head_dim": 32,
            "rms_norm_eps": 1e-5,
            "vocab_size": 256,
            "num_key_value_heads": 2,
            "sliding_window": 64,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention"
            ],
            "tie_word_embeddings": true,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0
        }
        """
        let cfg = try! JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
        let model = Gemma4TextModel(cfg)
        eval(model)

        // Verify the model's previousKVs map. Donor-build loop in
        // `Gemma4ModelInner.init` walks layers [0, N-numKvShared) and keeps
        // the LATEST layer-per-type — so with all 6 layers sliding and M=2,
        // the donor for shared layers is layer 1 (the last donor of that
        // type before the shared band).
        let prev = model.model.previousKVs
        #expect(prev.count == 6)
        #expect(prev[0] == 0 && prev[1] == 1)
        for j in 2..<6 {
            #expect(prev[j] == 1, "layer \(j) donor must be layer 1")
        }

        let raCfg = Self.makeRaConfig()
        let B = 2
        let raCaches = Self.buildRaCaches(
            model: model, B: B, kvHeads: 2, headDim: 32, maxSeq: 256,
            prefillT0: 128, raCfg: raCfg)

        // Sanity: caches at shared layers MUST be the donor's instance.
        for j in 2..<6 {
            #expect(raCaches[j] === raCaches[1],
                "shared layer \(j) must point at donor 1's cache")
        }

        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, raCaches: raCaches)
        eval(logits)
        #expect(logits.shape == [B, 1, 256])
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite },
            "logits must be finite under KV-shared sparse decode")
    }

    @Test func sparseDecodeWithMoEBlocks() {
        // Synthetic MoE config: small expert count so SwitchLinear stays cheap.
        // No KV sharing — keep MoE isolated as the variable under test.
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
            "hidden_size_per_layer_input": 0,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2
        }
        """
        let cfg = try! JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
        let model = Gemma4TextModel(cfg)
        eval(model)

        // Confirm the transformer blocks built the MoE experts/router.
        for block in model.model.layers {
            #expect(block.experts != nil, "MoE block must have experts")
            #expect(block.router != nil, "MoE block must have router")
        }

        let raCfg = Self.makeRaConfig()
        let B = 2
        let raCaches = Self.buildRaCaches(
            model: model, B: B, kvHeads: 2, headDim: 32, maxSeq: 256,
            prefillT0: 128, raCfg: raCfg)

        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, raCaches: raCaches)
        eval(logits)
        #expect(logits.shape == [B, 1, 256])
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite },
            "logits must be finite under MoE sparse decode")
    }
}
