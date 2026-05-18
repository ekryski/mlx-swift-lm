// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the Qwen35 hybrid batched-sparse decode hook.
//
// `newBatchedHybridSparseCache` produces a BatchedHybridCache whose
// attention slots are `.sparseAttention(BatchedRetrievalAttentionKVCache)`
// and GDN slots stay `.gdn(BatchedMambaCache)`. The smoke test rotates
// through the foundation surface (cache build, slot lifecycle) and then
// runs a tiny synthetic forward to validate shape + finiteness.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen35 hybrid batched-sparse synthetic smoke")
struct Qwen35BatchedSparseSmokeTests {

    /// Use the kernel-instantiated GDN dims (Dk=Dv=128, Hv=Hk=16, bfloat16)
    /// so the GDN Metal kernel finds its instantiation. The dense config in
    /// `Qwen35BatchedHybridCacheTests` uses tinier dims that would crash a
    /// forward call.
    private static let json = """
        {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "linear_num_value_heads": 16,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 512,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "head_dim": 8,
            "full_attention_interval": 4
        }
        """

    @Test
    func newBatchedHybridSparseCacheLayout() throws {
        let cfg = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Self.json.data(using: .utf8)!)
        let model = Qwen35TextModel(cfg)

        var raCfg = RetrievalAttentionConfig()
        raCfg.fineBlockSize = 32
        raCfg.coarseRescueEnabled = false
        raCfg.adaptiveTopK = false
        raCfg.fineTopK = 2
        raCfg.staticInit = 32
        raCfg.slidingWindow = 64
        raCfg.denseFirstN = 0
        raCfg.denseLastN = 0
        raCfg.sparseMinContext = 0

        let cache = model.newBatchedHybridSparseCache(
            maxBatch: 2, parameters: nil, raConfig: raCfg)

        #expect(cache.layers.count == 4, "layer count must match config")

        var sparseAttentionCount = 0
        var gdnCount = 0
        for layer in cache.layers {
            switch layer {
            case .sparseAttention: sparseAttentionCount += 1
            case .gdn: gdnCount += 1
            case .attention:
                Issue.record("hybrid sparse cache should not emit .attention")
            }
        }
        // 4 layers, fullAttentionInterval=4 ⇒ 3 GDN + 1 sparseAttention.
        #expect(sparseAttentionCount == 1)
        #expect(gdnCount == 3)
    }

    @Test
    func fullyBatchedSparseDecodeShapesAndFiniteLogits() throws {
        let cfg = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Self.json.data(using: .utf8)!)
        let model = Qwen35TextModel(cfg)
        eval(model)
        // Cast params to bf16 so the GDN Metal kernel finds its instantiation.
        let bf16Params = model.parameters().mapValues { (v: MLXArray) in v.asType(.bfloat16) }
        try model.update(parameters: bf16Params, verify: [.noUnusedKeys])
        eval(model)

        var raCfg = RetrievalAttentionConfig()
        raCfg.fineBlockSize = 32
        raCfg.coarseRescueEnabled = false
        raCfg.adaptiveTopK = false
        raCfg.fineTopK = 2
        raCfg.staticInit = 32
        raCfg.slidingWindow = 64
        raCfg.denseFirstN = 0
        raCfg.denseLastN = 0
        raCfg.sparseMinContext = 0

        let B = 2
        let cache = model.newBatchedHybridSparseCache(
            maxBatch: B, parameters: nil, raConfig: raCfg)
        for _ in 0..<B { cache.addSlot() }

        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, caches: cache)
        eval(logits)
        #expect(logits.shape == [B, 1, cfg.vocabularySize])
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite }, "logits must be finite")
    }
}
