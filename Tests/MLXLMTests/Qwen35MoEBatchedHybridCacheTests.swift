// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Cache-layout / lifecycle coverage for Qwen3.6 (Qwen35MoEModel). The MoE
// variant inherits BatchedHybridLLM conformance from Qwen35Model — the only
// model-side diff is that dense MLP becomes Qwen35SparseMoeBlock. The cache
// layout is identical to the dense Qwen3.5 case, so we just verify lifecycle
// against an MoE-flavored configuration here. End-to-end forward correctness
// is deferred to Phase 6.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Qwen35MoEModel BatchedHybridLLM cache lifecycle")
struct Qwen35MoEBatchedHybridCacheTests {

    // Constants mirror the JSON below — kept here so we don't depend on
    // internal-access fields of Qwen35TextConfiguration / Qwen35Configuration.
    fileprivate enum MoEShape {
        static let hiddenLayers = 4
        static let fullAttentionInterval = 4   // → layers 0,1,2 GDN; layer 3 attention
        static let linearConvKernelDim = 4
        static let linearKeyHeadDim = 16
        static let linearNumKeyHeads = 2
        static let linearValueHeadDim = 16
        static let linearNumValueHeads = 4
        static let kvHeads = 2
        static let headDim = 8
    }

    /// Minimal MoE Qwen3.6-style configuration. Set num_experts > 0 so the
    /// decoder layer wires Qwen35SparseMoeBlock. Layer count is a multiple of
    /// fullAttentionInterval so we exercise both GDN and attention slots.
    private static func makeMoEConfig() throws -> Qwen35Configuration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": \(MoEShape.hiddenLayers),
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": \(MoEShape.kvHeads),
                "linear_num_value_heads": \(MoEShape.linearNumValueHeads),
                "linear_num_key_heads": \(MoEShape.linearNumKeyHeads),
                "linear_key_head_dim": \(MoEShape.linearKeyHeadDim),
                "linear_value_head_dim": \(MoEShape.linearValueHeadDim),
                "linear_conv_kernel_dim": \(MoEShape.linearConvKernelDim),
                "rms_norm_eps": 1e-6,
                "vocab_size": 256,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "max_position_embeddings": 512,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "head_dim": \(MoEShape.headDim),
                "full_attention_interval": \(MoEShape.fullAttentionInterval),
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 64,
                "moe_intermediate_size": 64,
                "norm_topk_prob": true
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen35Configuration.self, from: json)
    }

    @Test
    func `Qwen35MoEModel inherits BatchedHybridLLM and exposes cache layout`() throws {
        let cfg = try Self.makeMoEConfig()
        let model = Qwen35MoEModel(cfg)
        // Inherited surface is reachable from the subclass.
        let cache = model.newBatchedHybridCache(maxBatch: 4, parameters: nil)

        #expect(cache.layers.count == MoEShape.hiddenLayers)
        #expect(cache.active == 0)

        for (i, layer) in cache.layers.enumerated() {
            let expectAttention = ((i + 1) % MoEShape.fullAttentionInterval == 0)
            switch layer {
            case .attention: #expect(expectAttention)
            case .gdn: #expect(!expectAttention)
            }
        }
    }

    @Test
    func `addSlot/removeSlot lockstep on Qwen35MoE cache`() throws {
        let cfg = try Self.makeMoEConfig()
        let model = Qwen35MoEModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 3, parameters: nil)

        cache.addSlot()
        cache.addSlot()
        cache.addSlot()
        #expect(cache.active == 3)

        cache.removeSlot(0)
        #expect(cache.active == 2)
        for layer in cache.layers {
            switch layer {
            case .attention(let c): #expect(c.active == 2)
            case .gdn(let c): #expect(c.active == 2)
            }
        }

        cache.reset()
        #expect(cache.active == 0)
    }

    @Test
    func `MoE cache shape matches Qwen3.6 configuration`() throws {
        let cfg = try Self.makeMoEConfig()
        let model = Qwen35MoEModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 2, parameters: nil)

        let expectedKernelMinusOne = MoEShape.linearConvKernelDim - 1
        let expectedKeyDim = MoEShape.linearKeyHeadDim * MoEShape.linearNumKeyHeads
        let expectedValueDim = MoEShape.linearValueHeadDim * MoEShape.linearNumValueHeads
        let expectedConvDim = expectedKeyDim * 2 + expectedValueDim

        for layer in cache.layers {
            switch layer {
            case .gdn(let c):
                #expect(c.kernelMinusOne == expectedKernelMinusOne)
                #expect(c.convDim == expectedConvDim)
                #expect(c.Hv == MoEShape.linearNumValueHeads)
                #expect(c.Dv == MoEShape.linearValueHeadDim)
                #expect(c.Dk == MoEShape.linearKeyHeadDim)
            case .attention(let c):
                #expect(c.kvHeads == MoEShape.kvHeads)
                #expect(c.headDim == MoEShape.headDim)
            }
        }
    }
}
