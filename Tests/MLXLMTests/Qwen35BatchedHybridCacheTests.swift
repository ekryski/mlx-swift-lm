// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Cache-layout / lifecycle coverage for the BatchedHybridCache produced by
// Qwen35TextModel.newBatchedHybridCache. End-to-end forward correctness
// against the per-request iterator path is deferred to Phase 6.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Qwen35Model BatchedHybridLLM cache lifecycle")
struct Qwen35BatchedHybridCacheTests {

    // Constants mirror the JSON below — kept here so we don't depend on
    // internal-access fields of Qwen35TextConfiguration.
    fileprivate enum DenseShape {
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

    /// Minimal dense Qwen3.5-style configuration. Layer count is a multiple of
    /// `fullAttentionInterval` so we exercise both GDN and attention slots.
    private static func makeDenseConfig() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_hidden_layers": \(DenseShape.hiddenLayers),
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": \(DenseShape.kvHeads),
                "linear_num_value_heads": \(DenseShape.linearNumValueHeads),
                "linear_num_key_heads": \(DenseShape.linearNumKeyHeads),
                "linear_key_head_dim": \(DenseShape.linearKeyHeadDim),
                "linear_value_head_dim": \(DenseShape.linearValueHeadDim),
                "linear_conv_kernel_dim": \(DenseShape.linearConvKernelDim),
                "rms_norm_eps": 1e-6,
                "vocab_size": 256,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "max_position_embeddings": 512,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "head_dim": \(DenseShape.headDim),
                "full_attention_interval": \(DenseShape.fullAttentionInterval)
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: json)
    }

    @Test
    func `newBatchedHybridCache produces correct per-layer types`() throws {
        let cfg = try Self.makeDenseConfig()
        let model = Qwen35TextModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 4, parameters: nil)

        #expect(cache.layers.count == DenseShape.hiddenLayers)
        #expect(cache.active == 0)

        // Layer pattern: isLinear = (layerIdx + 1) % fullAttentionInterval != 0
        // With fullAttentionInterval=4 and 4 layers: indices 0,1,2 are GDN,
        // index 3 is attention.
        for (i, layer) in cache.layers.enumerated() {
            let expectAttention = ((i + 1) % DenseShape.fullAttentionInterval == 0)
            switch layer {
            case .attention:
                #expect(expectAttention,
                        "layer \(i) is attention but pattern expects GDN")
            case .gdn:
                #expect(!expectAttention,
                        "layer \(i) is GDN but pattern expects attention")
            }
        }
    }

    @Test
    func `addSlot keeps every Qwen35 layer in lockstep`() throws {
        let cfg = try Self.makeDenseConfig()
        let model = Qwen35TextModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 3, parameters: nil)

        let s0 = cache.addSlot()
        #expect(s0 == 0)
        #expect(cache.active == 1)

        let s1 = cache.addSlot()
        #expect(s1 == 1)
        #expect(cache.active == 2)

        // Every layer must report the same active count.
        for layer in cache.layers {
            switch layer {
            case .attention(let c): #expect(c.active == 2)
            case .gdn(let c): #expect(c.active == 2)
            }
        }
    }

    @Test
    func `removeSlot keeps every Qwen35 layer in lockstep`() throws {
        let cfg = try Self.makeDenseConfig()
        let model = Qwen35TextModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 3, parameters: nil)

        cache.addSlot()
        cache.addSlot()
        cache.addSlot()
        #expect(cache.active == 3)

        cache.removeSlot(1)
        #expect(cache.active == 2)
        for layer in cache.layers {
            switch layer {
            case .attention(let c): #expect(c.active == 2)
            case .gdn(let c): #expect(c.active == 2)
            }
        }
    }

    @Test
    func `GDN slot reuse zero-inits stale state on Qwen35 cache`() throws {
        // Mirrors the BatchedMambaCache slot-reuse invariant test, but driven
        // through the Qwen35-shaped cache.
        let cfg = try Self.makeDenseConfig()
        let model = Qwen35TextModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 3, parameters: nil)

        cache.addSlot()
        cache.addSlot()
        cache.addSlot()

        // Find the first GDN layer and stuff its slot 1 with non-zero state.
        var gdnLayer: BatchedMambaCache?
        for layer in cache.layers {
            if case .gdn(let c) = layer { gdnLayer = c; break }
        }
        let gdn = try #require(gdnLayer)
        let convFill = MLXArray.ones(
            [gdn.kernelMinusOne, gdn.convDim], dtype: gdn.convState.dtype) * 7.5
        gdn.convState[1, 0..., 0...] = convFill

        // removeSlot(0) → slot 2 swaps in. active = 2.
        cache.removeSlot(0)

        // Now scribble garbage into the freed tail position 2 across all GDN
        // layers, then re-allocate it. addSlot must wipe.
        for layer in cache.layers {
            if case .gdn(let c) = layer {
                c.convState[2, 0..., 0...] =
                    MLXArray.ones([c.kernelMinusOne, c.convDim],
                                  dtype: c.convState.dtype) * 11.0
                c.recState[2, 0..., 0..., 0...] =
                    MLXArray.ones([c.Hv, c.Dv, c.Dk], dtype: .float32) * 13.0
            }
        }

        let reused = cache.addSlot()
        #expect(reused == 2)

        for layer in cache.layers {
            if case .gdn(let c) = layer {
                let convReused = c.convState[2, 0..., 0...]
                let recReused = c.recState[2, 0..., 0..., 0...]
                #expect(convReused.sum().item(Float.self) == 0,
                        "stale GDN conv leaked into reused slot")
                #expect(recReused.sum().item(Float.self) == 0,
                        "stale GDN rec leaked into reused slot")
            }
        }
    }

    /// Regression test for issue #8 phase 6 bug 2: identical inputs and identical
    /// per-slot state in `fullyBatchedDecode` must produce identical outputs across
    /// every slot. The original bug: `MLX.split(convOut, axis: -1)` returns
    /// non-contiguous slices, and the GDN Metal kernel uses raw pointer arithmetic
    /// with assumed contiguous strides. At B=1 the wrong stride coincidentally
    /// landed on correct data (single-slot, no inter-slot confusion), but at B>1
    /// the kernel read into the wrong slot's region of memory. Fix: `.contiguous()`
    /// on q/k/v after the split (Qwen35.swift, Qwen3Next.swift). This test
    /// explicitly exercises the per-slot equivalence invariant against the live
    /// model forward — without the fix, slots 1+ would diverge from slot 0.
    @Test
    func `fullyBatchedDecode produces identical outputs for identical inputs across slots`() async throws {
        // Use a config whose GDN dims match a metallib kernel instantiation
        // (Dk=128, Dv=128, Hk=16, Hv=16 → bfloat16). The default makeDenseConfig()
        // uses tiny dims that aren't pre-instantiated and crashes test runs.
        let json = """
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
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: json)

        let model = Qwen35TextModel(cfg)
        // Cast all parameters to bf16 so the Metal kernel finds its instantiation.
        eval(model)
        let bf16Params = model.parameters().mapValues { (v: MLXArray) in v.asType(.bfloat16) }
        try model.update(parameters: bf16Params, verify: [.noUnusedKeys])
        eval(model)

        let B = 4
        let cache = model.newBatchedHybridCache(maxBatch: B, parameters: nil)
        for _ in 0..<B { cache.addSlot() }

        // Same input token in every slot → identical inputs/state → identical
        // outputs across slots. Any divergence is the kernel-stride bug.
        let inputs = MLXArray.full([B, 1], values: MLXArray(Int32(7)), dtype: .int32)
        let logits = model.fullyBatchedDecode(inputs, caches: cache)
        eval(logits)

        let lastLogits = logits[0..., -1, 0...]  // [B, vocab]
        let slot0 = lastLogits[0]
        for s in 1..<B {
            let diff = (lastLogits[s] - slot0).abs().max().item(Float.self)
            #expect(diff == 0,
                    "slot \(s) logits diverged from slot 0 by max abs \(diff) — Metal kernel non-contiguous stride bug")
        }
    }

    @Test
    func `cache shape matches Qwen35 configuration`() throws {
        let cfg = try Self.makeDenseConfig()
        let model = Qwen35TextModel(cfg)
        let cache = model.newBatchedHybridCache(maxBatch: 2, parameters: nil)

        let expectedKernelMinusOne = DenseShape.linearConvKernelDim - 1
        let expectedKeyDim = DenseShape.linearKeyHeadDim * DenseShape.linearNumKeyHeads
        let expectedValueDim = DenseShape.linearValueHeadDim * DenseShape.linearNumValueHeads
        let expectedConvDim = expectedKeyDim * 2 + expectedValueDim

        for layer in cache.layers {
            switch layer {
            case .gdn(let c):
                #expect(c.maxBatch == 2)
                #expect(c.kernelMinusOne == expectedKernelMinusOne)
                #expect(c.convDim == expectedConvDim)
                #expect(c.Hv == DenseShape.linearNumValueHeads)
                #expect(c.Dv == DenseShape.linearValueHeadDim)
                #expect(c.Dk == DenseShape.linearKeyHeadDim)
            case .attention(let c):
                #expect(c.maxBatch == 2)
                #expect(c.kvHeads == DenseShape.kvHeads)
                #expect(c.headDim == DenseShape.headDim)
                #expect(c.maxSeq == 2048)  // default when parameters == nil
            }
        }
    }
}
