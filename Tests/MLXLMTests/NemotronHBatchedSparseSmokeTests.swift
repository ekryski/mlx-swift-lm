// SPDX-License-Identifier: Apache-2.0
// Synthetic smoke for the NemotronH hybrid batched-sparse decode hook.
//
// Mirrors `Qwen35BatchedSparseSmokeTests` in spirit:
//   - `newBatchedHybridSparseCache` produces a `BatchedHybridCache` whose
//     attention slots wrap `BatchedRetrievalAttentionKVCache` and Mamba2
//     slots wrap `BatchedMambaCache` (with `recDtype` = model dtype).
//   - MLP/MoE blocks contribute NO cache slot, mirroring `newCache`.
//   - Slot lifecycle (addSlot) propagates lockstep across mamba/attention.
//   - `fullyBatchedSparseDecode` runs end-to-end with synthetic weights.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

// `.serialized` — Mamba2 SSM kernel + `eval()` over freshly-built caches
// don't tolerate parallel test execution well (mutex contention inside
// MLX). Running serially keeps the smoke deterministic.
@Suite("NemotronH hybrid batched-sparse synthetic smoke", .serialized)
struct NemotronHBatchedSparseSmokeTests {

    /// Tiny synthetic config. Pattern `MEMEM*` exercises every block kind:
    /// mamba ('M') × 3, attention ('*') × 1, MoE ('E') × 2 — and ensures
    /// the cache index advances only on mamba/attention.
    private static func makeTinyConfig() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 128,
            hiddenSize: 64,
            numHiddenLayers: 6,                  // length of `MEMEM*`
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "MEMEM*",
            layerNormEpsilon: 1e-5,
            nGroup: 2,
            topkGroup: 1
        )
    }

    private static func makeRaConfig() -> RetrievalAttentionConfig {
        var ra = RetrievalAttentionConfig()
        ra.fineBlockSize = 32
        ra.coarseRescueEnabled = false
        ra.adaptiveTopK = false
        ra.fineTopK = 2
        ra.staticInit = 32
        ra.slidingWindow = 64
        ra.denseFirstN = 0
        ra.denseLastN = 0
        ra.sparseMinContext = 0
        return ra
    }

    @Test
    func newBatchedHybridSparseCacheLayout() throws {
        let cfg = Self.makeTinyConfig()
        let model = NemotronHModel(cfg)

        let cache = model.newBatchedHybridSparseCache(
            maxBatch: 2, parameters: nil, raConfig: Self.makeRaConfig())

        // `MEMEM*` → 3 mamba + 1 attention + 2 moe → 4 cache slots
        // (mamba × 3 + attention × 1; moe contributes nothing).
        #expect(cache.layers.count == 4,
            "hybrid sparse cache should have one slot per mamba/attention block")

        var gdnCount = 0
        var sparseAttnCount = 0
        for slot in cache.layers {
            switch slot {
            case .gdn: gdnCount += 1
            case .sparseAttention: sparseAttnCount += 1
            case .attention:
                Issue.record("sparse hybrid cache must not emit .attention slots")
            }
        }
        #expect(gdnCount == 3, "expected 3 mamba slots")
        #expect(sparseAttnCount == 1, "expected 1 sparse-attention slot")
    }

    @Test
    func mambaCacheRecDtypeMatchesModelDtype() throws {
        let cfg = Self.makeTinyConfig()
        let model = NemotronHModel(cfg)

        let cache = model.newBatchedHybridSparseCache(
            maxBatch: 2, parameters: nil, raConfig: Self.makeRaConfig())

        // NemotronH's input-dtype SSM state contract — the gdn cache's
        // `recDtype` should match the embedding (model compute) dtype,
        // NOT fp32 like the GDN family's default.
        let modelDtype = model.backbone.embeddings.weight.dtype
        for slot in cache.layers {
            if case .gdn(let mamba) = slot {
                #expect(mamba.recDtype == modelDtype,
                    "BatchedMambaCache.recDtype must follow NemotronH's input dtype")
            }
        }
    }

    @Test
    func fullyBatchedSparseDecodeShapesAndFiniteLogits() throws {
        let cfg = Self.makeTinyConfig()
        let model = NemotronHModel(cfg)
        eval(model)
        // Cast to bf16 — the SSM Metal kernel registers bf16/fp16/fp32
        // template instantiations; we pick bf16 to match `BatchedMambaCache`
        // default conv dtype.
        let bf16Params = model.parameters().mapValues { (v: MLXArray) in
            v.asType(.bfloat16)
        }
        try model.update(parameters: bf16Params, verify: [.noUnusedKeys])
        eval(model)

        let B = 2
        let cache = model.newBatchedHybridSparseCache(
            maxBatch: B, parameters: nil, raConfig: Self.makeRaConfig())
        for _ in 0..<B { cache.addSlot() }

        let tokens = MLXArray([0, 1] as [Int32]).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, caches: cache)
        eval(logits)
        #expect(logits.shape == [B, 1, cfg.vocabSize])
        let asArr = logits.asArray(Float.self)
        #expect(asArr.allSatisfy { $0.isFinite }, "logits must be finite")
    }
}
