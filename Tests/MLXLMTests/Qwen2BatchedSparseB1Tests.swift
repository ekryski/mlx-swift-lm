// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Qwen2 batched-sparse @ B=1. The batched cache is positioned as
// "B≥1" in the rest of the suite; this file proves it ALSO covers
// single-stream (B=1) decode, so the legacy non-batched single-stream
// RA wrapper (`RetrievalAttentionKVCache`) is redundant and not
// proposed for upstreaming. See `specs/044-batched-sparse-attention-decode.md`.
//
// Two tests:
//   1. `sparseDecodeB1ShapesFiniteAndOffsetAdvance` — synthetic prefill +
//      one decode step at B=1. Asserts output shape `[1, 1, vocab]`,
//      finite logits (no NaN/Inf), AND that the inner cache's
//      `offsets[0]` advanced from `T0` to `T0 + 1` after the step.
//   2. `sparseDecodeB1SlotSemanticEqualsB2Slot0` — runs the same input
//      through a B=1 cache stack AND a B=2 cache stack (with slot 1
//      held identical), then checks slot 0's logits match within
//      tolerance. Validates that the B=1 path is semantically equivalent
//      to the first slot of the B≥2 path — one cache, one selector, one
//      mask kernel covers all batch sizes.
//
// `.serialized` mirrors the convention used by Gemma4 / NemotronH
// smoke suites (MLX `compile()` + random-key shared state safety).

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Qwen2 batched-sparse B=1 smoke", .serialized)
struct Qwen2BatchedSparseB1Tests {

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

    private func makeRaConfig() -> RetrievalAttentionConfig {
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

    /// Build per-layer `BatchedRetrievalAttentionKVCache` list for a given
    /// `B`, register B slots on each inner cache, and synthetically prefill
    /// `T0` tokens of K/V using deterministic per-layer RNG keys so that
    /// the B=1 and B=2 paths see the SAME slot-0 contents in the
    /// `sparseDecodeB1SlotSemanticEqualsB2Slot0` test.
    private func buildCaches(
        cfg: Qwen2Configuration, raCfg: RetrievalAttentionConfig,
        B: Int, T0: Int, maxSeq: Int
    ) -> [BatchedRetrievalAttentionKVCache] {
        let nLayers = cfg.hiddenLayers
        let kvHeads = cfg.kvHeads
        let headDim = cfg.hiddenSize / cfg.attentionHeads

        return (0..<nLayers).map { layer in
            let inner = BatchedKVCache(
                maxBatch: B, kvHeads: kvHeads, headDim: headDim,
                maxSeq: maxSeq, dtype: .float32)
            for _ in 0..<B { _ = inner.addRequest() }
            let raCache = BatchedRetrievalAttentionKVCache(
                inner: inner, B: B, nKVHeads: kvHeads, dHead: headDim,
                layerIdx: layer, totalLayers: nLayers, raConfig: raCfg)

            // Slot 0 prefill — deterministic per-layer key, identical
            // across B=1 and B=2 callers so cross-cache comparison works.
            let kSlot0 = MLXRandom.normal(
                [1, kvHeads, T0, headDim],
                key: MLXRandom.key(UInt64(layer + 1))
            ).asType(.float32)
            let vSlot0 = MLXRandom.normal(
                [1, kvHeads, T0, headDim],
                key: MLXRandom.key(UInt64(layer + 101))
            ).asType(.float32)
            inner.keys[0..<1, 0..., ..<T0, 0...] = kSlot0
            inner.values[0..<1, 0..., ..<T0, 0...] = vSlot0
            inner.offsets[0] = T0

            // Remaining slots (only when B > 1) — distinct content so the
            // sparse path's per-slot bookkeeping isn't accidentally
            // sharing state across slot indices.
            if B > 1 {
                for s in 1..<B {
                    let kS = MLXRandom.normal(
                        [1, kvHeads, T0, headDim],
                        key: MLXRandom.key(UInt64(layer * 1000 + s * 7 + 13))
                    ).asType(.float32)
                    let vS = MLXRandom.normal(
                        [1, kvHeads, T0, headDim],
                        key: MLXRandom.key(UInt64(layer * 1000 + s * 7 + 113))
                    ).asType(.float32)
                    inner.keys[s..<(s + 1), 0..., ..<T0, 0...] = kS
                    inner.values[s..<(s + 1), 0..., ..<T0, 0...] = vS
                    inner.offsets[s] = T0
                }
            }

            // Feed selector: stack across all slots so block features
            // exist for every (slot, kv-head) pair.
            // `BatchedRetrievalAttentionKVCache.updateIndex` expects the
            // full `[B, kvHeads, T0, headDim]` shape — read it back from
            // `inner.keys` which we've just populated.
            let fullPrefill = inner.keys[..<B, 0..., ..<T0, 0...]
            raCache.updateIndex(newKeys: fullPrefill)

            return raCache
        }
    }

    @Test func sparseDecodeB1ShapesFiniteAndOffsetAdvance() {
        let cfg = makeTinyConfig()
        let raCfg = makeRaConfig()
        let model = Qwen2Model(cfg)
        eval(model)

        let B = 1
        let T0 = 128
        let maxSeq = 256
        let raCaches = buildCaches(
            cfg: cfg, raCfg: raCfg, B: B, T0: T0, maxSeq: maxSeq)

        // Pre-step offset snapshot — defensive copy so we can compare
        // after the in-place advance in `sparseAttend` → inner.update.
        let preOffsets = raCaches.map { $0.inner.offsets[0] }
        #expect(preOffsets.allSatisfy { $0 == T0 },
            "all layers should start at offset \(T0)")

        // One decode step at B=1.
        let tokens = MLXArray([Int32(0)]).reshaped(1, 1)
        let logits = model.fullyBatchedSparseDecode(tokens, raCaches: raCaches)
        eval(logits)

        // Shape: [B=1, T=1, vocab].
        #expect(logits.shape == [1, 1, cfg.vocabularySize],
            "B=1 fullyBatchedSparseDecode output shape")

        // Finite logits: no NaN/Inf leaking from the sparse path at B=1.
        let asArr = logits.asArray(Float.self)
        let allFinite = asArr.allSatisfy { $0.isFinite }
        #expect(allFinite, "B=1 logits must be finite")

        // Offset advance: every layer's slot 0 must have moved from T0
        // to T0 + 1 after one decode step.
        for (li, raCache) in raCaches.enumerated() {
            #expect(raCache.inner.offsets[0] == T0 + 1,
                "layer \(li) slot 0 offset advanced T0=\(T0) → T0+1")
        }
    }

    @Test func sparseDecodeB1SlotSemanticEqualsB2Slot0() {
        let cfg = makeTinyConfig()
        let raCfg = makeRaConfig()
        let model = Qwen2Model(cfg)
        eval(model)

        let T0 = 128
        let maxSeq = 256

        // Stack 1: B=1.
        let raCachesB1 = buildCaches(
            cfg: cfg, raCfg: raCfg, B: 1, T0: T0, maxSeq: maxSeq)
        // Stack 2: B=2. Slot 0 deterministically equal to B1 slot 0
        // (same per-layer RNG key); slot 1 unrelated.
        let raCachesB2 = buildCaches(
            cfg: cfg, raCfg: raCfg, B: 2, T0: T0, maxSeq: maxSeq)

        // Decode tokens — slot 0 gets the same input in both runs.
        // Slot 1 (B=2 only) gets a different token — doesn't matter for
        // the slot-0 comparison.
        let tokB1 = MLXArray([Int32(0)]).reshaped(1, 1)
        let tokB2 = MLXArray([Int32(0), Int32(1)]).reshaped(2, 1)

        let logitsB1 = model.fullyBatchedSparseDecode(tokB1, raCaches: raCachesB1)
        let logitsB2 = model.fullyBatchedSparseDecode(tokB2, raCaches: raCachesB2)
        eval(logitsB1, logitsB2)

        #expect(logitsB1.shape == [1, 1, cfg.vocabularySize])
        #expect(logitsB2.shape == [2, 1, cfg.vocabularySize])

        // Slot 0 comparison. The sparse mask kernel + selector are
        // per-slot independent, so slot 0 in the B=2 pass should match
        // the lone slot of the B=1 pass to within fp32 noise.
        let l1 = logitsB1[0, 0, 0...].asArray(Float.self)
        let l2Slot0 = logitsB2[0, 0, 0...].asArray(Float.self)
        #expect(l1.count == l2Slot0.count)

        // Tolerance: SDPA + matmul reduction order varies with batch
        // dimension on Metal (the kernel partitions work differently
        // across `B`), so we cannot expect bitwise equality. What we
        // CAN assert is that the slot 0 output stays close — i.e. there
        // is no per-slot indexing bug where B=1 takes a fundamentally
        // different code path than B=2 slot 0. Cosine similarity is the
        // right metric: it is invariant to small fp magnitude drift but
        // catches any systematic divergence.
        var dot: Float = 0
        var n1: Float = 0
        var n2: Float = 0
        var maxAbs: Float = 0
        for i in 0..<l1.count {
            dot += l1[i] * l2Slot0[i]
            n1 += l1[i] * l1[i]
            n2 += l2Slot0[i] * l2Slot0[i]
            maxAbs = max(maxAbs, abs(l1[i] - l2Slot0[i]))
        }
        let cos = dot / (sqrt(n1) * sqrt(n2) + 1e-12)
        let detail = "B=1 slot 0 logits must match B=2 slot 0 "
            + "(cosine=\(cos), maxAbs=\(maxAbs))"
        // Cosine > 0.999 means the directions are effectively the same
        // — semantic equivalence proven, micro-noise from reduction
        // ordering ignored. Empirically the sparse path lands at >0.9999
        // on synthetic random weights.
        #expect(cos > 0.999, "\(detail)")
    }
}
