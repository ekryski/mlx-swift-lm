// SPDX-License-Identifier: Apache-2.0
// Unit tests for BatchedRetrievalAttentionIndexB.
//
// Asserts: same K + same Q across slots → same top-K; different Q per
// slot → different top-K (per-slot selection is real, not shared).

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("BatchedRetrievalAttentionIndexB (B dim)")
struct BatchedRetrievalAttentionIndexBTests {

    @Test func b2RectangularPrefillBuildsBlockFeatures() {
        let cfg = RetrievalAttentionConfig()  // λ=0 default
        let B = 2
        let nKVH = 4
        let dHead = 64
        let T = 256       // 4 fine blocks (256/64) + sub-coarse-block
        let index = BatchedRetrievalAttentionIndexB(
            config: cfg, B: B, dHead: dHead, nKVHeads: nKVH,
            ropeBase: 10_000, layerIdx: 0)
        // Build deterministic K via fixed seed so the test is reproducible.
        let keys = MLXRandom.normal(
            [B, nKVH, T, dHead], key: MLXRandom.key(42)
        ).asType(.float32)
        index.update(newKeys: keys)
        // Sanity: per-slot populated count.
        #expect(index.seqLen(slot: 0) == T)
        #expect(index.seqLen(slot: 1) == T)
        // Block feature shape: ceil(256/64) = 4 fine blocks.
        let ff = index.fineBlockFeatures
        #expect(ff != nil)
        #expect(ff!.shape == [B, nKVH, 4, cfg.contentDim])
    }

    @Test func b2PerSlotTopKIndependent() {
        // CORE BEHAVIOR — different Q for slot 0 vs slot 1 must pick
        // different top-K. If we picked the same blocks for both slots,
        // we'd be batching with a shared selector (semantically wrong).
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 64
        cfg.fineTopK = 4
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false  // keep K fixed at 4 so the assert is clean
        let B = 2
        let nKVH = 2
        let dHead = 64
        let T = 64 * 32   // 32 fine blocks
        let index = BatchedRetrievalAttentionIndexB(
            config: cfg, B: B, dHead: dHead, nKVHeads: nKVH,
            ropeBase: 10_000, layerIdx: 0)
        // SAME K across both slots so the only difference is Q.
        let oneK = MLXRandom.normal(
            [1, nKVH, T, dHead], key: MLXRandom.key(7)
        ).asType(.float32)
        let keys = concatenated([oneK, oneK], axis: 0)  // [2, nKVH, T, dHead]
        index.update(newKeys: keys)
        // Different Q per slot (so they hit different blocks).
        let q0 = MLXRandom.normal([nKVH, dHead], key: MLXRandom.key(101))
        let q1 = MLXRandom.normal([nKVH, dHead], key: MLXRandom.key(202))
        let qBatched = MLX.stacked([q0, q1], axis: 0)  // [B, nKVH, dHead]
        let projQ = index.projectQueriesBatched(qBatched)
        let fineStarts = index.topKFineBlockStarts(projectedQ: projQ)
        #expect(fineStarts.shape == [B, nKVH, cfg.fineTopK])
        eval(fineStarts)
        // Pull per-slot, per-head sets and assert at least one head's set
        // differs between slot 0 and slot 1.
        let arr = fineStarts.asArray(Int32.self)
        var anyDiffer = false
        for h in 0..<nKVH {
            let base0 = h * cfg.fineTopK
            let base1 = nKVH * cfg.fineTopK + h * cfg.fineTopK
            let slot0Set = Set((0..<cfg.fineTopK).map { Int(arr[base0 + $0]) })
            let slot1Set = Set((0..<cfg.fineTopK).map { Int(arr[base1 + $0]) })
            if slot0Set != slot1Set {
                anyDiffer = true
            }
        }
        #expect(anyDiffer, "per-slot top-K must differ when Q differs")
    }

    @Test func b2SameQSameKMatchesAcrossSlots() {
        // Symmetric to the above: SAME K, SAME Q across slots → top-K
        // sets MUST be identical (otherwise selection is non-deterministic
        // across batch position).
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 64
        cfg.fineTopK = 4
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        let B = 2
        let nKVH = 2
        let dHead = 64
        let T = 64 * 16
        let index = BatchedRetrievalAttentionIndexB(
            config: cfg, B: B, dHead: dHead, nKVHeads: nKVH,
            ropeBase: 10_000, layerIdx: 0)
        let oneK = MLXRandom.normal([1, nKVH, T, dHead], key: MLXRandom.key(11)).asType(.float32)
        let keys = concatenated([oneK, oneK], axis: 0)
        index.update(newKeys: keys)
        let q = MLXRandom.normal([nKVH, dHead], key: MLXRandom.key(13))
        let qBatched = MLX.stacked([q, q], axis: 0)
        let projQ = index.projectQueriesBatched(qBatched)
        let fineStarts = index.topKFineBlockStarts(projectedQ: projQ)
        eval(fineStarts)
        let arr = fineStarts.asArray(Int32.self)
        for h in 0..<nKVH {
            let base0 = h * cfg.fineTopK
            let base1 = nKVH * cfg.fineTopK + h * cfg.fineTopK
            let slot0Set = Set((0..<cfg.fineTopK).map { Int(arr[base0 + $0]) })
            let slot1Set = Set((0..<cfg.fineTopK).map { Int(arr[base1 + $0]) })
            #expect(slot0Set == slot1Set,
                "head \(h): same K, same Q → expected matching top-K across slots")
        }
    }

    @Test func b2GatherShapeMatchesGroupSDPASignature() {
        // The group-SDPA kernel expects gather shape [B, nKVH, K_padded] int32.
        // This test pins that the batched index emits that shape so the
        // cache can hand it directly to retrievalAttentionGroupSparseSDPA.
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 64
        cfg.fineTopK = 2
        cfg.coarseRescueEnabled = false
        cfg.adaptiveTopK = false
        cfg.staticInit = 128
        cfg.slidingWindow = 256
        let B = 2
        let nKVH = 2
        let dHead = 64
        let T = 1024
        let index = BatchedRetrievalAttentionIndexB(
            config: cfg, B: B, dHead: dHead, nKVHeads: nKVH,
            ropeBase: 10_000, layerIdx: 0)
        let keys = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(99)).asType(.float32)
        index.update(newKeys: keys)
        let q = MLXRandom.normal([B, nKVH, dHead], key: MLXRandom.key(100))
        let projQ = index.projectQueriesBatched(q)
        let (gather, kPadded) = index.perKVHeadGatherBatched(projectedQ: projQ, seqLen: T)
        // Expected K_padded = static + sliding + kFine*BS + 0 = 128 + 256 + 2*64 = 512
        #expect(kPadded == 128 + 256 + 2 * 64)
        #expect(gather.shape == [B, nKVH, kPadded])
        #expect(gather.dtype == .int32)
        // All positions clipped to [0, T-1].
        eval(gather)
        let arr = gather.asArray(Int32.self)
        let allInRange = arr.allSatisfy { $0 >= 0 && $0 < Int32(T) }
        #expect(allInRange, "gather positions must be in [0, T-1]")
    }

    @Test func decodeStepRunningMeanMatchesScratchMean() {
        // Build identical K via prefill, vs prefill-then-decode-1-token,
        // then check the partial-block mean matches a re-computed scratch
        // mean. Validates the running-mean shortcut.
        var cfg = RetrievalAttentionConfig()
        cfg.fineBlockSize = 8  // small so we hit a partial block
        cfg.coarseRescueEnabled = false
        let B = 1
        let nKVH = 1
        let dHead = 8
        let T = 13  // 1 full block (8) + partial (5)
        let keys = MLXRandom.normal([B, nKVH, T, dHead], key: MLXRandom.key(7)).asType(.float32)

        // Path A: prefill all 13 at once.
        let indexA = BatchedRetrievalAttentionIndexB(
            config: cfg, B: B, dHead: dHead, nKVHeads: nKVH,
            ropeBase: 10_000, layerIdx: 0)
        indexA.update(newKeys: keys)
        // Path B: prefill 12, then decode 1.
        let indexB = BatchedRetrievalAttentionIndexB(
            config: cfg, B: B, dHead: dHead, nKVHeads: nKVH,
            ropeBase: 10_000, layerIdx: 0)
        indexB.update(newKeys: keys[0..., 0..., ..<12, 0...])
        indexB.update(newKeys: keys[0..., 0..., 12 ..< 13, 0...])

        // Block features must match within float-rounding tolerance.
        let ffA = indexA.fineBlockFeatures!
        let ffB = indexB.fineBlockFeatures!
        #expect(ffA.shape == ffB.shape)
        let diff = (ffA - ffB).abs().max().item(Float.self)
        // Running-mean is computed via (mean_old + (added - mean_old)/count)
        // vs scratch sum/count — fp32 rounding diverges by ~1e-4 per step.
        // For 5 tokens in the partial block, ~5e-4 is within float noise.
        #expect(diff < 1e-3, "running-mean drift \(diff) too large")
    }
}
