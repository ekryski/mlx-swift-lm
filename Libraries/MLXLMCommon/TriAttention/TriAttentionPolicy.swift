// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// TriAttention V3 selection policy — Swift / MLX port of
// vllm-turboquant/vllm/v1/attention/triattention/policy.py.
//
// Score input convention: HIGHER score = evict first (the trig formula
// in TriAttentionScoring measures orthogonality to the calibration
// center, not alignment to it).
//
// Modes (matches the Python `hybridMode` field):
//   0 = V1: global sort, evict highest-scoring N (excluding window).
//   1 = V2: per-segment quota, no prefix protection.
//   2 = V3: per-segment quota + first prefix_protect tokens protected.
//
// Returns Int positions to evict — caller stamps `valid_mask = false`
// at those positions and fires the Tier 2 callback. The function works
// in MLX tensor space for the masking but does the small per-segment
// loop on the CPU side; eviction sets are O(100s-1000s) cells, so the
// loop overhead is negligible vs the scoring matmul cost.
import Foundation
import MLX

public enum TriAttentionPolicy {

    /// Select positions to evict. `windowThr`/`prefixLo` are inclusive
    /// boundaries: positions in `[0, prefixLo)` and `[windowThr, seqLen)`
    /// are protected. Returns at most `nToEvict` positions.
    public static func selectEvictions(
        scores: MLXArray,        // [seqLen] f32 — higher = evict first
        valid: MLXArray,         // [seqLen] bool
        nToEvict: Int,
        windowThr: Int,
        prefixLo: Int,
        nSegments: Int,
        mode: Int = 2
    ) -> [Int] {
        guard nToEvict > 0 else { return [] }
        let seqLen = scores.dim(0)

        // Build candidate positions on CPU: MLX Swift doesn't support
        // boolean indexing (`positions[bool_mask]` crashes at gather).
        // Iterate the materialized arrays instead.
        let validArr: [Bool] = valid.asArray(Bool.self)
        let scoresArr: [Float] = scores.asArray(Float.self)
        var candPosArr: [Int32] = []
        var candScoreArr: [Float] = []
        candPosArr.reserveCapacity(seqLen)
        candScoreArr.reserveCapacity(seqLen)
        for p in 0..<seqLen {
            if !validArr[p] { continue }
            if p >= windowThr { continue }
            if mode == 2, p < prefixLo { continue }
            candPosArr.append(Int32(p))
            candScoreArr.append(scoresArr[p])
        }
        guard !candPosArr.isEmpty else { return [] }

        if mode == 0 {
            // V1: global top-N over all candidates.
            let nTake = min(nToEvict, candPosArr.count)
            let pairs = zip(candScoreArr, candPosArr).map { ($0, Int($1)) }
            let sorted = pairs.sorted { $0.0 > $1.0 }
            return sorted.prefix(nTake).map { $0.1 }
        }

        // V2/V3: per-segment quota.
        let k = max(1, nSegments)
        let posHi = max(windowThr, prefixLo + 1)
        let segWidth = max(1.0, Float(posHi - prefixLo) / Float(k))

        let totalEligible = candPosArr.count
        let targetFrac =
            totalEligible > 0 ? Float(nToEvict) / Float(totalEligible) : 0.0

        // Bucket candidates by segment.
        var buckets: [[(score: Float, pos: Int)]] =
            Array(repeating: [], count: k)
        for i in 0..<candPosArr.count {
            let p = Int(candPosArr[i])
            var seg = Int(Float(p - prefixLo) / segWidth)
            if seg < 0 { seg = 0 }
            if seg >= k { seg = k - 1 }
            buckets[seg].append((candScoreArr[i], p))
        }

        var chosen: [Int] = []
        chosen.reserveCapacity(nToEvict)
        for s in 0..<k {
            if buckets[s].isEmpty { continue }
            let bucketTarget = Int(Float(buckets[s].count) * targetFrac)
            let nTake = min(
                bucketTarget, buckets[s].count, nToEvict - chosen.count)
            if nTake <= 0 { continue }
            let sorted = buckets[s].sorted { $0.score > $1.score }
            for i in 0..<nTake { chosen.append(sorted[i].pos) }
        }

        // Cleanup: if rounding left a deficit, fill from any remaining
        // candidates ranked by score. Mirrors the Python cleanup pass.
        if chosen.count < nToEvict {
            let alreadySet = Set(chosen)
            var leftovers: [(score: Float, pos: Int)] = []
            for i in 0..<candPosArr.count {
                let p = Int(candPosArr[i])
                if alreadySet.contains(p) { continue }
                leftovers.append((candScoreArr[i], p))
            }
            leftovers.sort { $0.score > $1.score }
            let needed = nToEvict - chosen.count
            for i in 0..<min(needed, leftovers.count) {
                chosen.append(leftovers[i].pos)
            }
        }
        return chosen
    }
}
