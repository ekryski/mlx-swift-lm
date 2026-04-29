// Copyright © 2026 Apple Inc.

import Foundation
@testable import MLXLMCommon
import Testing

// MARK: - PLD+ attention-aware span selector tests (spec 019 phase 1)
//
// Pure-Swift; no MLX evaluation. Phase 1 covers:
//   - `cosineSimilarity` math correctness on identical / orthogonal /
//     opposite / zero-vector / mixed-magnitude inputs.
//   - `pickBestCandidate` selection + tie semantics.
//   - `HiddenStateCosineSelector` end-to-end with synthetic hidden
//     states.
//   - `RecencyTiebreakSelector` matches today's PLD behaviour.
//   - `ScriptedSpanSelector` returns prescribed scores; default applies.

@Suite
struct CosineSimilarityTests {

    @Test
    func `Identical vectors return 1`() {
        #expect(abs(cosineSimilarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6)
    }

    @Test
    func `Opposite vectors return -1`() {
        #expect(abs(cosineSimilarity([1, 2, 3], [-1, -2, -3]) + 1.0) < 1e-6)
    }

    @Test
    func `Orthogonal vectors return 0`() {
        #expect(abs(cosineSimilarity([1, 0], [0, 1])) < 1e-6)
    }

    @Test
    func `Zero magnitude returns 0 (no NaN)`() {
        #expect(cosineSimilarity([0, 0, 0], [1, 2, 3]) == 0)
        #expect(cosineSimilarity([1, 2, 3], [0, 0, 0]) == 0)
        #expect(cosineSimilarity([0, 0], [0, 0]) == 0)
    }

    @Test
    func `Empty vectors return 0`() {
        #expect(cosineSimilarity([], []) == 0)
    }

    @Test
    func `Magnitude does not affect cosine`() {
        let a: [Float] = [1, 2, 3]
        let b: [Float] = [10, 20, 30]
        #expect(abs(cosineSimilarity(a, b) - 1.0) < 1e-6)
    }
}

// MARK: - Best-candidate picker

@Suite
struct PickBestCandidateTests {

    @Test
    func `Empty input returns nil`() {
        #expect(pickBestCandidate(candidates: [], scores: []) == nil)
    }

    @Test
    func `Picks highest-scoring candidate`() {
        let cands = [10, 20, 30]
        let scores: [Float] = [0.1, 0.9, 0.5]
        #expect(pickBestCandidate(candidates: cands, scores: scores) == 20)
    }

    @Test
    func `Tie breaks on first-wins`() {
        let cands = [10, 20, 30]
        let scores: [Float] = [0.5, 0.5, 0.5]
        #expect(pickBestCandidate(candidates: cands, scores: scores) == 10)
    }

    @Test
    func `Single candidate returns itself`() {
        #expect(pickBestCandidate(candidates: [42], scores: [0.0]) == 42)
    }
}

// MARK: - HiddenStateCosineSelector

@Suite
struct HiddenStateCosineSelectorTests {

    @Test
    func `Returns zero scores when layer not captured`() {
        let s = HiddenStateCosineSelector(layerID: 11)
        let scores = s.score(
            candidates: [1, 2, 3],
            currentPos: 5,
            hiddenStates: [:])  // empty
        #expect(scores == [0, 0, 0])
    }

    @Test
    func `Returns zero scores when currentPos is invalid`() {
        let s = HiddenStateCosineSelector(layerID: 11)
        let layer: [[Float]] = [[1, 0], [0, 1]]
        // currentPos = 0 → currentPos - 1 = -1, invalid.
        let scores = s.score(
            candidates: [1, 2],
            currentPos: 0,
            hiddenStates: [11: layer])
        #expect(scores == [0, 0])
    }

    @Test
    func `Identical hidden states score 1`() {
        let s = HiddenStateCosineSelector(layerID: 11)
        let layer: [[Float]] = [
            [1, 2, 3],  // pos 0
            [1, 2, 3],  // pos 1 — same as pos 0
            [4, 5, 6],  // pos 2 — different
        ]
        // currentPos = 2 → query = layer[1] = [1,2,3].
        // Candidate 1 → layer[0] = [1,2,3] — identical, score 1.
        // Candidate 3 → layer[2] = [4,5,6] — different.
        let scores = s.score(
            candidates: [1, 3],
            currentPos: 2,
            hiddenStates: [11: layer])
        #expect(abs(scores[0] - 1.0) < 1e-6)
        #expect(scores[1] < 1.0)
        #expect(scores[1] > 0)  // non-trivial cosine
    }

    @Test
    func `Out-of-range candidate position scores zero`() {
        let s = HiddenStateCosineSelector(layerID: 11)
        let layer: [[Float]] = [[1, 2], [3, 4]]
        // Candidate 99 is out of range.
        let scores = s.score(
            candidates: [1, 99],
            currentPos: 2,
            hiddenStates: [11: layer])
        #expect(scores[0] != 0)  // valid
        #expect(scores[1] == 0)  // out of range → 0
    }
}

// MARK: - RecencyTiebreakSelector

@Suite
struct RecencyTiebreakSelectorTests {

    @Test
    func `Score equals candidate position (recency-monotone)`() {
        let s = RecencyTiebreakSelector()
        let scores = s.score(
            candidates: [10, 25, 100],
            currentPos: 200,
            hiddenStates: [:])
        #expect(scores == [10, 25, 100])
    }

    @Test
    func `Most recent candidate wins through pickBestCandidate`() {
        let s = RecencyTiebreakSelector()
        let cands = [10, 25, 100]
        let scores = s.score(candidates: cands, currentPos: 200, hiddenStates: [:])
        #expect(pickBestCandidate(candidates: cands, scores: scores) == 100)
    }
}

// MARK: - ScriptedSpanSelector

@Suite
struct ScriptedSpanSelectorTests {

    @Test
    func `Returns scripted scores in candidate order`() {
        let s = ScriptedSpanSelector(scoreMap: [10: 0.9, 20: 0.5, 30: 0.1])
        let scores = s.score(
            candidates: [30, 10, 20],
            currentPos: 0,
            hiddenStates: [:])
        #expect(scores == [0.1, 0.9, 0.5])
    }

    @Test
    func `Unmapped candidates use defaultScore`() {
        let s = ScriptedSpanSelector(scoreMap: [10: 0.9], defaultScore: -1)
        let scores = s.score(
            candidates: [10, 99, 20],
            currentPos: 0,
            hiddenStates: [:])
        #expect(scores == [0.9, -1, -1])
    }
}
