// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Attention-aware span selector (spec 019 phase 1 scaffold)
//
// Today's `NGramLookup.proposeDraft` tiebreaks multi-candidate matches
// by recency (most-recent occurrence wins) when more than one prior
// occurrence has the same n-gram key. PLD+ proposes scoring candidate
// occurrences by **how likely the model is to "copy from" that position
// next** — the attention-weighted analog of recency. PLD+'s headline is
// 30-40% accept-rate improvement on summarisation / QA over plain PLD.
//
// Phase 1 (this file) ships:
//   - `AttentionAwareSpanSelector` protocol — given a list of candidate
//     occurrence positions and the model's recent hidden states,
//     return one score per candidate.
//   - `HiddenStateCosineSelector` — phase-1 scoring uses cosine
//     similarity of the per-position hidden state vectors at a
//     configured layer. Works model-agnostic; doesn't need induction-
//     head identification (phase 2).
//   - Pure-Swift `cosineSimilarity` helper used by the selector and
//     unit-tested independently.
//
// The model-side `HiddenStateCapture` protocol + per-model conformance
// land in phase 2 alongside the iterator wiring; this PR is the pure
// data structures + scoring math.

/// One score per candidate occurrence, indicating how likely the model
/// is to attend to (or "copy from") that position next. Higher = more
/// preferred for tiebreaking.
public protocol AttentionAwareSpanSelector: Sendable {
    /// - Parameter candidates: list of candidate first-token positions
    ///   in the lookup history. Each entry is an index into the model's
    ///   recent token history.
    /// - Parameter currentPos: the position of the most recently
    ///   accepted token (the "query position" — what the score is
    ///   evaluated relative to).
    /// - Parameter hiddenStates: layer ID → per-position hidden state
    ///   vectors, shape `[seqLen, hiddenDim]` represented as
    ///   `[[Float]]`. Phase 1 uses `[Float]` arrays so the protocol is
    ///   independent of MLX; phase 2 will add an MLX-tensor variant
    ///   when real model integration lands.
    /// - Returns: one score per candidate, in the same order. Empty
    ///   when `candidates` is empty.
    func score(
        candidates: [Int],
        currentPos: Int,
        hiddenStates: [Int: [[Float]]]
    ) -> [Float]
}

/// Phase-1 selector — cosine similarity between the hidden-state
/// vectors at `candidates[i]` and `currentPos` at the configured
/// layer. Works without induction-head identification; PLD+'s ablation
/// shows layers 9-13 work best across model sizes (the spec defaults
/// us to 11 — middle of most pretrained transformers).
public struct HiddenStateCosineSelector: AttentionAwareSpanSelector {
    public let layerID: Int

    public init(layerID: Int = 11) {
        precondition(layerID >= 0, "layerID must be >= 0 (got \(layerID))")
        self.layerID = layerID
    }

    public func score(
        candidates: [Int],
        currentPos: Int,
        hiddenStates: [Int: [[Float]]]
    ) -> [Float] {
        guard let layer = hiddenStates[layerID] else {
            // Layer not captured — return zero scores; caller falls
            // back to recency tiebreak.
            return [Float](repeating: 0, count: candidates.count)
        }
        guard currentPos > 0, currentPos - 1 < layer.count else {
            return [Float](repeating: 0, count: candidates.count)
        }
        let queryVec = layer[currentPos - 1]
        return candidates.map { j -> Float in
            guard j > 0, j - 1 < layer.count else { return 0 }
            let candVec = layer[j - 1]
            return cosineSimilarity(queryVec, candVec)
        }
    }
}

/// Recency-tiebreak selector — drop-in default that returns the same
/// scores PLD's existing tiebreaker would produce (more recent = higher
/// score, so identical to today's behavior). Lets call sites adopt the
/// `AttentionAwareSpanSelector` protocol uniformly without flipping
/// PLD+'s actual scoring on for everyone.
public struct RecencyTiebreakSelector: AttentionAwareSpanSelector {
    public init() {}
    public func score(
        candidates: [Int],
        currentPos: Int,
        hiddenStates: [Int: [[Float]]]
    ) -> [Float] {
        candidates.map { Float($0) }
    }
}

/// Test stub — returns a fixed mapping from candidate position to score.
public struct ScriptedSpanSelector: AttentionAwareSpanSelector {
    public let scoreMap: [Int: Float]
    public let defaultScore: Float

    public init(scoreMap: [Int: Float], defaultScore: Float = 0) {
        self.scoreMap = scoreMap
        self.defaultScore = defaultScore
    }

    public func score(
        candidates: [Int],
        currentPos: Int,
        hiddenStates: [Int: [[Float]]]
    ) -> [Float] {
        candidates.map { scoreMap[$0] ?? defaultScore }
    }
}

// MARK: - Cosine similarity (pure-Swift)

/// Cosine similarity of two `[Float]` vectors. Returns 0 when either
/// vector has zero magnitude (the geometrically degenerate case) — this
/// is the same convention as `MLXNN.cosineSimilarity` and avoids
/// returning NaN to callers.
///
/// - Precondition: `a.count == b.count`. Mismatched-length vectors are a
///   programmer error in this context (different layer outputs at the
///   same model would always be the same dim); we trap rather than
///   silently truncate.
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count,
        "cosineSimilarity: lengths differ (\(a.count) vs \(b.count))")
    if a.isEmpty { return 0 }
    var dot: Float = 0
    var sa: Float = 0
    var sb: Float = 0
    for i in 0 ..< a.count {
        dot += a[i] * b[i]
        sa += a[i] * a[i]
        sb += b[i] * b[i]
    }
    let denom = (sa * sb).squareRoot()
    if denom == 0 { return 0 }
    return dot / denom
}

/// Pick the highest-scoring candidate. Returns nil when `candidates` is
/// empty. Ties are broken by `candidates`-order (first-wins) — matches
/// `[Float].max`'s behaviour on equal values.
public func pickBestCandidate(
    candidates: [Int],
    scores: [Float]
) -> Int? {
    precondition(candidates.count == scores.count,
        "candidates and scores must have the same count "
            + "(got \(candidates.count) vs \(scores.count))")
    guard !candidates.isEmpty else { return nil }
    var bestIdx = 0
    var bestScore = scores[0]
    for i in 1 ..< scores.count where scores[i] > bestScore {
        bestScore = scores[i]
        bestIdx = i
    }
    return candidates[bestIdx]
}
