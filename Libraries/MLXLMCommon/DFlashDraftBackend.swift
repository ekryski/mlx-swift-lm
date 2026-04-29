// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Draft-side protocol surface for DFlash speculative decoding
//
// The DFlash draft is a small block-diffusion transformer that, given the
// target's most-recent hidden states at a configured set of capture layers
// and the last committed token, emits K (typically 16) candidate
// continuation tokens in **one** forward pass. The target then verifies
// all K in one batched forward pass and accepts the longest matching
// prefix.
//
// This file defines the **draft-side** protocol — what a draft backend
// must expose to ``DFlashSpeculativeTokenIterator``. Phase 1 ships two
// implementations:
//
//   - ``ZeroAcceptDraftBackend`` — always returns a constant non-matching
//     token. Lets the iterator test the "all rejected" path end-to-end
//     without GPU model weights, exercising the rollback and bonus-token
//     emission code paths. Acceptance rate is by construction zero.
//   - ``ScriptedDraftBackend`` — replays a caller-provided list of blocks.
//     Tests use this to drive specific accept/reject patterns.
//
// Phase 2 will add a real ``DFlashTransformerDraftBackend`` that loads
// trained weights from a HuggingFace draft repo (e.g.
// `z-lab/Qwen3.5-9B-DFlash`) and runs a block-diffusion forward.

/// A draft-side backend for DFlash speculative decoding.
///
/// Implementations encapsulate everything the iterator needs to know about
/// the draft model: the iterator does not own draft KV state, draft model
/// weights, or block-diffusion specifics. This lets us swap in a stub for
/// scaffold-time testing, a scripted backend for unit tests, and the real
/// transformer backend without iterator changes.
public protocol DFlashDraftBackend {
    /// The fixed block size this backend emits per ``draftBlock`` call.
    /// dflash-mlx uses 16 by default; smaller drafts are valid for
    /// experimentation.
    var blockSize: Int { get }

    /// The set of target-side layer IDs whose post-block hidden states this
    /// backend wants to consume. Returned by the registry / draft config so
    /// the iterator knows which layers to ask the target to capture during
    /// verify forward.
    var captureLayerIDs: Set<Int> { get }

    /// Generate one block of `blockSize` candidate tokens, conditioned on
    /// the target's captured hidden states and the last committed token.
    ///
    /// - Parameter targetHidden: `layerID -> [B=1, T=1, H]` hidden states
    ///   captured at the configured layers during the previous verify
    ///   forward (or the prefill forward, on the first call). May be empty
    ///   on the very first cycle if the iterator has not yet captured any
    ///   target hidden state — backends should handle that case (the stub
    ///   backends do; the real diffusion backend will warm-start from the
    ///   target's prefill capture).
    /// - Parameter lastCommittedToken: The most recently emitted token from
    ///   the target (the seed for the next block).
    /// - Returns: `blockSize` candidate token IDs. Returning fewer is
    ///   permitted (e.g. when generation is approaching `maxTokens`); the
    ///   iterator clamps the verify batch shape to the returned length.
    mutating func draftBlock(
        targetHidden: [Int: MLXArray],
        lastCommittedToken: Int
    ) -> [Int]

    /// Reset any internal draft KV state — called on iterator init and any
    /// time the iterator wants to discard accumulated draft context (e.g.
    /// reuse across requests). Stub backends typically no-op.
    mutating func reset()
}

// MARK: - Phase-1 stub backends

/// Phase-1 stub draft backend that emits a single repeated token for every
/// position in the block. Designed to never match the target's argmax, so
/// the iterator runs the full "draft → verify → reject all → emit bonus
/// → trim cache" cycle on every round.
///
/// Acceptance rate is by construction `0/blockSize` per cycle. Throughput
/// is _slower_ than baseline `TokenIterator` (because the verify forward
/// processes `blockSize+1` positions instead of 1). This is intentional:
/// the goal is to validate the cycle plumbing — protocol conformances,
/// hidden-state capture, verify forward, accept-prefix computation,
/// cache trim — before we add the real diffusion draft model in phase 2.
public struct ZeroAcceptDraftBackend: DFlashDraftBackend {
    public let blockSize: Int
    public let captureLayerIDs: Set<Int>

    /// The "draft" we emit each round. Picked to be a token that the target
    /// is extremely unlikely to argmax to (negative would be invalid; we
    /// settle for token 0 since an EOS or pad token usually sits there for
    /// most tokenizers but we do _not_ depend on that — we depend on the
    /// target argmaxing to something, and as long as it's not 0 we get the
    /// "all rejected" path exercised). On the rare prompt where the target
    /// _does_ argmax to 0, the cycle still completes correctly — it just
    /// happens to accept one token per cycle for that prompt.
    public let stubToken: Int

    public init(blockSize: Int = 16, captureLayerIDs: Set<Int> = [], stubToken: Int = 0) {
        precondition(blockSize >= 1, "blockSize must be >= 1 (got \(blockSize))")
        self.blockSize = blockSize
        self.captureLayerIDs = captureLayerIDs
        self.stubToken = stubToken
    }

    public mutating func draftBlock(
        targetHidden: [Int: MLXArray],
        lastCommittedToken: Int
    ) -> [Int] {
        Array(repeating: stubToken, count: blockSize)
    }

    public mutating func reset() {}
}

/// Test-only draft backend that replays a caller-supplied list of blocks
/// in order. When the script is exhausted, returns an empty block (which
/// the iterator treats as "skip this round, fall through to AR decode").
///
/// Used by the unit tests to drive specific accept/reject patterns
/// deterministically — e.g. "first cycle: matches first 3 of 4, rejects
/// the 4th; second cycle: full reject" — without needing a real draft
/// model.
public struct ScriptedDraftBackend: DFlashDraftBackend {
    public let blockSize: Int
    public let captureLayerIDs: Set<Int>
    private var script: [[Int]]
    private var cursor: Int = 0

    /// Number of `draftBlock` calls served so far. Tests use this to assert
    /// the iterator made the expected number of round-trips.
    public var callCount: Int { cursor }

    public init(
        blockSize: Int,
        script: [[Int]],
        captureLayerIDs: Set<Int> = []
    ) {
        precondition(blockSize >= 1, "blockSize must be >= 1 (got \(blockSize))")
        for (i, block) in script.enumerated() {
            precondition(
                block.count <= blockSize,
                "ScriptedDraftBackend block[\(i)] has \(block.count) tokens, "
                    + "exceeds blockSize=\(blockSize)")
        }
        self.blockSize = blockSize
        self.script = script
        self.captureLayerIDs = captureLayerIDs
    }

    public mutating func draftBlock(
        targetHidden: [Int: MLXArray],
        lastCommittedToken: Int
    ) -> [Int] {
        guard cursor < script.count else { return [] }
        defer { cursor += 1 }
        return script[cursor]
    }

    public mutating func reset() {
        cursor = 0
    }
}

// MARK: - Pure-Swift accept-prefix helper
//
// The iterator's verify path computes the accepted-prefix length by
// matching draft tokens against target argmax position-by-position. We
// factor that out as a pure-Swift function so it can be unit-tested
// without standing up a model + cache. Mirrors dflash-mlx's
// `cumprod`-over-equality-mask (the first zero in the cumulative
// product marks the first mismatch); we use the linear-scan form because
// at K=16 the constant factor is irrelevant and the linear form is
// the easier port target.

/// Compute the longest matching prefix length between `draft` and
/// `targetArgmax`. Both arrays are tokens at the same set of verify
/// positions; the iterator emits exactly `acceptedPrefixLength + 1`
/// tokens — the matching prefix plus the target's "bonus" token at the
/// first mismatch (or the position past the matching prefix on full
/// accept).
///
/// - Parameter draft: candidate tokens from the draft backend.
/// - Parameter targetArgmax: tokens the target argmaxes to at each
///   verify position. Must have at least `draft.count + 1` entries
///   (one bonus position past the draft).
/// - Returns: the largest `n` such that `draft[0..<n] == targetArgmax[0..<n]`.
public func dflashAcceptedPrefixLength(
    draft: [Int],
    targetArgmax: [Int]
) -> Int {
    precondition(
        targetArgmax.count >= draft.count + 1,
        "targetArgmax must contain at least draft.count + 1 entries "
            + "(got \(targetArgmax.count) for draft \(draft.count))")
    for i in 0 ..< draft.count {
        if draft[i] != targetArgmax[i] {
            return i
        }
    }
    return draft.count
}
