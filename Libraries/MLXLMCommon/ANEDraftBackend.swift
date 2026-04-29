// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - ANE-side draft backend protocol surface (Mirror Speculative Decoding)
//
// Spec 021's headline path: a small draft model running on the Apple
// Neural Engine via Core ML, paired with an MLX target on GPU. Different
// physical compute units, so they can run in parallel — the ANE drafts
// while the GPU verifies. This file declares the **draft-side protocol**
// the Mirror SD iterator uses to talk to whichever Core ML draft is
// loaded for the current target.
//
// The shape is deliberately parallel to ``DFlashDraftBackend`` (spec 015):
// the iterator owns the cycle; the backend owns the draft model + its
// internal state. Same swappability for testing.
//
// **Phase 1A status:** this file ships the protocol + a scripted test
// stub + a vocabulary-equivalence gate. The actual Core ML ANE backend
// lands in Phase 1B alongside the `CoreML-LLM` dependency decision; the
// scaffold here lets us write the iterator + registry + tests without
// pulling Core ML into the build yet.

/// Draft backend for Mirror Speculative Decoding. The Mirror SD iterator
/// asks this backend for K candidate continuation tokens given the last
/// committed token; the backend implementation decides where the draft
/// runs (ANE via Core ML, ANE via the private NeuralEngine API, or a
/// pure-CPU stub for tests).
///
/// Unlike ``DFlashDraftBackend`` the Mirror SD draft does *not* receive
/// target hidden states — Mirror SD pairs an independently-trained draft
/// with the target through token I/O alone. That keeps the cross-
/// framework plumbing trivially shippable: just an `[Int]` round-trips
/// between MLX and Core ML; no tensor interop required.
public protocol ANEDraftBackend {
    /// Number of candidate tokens emitted per cycle. The iterator builds
    /// a `[1, K+1]` verify batch for the target; smaller K reduces verify
    /// cost when the draft's accept rate is low.
    var draftLength: Int { get }

    /// Generate `draftLength` candidate tokens conditioned on the
    /// recently-committed token sequence. Implementations are responsible
    /// for managing their own KV state across calls — the iterator never
    /// touches the draft's cache.
    ///
    /// - Parameter committedTokens: full prefix of accepted target tokens
    ///   so far. The backend may use this to warm a fresh KV cache or to
    ///   resync after rejection — most backends only need the suffix.
    /// - Parameter lastCommittedToken: the most recent accepted target
    ///   token (also `committedTokens.last`, when non-empty). Passed
    ///   separately so backends that only condition on the seed don't
    ///   need to subscript.
    /// - Returns: `draftLength` candidate token IDs.
    mutating func draftBlock(
        committedTokens: [Int],
        lastCommittedToken: Int
    ) -> [Int]

    /// Reset the backend's internal draft KV state. Called on iterator
    /// init and on iterator-level rollbacks the backend cannot itself
    /// observe (e.g. the iterator advancing past `maxTokens`).
    mutating func reset()

    /// Bytes held by the backend's internal cache. Returned for parity
    /// with ``KVCache/memoryBytes`` accounting on the GPU side; nil when
    /// the backend doesn't expose a measurable footprint.
    var draftCacheMemoryBytes: Int? { get }
}

extension ANEDraftBackend {
    public var draftCacheMemoryBytes: Int? { nil }
}

// MARK: - Scripted test backend

/// Test-only backend that replays a pre-baked list of draft blocks. Mirrors
/// ``ScriptedDraftBackend`` (DFlash side) so the test surface is the same
/// pattern across all speculative paths: drive the iterator with a
/// deterministic accept/reject sequence and assert on bookkeeping.
///
/// When the script is exhausted, returns an empty block — the iterator
/// treats that as "fall through to AR decode" (same convention as
/// ``DFlashDraftBackend``).
public struct ScriptedANEDraftBackend: ANEDraftBackend {
    public let draftLength: Int
    private var script: [[Int]]
    private var cursor: Int = 0

    /// Number of `draftBlock` calls served so far (excludes empty
    /// fallback returns).
    public var callCount: Int { cursor }

    public init(draftLength: Int, script: [[Int]]) {
        precondition(draftLength >= 1, "draftLength must be >= 1 (got \(draftLength))")
        for (i, block) in script.enumerated() {
            precondition(
                block.count <= draftLength,
                "ScriptedANEDraftBackend block[\(i)] has \(block.count) tokens, "
                    + "exceeds draftLength=\(draftLength)")
        }
        self.draftLength = draftLength
        self.script = script
    }

    public mutating func draftBlock(
        committedTokens: [Int],
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

/// Always-rejected stub — emits a constant out-of-vocabulary token for
/// every draft slot. Used by the iterator's correctness tests to exercise
/// the "verify rejects everything, emit only the bonus" path on every
/// cycle. By construction zero acceptance.
public struct ZeroAcceptANEDraftBackend: ANEDraftBackend {
    public let draftLength: Int
    public let stubToken: Int

    public init(draftLength: Int = 4, stubToken: Int = 0) {
        precondition(draftLength >= 1, "draftLength must be >= 1 (got \(draftLength))")
        self.draftLength = draftLength
        self.stubToken = stubToken
    }

    public mutating func draftBlock(
        committedTokens: [Int],
        lastCommittedToken: Int
    ) -> [Int] {
        Array(repeating: stubToken, count: draftLength)
    }

    public mutating func reset() {}
}
