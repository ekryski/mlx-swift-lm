// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Tape-replay rollback (spec 020 phase 1 — protocol + no-op defaults)
//
// Today's `NGramSpeculativeTokenIterator` and the new
// `DFlashSpeculativeTokenIterator` both refuse to run on hybrid models
// (Qwen 3.5 / 3.6 GatedDeltaNet, Nemotron-H, Jamba, …) because their
// Mamba / SSM layers don't have positional cache state — there's nothing
// to "trim" by N positions on rejection. dflash-mlx solves this with
// **innovation-tape rollback**: during the verify forward, record the
// per-step recurrent updates; on partial accept, replay only the
// accepted prefix's updates onto a snapshot taken at round entry.
//
// **Phase 1 scope** (this file): land the protocol + a no-op default
// conformance for the caches that already trim cleanly. The iterator's
// rollback path becomes mode-aware via `canRollbackPromptCache(_:)` and
// `rollbackPromptCache(_:acceptedPrefix:numDraft:)`. Concrete `SSMStateCache`
// conformance + the Metal replay kernel land in phase 2.
//
// Why ship phase 1 standalone: the predicate change unblocks call sites
// to start preparing for tape-replay (e.g. `MLXLMCommon.generate(...)`'s
// auto-routing can stop discriminating against hybrid caches at the
// surface) without risk to the existing trim path. Phase 2 is then a
// drop-in: `SSMStateCache` conforms, the iterators flip a flag, and the
// gate auto-engages.

/// Cost class for the rollback operation. Used by the iterator's draft
/// budget heuristic — at very high blockSize × layer-count, a `reforward`
/// is sometimes cheaper than recording deltas every cycle.
public enum TapeReplayCost: Equatable, Sendable {
    /// Constant-time rollback (linear SSMs / pure-attention trim). Cheap.
    case o1
    /// Linear in accepted-prefix length (Mamba gated delta). Mid-cost.
    case ok
    /// Falls back to re-running the layer from snapshot. Slow.
    case reforward
}

/// Cache that supports innovation-tape rollback in addition to (or
/// instead of) positional trim.
///
/// The recording session lifecycle is:
///   `beginRecord()` → zero or more `appendInnovation(...)` calls →
///   exactly one of `commitFull()`, `rollback(acceptedPrefix:)`, or
///   `cancel()`. Calling those terminators outside an active session is
///   a programmer error (precondition).
///
/// `update(...)` becomes mode-aware on conforming caches: outside a
/// recording session it behaves as today; inside one, the layer's update
/// implementation is responsible for also calling `appendInnovation(...)`
/// with the per-step delta (or a logical equivalent — for SSMs this is
/// the `B_t * x_t` term; for trim-only caches the implementation can
/// pass an empty placeholder since `rollback` will translate to `trim`).
public protocol TapeReplayCache: KVCache {
    /// Whether this cache currently supports tape replay. False on
    /// caches that only conform via the no-op default — those still
    /// route through `rollbackPromptCache` correctly because the helper
    /// falls back to `trim` for them, but the iterator can branch on
    /// this flag to skip recording overhead entirely.
    var canTapeReplay: Bool { get }

    /// Cost class — drives the iterator's choice between recording every
    /// cycle vs. snapshotting + reforward.
    var replayCost: TapeReplayCost { get }

    /// Begin recording an innovation tape. Caller commits to one of
    /// `commitFull()`, `rollback(acceptedPrefix:)`, or `cancel()` before
    /// starting a new round.
    func beginRecord()

    /// Append the next innovation. Called from inside the layer's
    /// `update(...)` during a recording session. The implementation
    /// stores the delta for later replay; the no-op default on
    /// trimmable caches discards it (correct behaviour — those caches
    /// don't *need* the delta, they trim positionally on rollback).
    func appendInnovation(_ delta: MLXArray)

    /// Accept all tape entries: cache state advances to end-of-tape.
    /// Tape buffer is cleared; cache is now in steady state.
    func commitFull()

    /// Accept only the first `k` tape entries. Tape entries `k..<count`
    /// are discarded. For positional-trim caches this is equivalent to
    /// `trim(count - k)`; for SSM caches this folds the deltas through
    /// the recurrence.
    func rollback(acceptedPrefix k: Int)

    /// Reject all tape entries: cache state restored to the pre-record
    /// snapshot. Equivalent to `rollback(acceptedPrefix: 0)` semantically
    /// but typically cheaper to dispatch — implementations may special-
    /// case the all-rejected path.
    func cancel()
}

// MARK: - Default no-op conformance via free helpers
//
// We don't want to retroactively mark every existing cache as
// `TapeReplayCache` — that's a breaking ABI change for downstream
// adopters. Instead, the rollback helpers below check for protocol
// conformance and fall back to `trim` semantics on caches that don't
// conform. This means every existing trimmable cache (KVCacheSimple,
// QuantizedKVCache, RotatingKVCache, etc.) just works with the new
// rollback API; only the hybrid caches need to opt in for phase 2.

/// Whether the cache array supports rollback on partial accept — either
/// every layer is positionally trimmable, or every non-trimmable layer
/// is a `TapeReplayCache`.
///
/// Mirrors `canTrimPromptCache` in shape but accepts the broader set of
/// caches that can be rolled back via tape replay. Used by the
/// speculative iterators to gate construction; their existing
/// `canTrimPromptCache` checks become this for hybrid-model support.
public func canRollbackPromptCache(_ cache: [KVCache]) -> Bool {
    cache.allSatisfy { layer in
        layer.isTrimmable || (layer as? TapeReplayCache)?.canTapeReplay == true
    }
}

/// Begin a recording session on every layer that supports tape replay.
/// Layers that don't support it (positional-trim caches, or `TapeReplayCache`s
/// reporting `canTapeReplay == false`) are skipped — the helper is
/// idempotent there.
public func beginCacheRecord(_ cache: [KVCache]) {
    for layer in cache {
        if let tape = layer as? TapeReplayCache, tape.canTapeReplay {
            tape.beginRecord()
        }
    }
}

/// Commit the recording session on every layer (full-accept path).
/// Layers without an active session are skipped.
public func commitCacheRecord(_ cache: [KVCache]) {
    for layer in cache {
        if let tape = layer as? TapeReplayCache, tape.canTapeReplay {
            tape.commitFull()
        }
    }
}

/// Roll back the cache by accepting only the first `acceptedPrefix`
/// drafts of a `numDraft`-token verify round. Layers that support tape
/// replay use it; trimmable layers fall back to `trim(numDraft -
/// acceptedPrefix)`. Mixed-cache stacks (some trimmable, some tape-
/// replay) are supported transparently.
///
/// - Parameter cache: per-layer cache stack.
/// - Parameter acceptedPrefix: count of accepted draft tokens (0 ≤
///   acceptedPrefix ≤ numDraft).
/// - Parameter numDraft: total drafts in the just-completed verify round.
/// - Returns: number of tokens trimmed by the *first* trimmable layer
///   (mirrors `trimPromptCache`'s return shape — pure information for
///   diagnostics, not a contract).
@discardableResult
public func rollbackPromptCache(
    _ cache: [KVCache],
    acceptedPrefix: Int,
    numDraft: Int
) -> Int {
    precondition(
        acceptedPrefix >= 0 && acceptedPrefix <= numDraft,
        "acceptedPrefix (\(acceptedPrefix)) must be in [0, numDraft=\(numDraft)]")
    let rejected = numDraft - acceptedPrefix
    var firstReturn: Int?
    for layer in cache {
        if let tape = layer as? TapeReplayCache, tape.canTapeReplay {
            // Tape-replay layer: full accept skips work; partial calls
            // through to the kernel; zero accept goes through cancel.
            if rejected == 0 {
                tape.commitFull()
            } else if acceptedPrefix == 0 {
                tape.cancel()
            } else {
                tape.rollback(acceptedPrefix: acceptedPrefix)
            }
            // Tape-replay caches don't have a meaningful "tokens
            // trimmed" return; report 0 unless the layer is also
            // trimmable.
            if firstReturn == nil { firstReturn = 0 }
        } else if layer.isTrimmable {
            let trimmed = layer.trim(rejected)
            if firstReturn == nil { firstReturn = trimmed }
        }
    }
    return firstReturn ?? 0
}

/// Cancel the recording session on every layer that supports tape
/// replay. Used by the iterator on rounds it abandons (e.g. `maxTokens`
/// reached mid-record). Trimmable-only caches are no-op.
public func cancelCacheRecord(_ cache: [KVCache]) {
    for layer in cache {
        if let tape = layer as? TapeReplayCache, tape.canTapeReplay {
            tape.cancel()
        }
    }
}
