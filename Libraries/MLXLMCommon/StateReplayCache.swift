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
// **state-replay rollback**: during the verify forward, record the
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
// to start preparing for state-replay (e.g. `MLXLMCommon.generate(...)`'s
// auto-routing can stop discriminating against hybrid caches at the
// surface) without risk to the existing trim path. Phase 2 is then a
// drop-in: `SSMStateCache` conforms, the iterators flip a flag, and the
// gate auto-engages.

/// Cost class for the rollback operation. Used by the iterator's draft
/// budget heuristic — at very high blockSize × layer-count, a `reforward`
/// is sometimes cheaper than recording deltas every cycle.
public enum StateReplayCost: Equatable, Sendable {
    /// Constant-time rollback (linear SSMs / pure-attention trim). Cheap.
    case o1
    /// Linear in accepted-prefix length (Mamba gated delta). Mid-cost.
    case ok
    /// Falls back to re-running the layer from snapshot. Slow.
    case reforward
}

/// Typed error surface for the state-replay lifecycle. Adopted per dflash-mlx
/// upstream commit `4bc72c8` (2026-05-10) — fail fast on cache contract
/// violations rather than silently no-op.
///
/// Note: the canonical `SSMStateCache` conformance uses Swift `precondition`s
/// for these (programmer errors, sole caller is the iterator) — the error
/// enum exists for protocol conformers that want a throws-based discipline
/// (e.g., trim-fallback adapters in future cache types).
public enum StateReplayCacheError: Error, Equatable {
    /// `beginRecord()` called while a log is already active.
    case alreadyRecording
    /// `recordStep` / `commitFull` / `rollback` / `cancel` called without
    /// an active recording session.
    case notRecording
    /// `recordStep` received the wrong number of tensors for this cache type.
    case arityMismatch(expected: Int, got: Int)
    /// `rollback(acceptedPrefix:)` called with `k` outside `[0, log.count]`.
    case outOfRange(k: Int, logLength: Int)
}

/// Cache that supports state-replay rollback in addition to (or instead of)
/// positional trim.
///
/// The recording session lifecycle is:
///   `beginRecord()` → zero or more `recordStep(...)` calls →
///   exactly one of `commitFull()`, `rollback(acceptedPrefix:)`, or
///   `cancel()`. Calling those terminators outside an active session is
///   a programmer error (precondition).
///
/// `update(...)` becomes mode-aware on conforming caches: outside a
/// recording session it behaves as today; inside one, the layer's update
/// implementation is responsible for also calling `recordStep(...)`
/// with the per-step recurrence tensors (for SSMs this is the
/// `(delta_t, k_t, g_t)` triple; for trim-only caches the implementation
/// can pass an empty placeholder since `rollback` will translate to `trim`).
public protocol StateReplayCache: KVCache {
    /// Whether this cache currently supports state replay. False on
    /// caches that only conform via the no-op default — those still
    /// route through `rollbackPromptCache` correctly because the helper
    /// falls back to `trim` for them, but the iterator can branch on
    /// this flag to skip recording overhead entirely.
    var canStateReplay: Bool { get }

    /// Cost class — drives the iterator's choice between recording every
    /// cycle vs. snapshotting + reforward.
    var replayCost: StateReplayCost { get }

    /// Begin recording a delta log. Caller commits to one of
    /// `commitFull()`, `rollback(acceptedPrefix:)`, or `cancel()` before
    /// starting a new round.
    func beginRecord()

    /// Append the next per-step recurrence tensors. Called from inside
    /// the layer's `update(...)` during a recording session.
    ///
    /// The shape of `tensors` is per-cache-type:
    /// - `SSMStateCache` (GDN, Qwen 3.5 / 3.6 / Nemotron-H / Jamba): the
    ///   per-step `[delta_t, k_t, g_t]` triple that the `state_replay`
    ///   Metal kernel needs to re-fold the recurrence.
    /// - Trim-only caches: ignored; the no-op default discards them
    ///   (correct behaviour — those caches don't *need* the per-step
    ///   tensors, they trim positionally on rollback).
    ///
    /// The protocol takes `[MLXArray]` rather than a typed step struct so
    /// future SSM variants (Mamba 2, S4, …) can use the same surface
    /// without re-shaping the protocol.
    func recordStep(_ tensors: [MLXArray])

    /// Accept all delta-log entries: cache state advances to end-of-log.
    /// Tape buffer is cleared; cache is now in steady state.
    func commitFull()

    /// Accept only the first `k` delta-log entries. Delta-log entries `k..<count`
    /// are discarded. For positional-trim caches this is equivalent to
    /// `trim(count - k)`; for SSM caches this folds the per-step tensors
    /// through the recurrence.
    func rollback(acceptedPrefix k: Int)

    /// Reject all delta-log entries: cache state restored to the pre-record
    /// snapshot. Equivalent to `rollback(acceptedPrefix: 0)` semantically
    /// but typically cheaper to dispatch — implementations may special-
    /// case the all-rejected path.
    func cancel()
}

// MARK: - Default no-op conformance via free helpers
//
// We don't want to retroactively mark every existing cache as
// `StateReplayCache` — that's a breaking ABI change for downstream
// adopters. Instead, the rollback helpers below check for protocol
// conformance and fall back to `trim` semantics on caches that don't
// conform. This means every existing trimmable cache (KVCacheSimple,
// QuantizedKVCache, RotatingKVCache, etc.) just works with the new
// rollback API; only the hybrid caches need to opt in for phase 2.

/// Whether the cache array supports rollback on partial accept — either
/// every layer is positionally trimmable, or every non-trimmable layer
/// is a `StateReplayCache`.
///
/// Mirrors `canTrimPromptCache` in shape but accepts the broader set of
/// caches that can be rolled back via state replay. Used by the
/// speculative iterators to gate construction; their existing
/// `canTrimPromptCache` checks become this for hybrid-model support.
public func canRollbackPromptCache(_ cache: [KVCache]) -> Bool {
    cache.allSatisfy { layer in
        layer.isTrimmable || (layer as? StateReplayCache)?.canStateReplay == true
    }
}

/// Begin a recording session on every layer that supports state replay.
/// Layers that don't support it (positional-trim caches, or `StateReplayCache`s
/// reporting `canStateReplay == false`) are skipped — the helper is
/// idempotent there.
public func beginCacheRecord(_ cache: [KVCache]) {
    for layer in cache {
        if let replay = layer as? StateReplayCache, replay.canStateReplay {
            replay.beginRecord()
        }
    }
}

/// Commit the recording session on every layer (full-accept path).
/// Layers without an active session are skipped.
public func commitCacheRecord(_ cache: [KVCache]) {
    for layer in cache {
        if let replay = layer as? StateReplayCache, replay.canStateReplay {
            replay.commitFull()
        }
    }
}

/// Roll back the cache by accepting only the first `acceptedPrefix`
/// drafts of a `numDraft`-token verify round. Layers that support state
/// replay use it; trimmable layers fall back to `trim(numDraft -
/// acceptedPrefix)`. Mixed-cache stacks (some trimmable, some state-
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
        if let replay = layer as? StateReplayCache, replay.canStateReplay {
            // Tape-replay layer: full accept skips work; partial calls
            // through to the kernel; zero accept goes through cancel.
            if rejected == 0 {
                replay.commitFull()
            } else if acceptedPrefix == 0 {
                replay.cancel()
            } else {
                replay.rollback(acceptedPrefix: acceptedPrefix)
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

/// Cancel the recording session on every layer that supports state
/// replay. Used by the iterator on rounds it abandons (e.g. `maxTokens`
/// reached mid-record). Trimmable-only caches are no-op.
public func cancelCacheRecord(_ cache: [KVCache]) {
    for layer in cache {
        if let replay = layer as? StateReplayCache, replay.canStateReplay {
            replay.cancel()
        }
    }
}
