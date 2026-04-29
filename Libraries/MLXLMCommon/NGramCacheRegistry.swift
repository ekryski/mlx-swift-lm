// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Cross-request n-gram cache + registry (spec 016 phases 1-2)
//
// Today's `NGramSpeculativeTokenIterator` rebuilds its lookup table per
// request from `prompt + history` only. Multi-turn chat throws away the
// previous turn's lookup at request end, even though every turn shares
// system prompt + accumulated history. Spec 016 keeps the lookup alive
// across requests via a process-level registry; with the dynamic tier
// the lookup grows continuously over a session instead of resetting.
//
// **Phase 1-2 scope** (this file):
//   - `NGramCache` — token-history container with three tiers (context,
//     dynamic, static). Phase 1 lives in the context tier; phase 2
//     adds the dynamic-tier promotion lifecycle.
//   - `NGramCacheRegistry` — process-level singleton keyed by
//     `(modelID, ngramRange)`. Iterator looks up or creates the cache
//     for the current model on entry; commits context tokens into the
//     dynamic tier on exit.
//   - `NGramCacheThresholds` — per-tier threshold profile (lax /
//     strict / static), porting llama.cpp's `ngram-cache.cpp`
//     constants.
//
// Phase 3 (disk persistence + static-cache bootstrap) and phase 4
// (three-tier draft selection in the iterator) land in follow-up PRs.

/// Per-tier confidence thresholds for ngram-cache draft selection. Mirrors
/// llama.cpp's `draft_min_sample_size_*` and `draft_min_percent_*`
/// constants from `ngram-cache.cpp`. Each tier has its own profile;
/// strict is for less-trusted accumulated history, lax for the current
/// request's own context.
public struct NGramCacheThresholds: Equatable, Sendable {
    /// Minimum total observations of the n-gram before its top-1
    /// continuation is admissible. Index 0 is for ngram size 1, index 1
    /// for size 2, etc. — same shape as llama.cpp's
    /// `draft_min_sample_size_*`.
    public let minSampleSize: [Int]

    /// Minimum percentage (0-100) of the top-1 continuation's share of
    /// total observations. Same indexing as `minSampleSize`.
    public let minPercent: [Int]

    public init(minSampleSize: [Int], minPercent: [Int]) {
        precondition(
            minSampleSize.count == minPercent.count && !minSampleSize.isEmpty,
            "minSampleSize and minPercent must be non-empty and same length")
        for v in minPercent {
            precondition(v >= 0 && v <= 100,
                "minPercent values must be in [0, 100] (got \(v))")
        }
        self.minSampleSize = minSampleSize
        self.minPercent = minPercent
    }

    /// llama.cpp's lax profile (context tier).
    public static let lax = NGramCacheThresholds(
        minSampleSize: [2, 2, 1, 1],
        minPercent: [66, 50, 50, 50])

    /// llama.cpp's strict profile (dynamic tier).
    public static let strict = NGramCacheThresholds(
        minSampleSize: [4, 3, 2, 2],
        minPercent: [75, 66, 66, 66])

    /// Threshold for the static / pre-trained tier. Same shape as
    /// strict but used as a validator (multiply primary count by static
    /// count); admission policy is identical.
    public static let staticTier = NGramCacheThresholds.strict

    /// Look up the threshold pair for a given ngram size. Returns the
    /// tightest entry available; for sizes above the array length, uses
    /// the last entry.
    public func thresholds(forNgramSize n: Int) -> (sampleSize: Int, percent: Int) {
        precondition(n >= 1, "ngram size must be >= 1 (got \(n))")
        let idx = Swift.min(n - 1, minSampleSize.count - 1)
        return (minSampleSize[idx], minPercent[idx])
    }
}

/// Tiered n-gram cache for a single (model, ngram-range) pair.
///
/// Three tiers:
///   - `.context`: the current request's history (prompt + accepted
///     drafts). Reset on every new request — cheap.
///   - `.dynamic`: accumulated across all prior requests in this
///     process. Survives request boundaries; subject to LRU-ish
///     eviction at `maxTokens`.
///   - `.static`: loaded from disk; read-only at runtime. Phase 3 will
///     wire load/save; phase 1-2 ship the storage but no I/O.
///
/// Phase 1 keeps lookup alive across requests by re-using the same
/// `NGramCache` instance via the registry. Phase 2 adds the
/// `commitContext(...)` lifecycle that promotes context tokens into
/// the dynamic tier at request end.
public final class NGramCache: @unchecked Sendable {

    public enum Tier: Equatable, Hashable, Sendable {
        case context
        case dynamic
        case `static`
    }

    public let modelID: String
    public let ngramRange: ClosedRange<Int>
    public let maxTokens: Int

    /// Tokens in the current request's context tier. Cleared on each
    /// `resetContext()` (typically called at the start of a new
    /// request); promoted into `dynamic` via `commitContext()` at the
    /// end of a successful generation.
    public private(set) var context: [Int] = []

    /// Tokens in the dynamic tier — accumulated across requests.
    /// FIFO-truncated when length exceeds `maxTokens`.
    public private(set) var dynamic: [Int] = []

    /// Tokens in the static tier — loaded from disk in phase 3.
    /// Read-only at runtime: callers go through `loadStatic(_:)` to
    /// hydrate, never mutate directly.
    public private(set) var staticTokens: [Int] = []

    public init(
        modelID: String,
        ngramRange: ClosedRange<Int>,
        maxTokens: Int = 100_000
    ) {
        precondition(modelID.isEmpty == false, "modelID must be non-empty")
        precondition(ngramRange.lowerBound >= 1,
            "ngramRange.lowerBound must be >= 1 (got \(ngramRange.lowerBound))")
        precondition(maxTokens >= 1, "maxTokens must be >= 1 (got \(maxTokens))")
        self.modelID = modelID
        self.ngramRange = ngramRange
        self.maxTokens = maxTokens
    }

    // MARK: - Context tier

    /// Append tokens to the context tier. Called by the iterator as it
    /// commits accepted drafts.
    public func extendContext(_ tokens: [Int]) {
        context.append(contentsOf: tokens)
    }

    /// Reset the context tier — the iterator does this when a new
    /// request starts.
    public func resetContext() {
        context.removeAll(keepingCapacity: true)
    }

    // MARK: - Dynamic tier

    /// Promote `tokens` into the dynamic tier. Called at end of request
    /// (typically with the iterator's `committedTokens` suffix).
    /// Truncates the dynamic tier to `maxTokens` from the head if the
    /// addition exceeds the budget — old history is dropped first.
    public func commitToDynamic(_ tokens: [Int]) {
        dynamic.append(contentsOf: tokens)
        if dynamic.count > maxTokens {
            // FIFO drop: older tokens fall out first.
            let drop = dynamic.count - maxTokens
            dynamic.removeFirst(drop)
        }
    }

    // MARK: - Static tier

    /// Replace the static-tier corpus. Phase 3's loader will call this
    /// on registry construction; for phase 1-2 it's manually invoked
    /// by tests.
    public func loadStatic(_ tokens: [Int]) {
        staticTokens = tokens
    }

    /// Clear all tiers. Mostly a test convenience.
    public func clearAll() {
        context.removeAll(keepingCapacity: true)
        dynamic.removeAll(keepingCapacity: true)
        staticTokens.removeAll(keepingCapacity: true)
    }

    /// Read accessor for a tier — used by the iterator's lookup ladder
    /// in phase 4 to consult tiers in priority order.
    public func tokens(in tier: Tier) -> [Int] {
        switch tier {
        case .context: return context
        case .dynamic: return dynamic
        case .static: return staticTokens
        }
    }

    /// Token count by tier. Useful for telemetry and tests.
    public func count(in tier: Tier) -> Int {
        switch tier {
        case .context: return context.count
        case .dynamic: return dynamic.count
        case .static: return staticTokens.count
        }
    }
}

// MARK: - Registry

/// Process-level singleton mapping `(modelID, ngramRange)` to
/// `NGramCache` instances. Iterator construction looks up the cache
/// for the current model; the dynamic tier survives across iterator
/// instances on the same model.
///
/// Like ``ANEDraftRegistry`` and ``PrefixKVCache``, the shared
/// instance is `@unchecked Sendable` — phase 1-2 is **not**
/// thread-safe; callers using shared serving wrap accesses with their
/// own queue. Phase 4's actor wrapper lands when concurrent serving
/// is needed.
public final class NGramCacheRegistry: @unchecked Sendable {

    /// Process-wide shared registry. Production callers default to
    /// this; tests construct local instances.
    public static let shared = NGramCacheRegistry()

    private struct Key: Hashable {
        let modelID: String
        let lower: Int
        let upper: Int
    }

    private var caches: [Key: NGramCache] = [:]

    public init() {}

    /// Get or create the cache for the given (modelID, ngramRange).
    /// Re-uses an existing cache when its `(modelID, ngramRange)`
    /// matches; constructs a fresh one otherwise.
    public func cache(
        for modelID: String,
        ngramRange: ClosedRange<Int>,
        maxTokens: Int = 100_000
    ) -> NGramCache {
        let key = Key(
            modelID: modelID,
            lower: ngramRange.lowerBound,
            upper: ngramRange.upperBound)
        if let existing = caches[key] {
            return existing
        }
        let cache = NGramCache(
            modelID: modelID,
            ngramRange: ngramRange,
            maxTokens: maxTokens)
        caches[key] = cache
        return cache
    }

    /// Number of registered caches. Useful for tests + diagnostics.
    public var count: Int { caches.count }

    /// Drop a cache by `(modelID, ngramRange)`. Returns true when the
    /// entry existed.
    @discardableResult
    public func remove(modelID: String, ngramRange: ClosedRange<Int>) -> Bool {
        let key = Key(
            modelID: modelID,
            lower: ngramRange.lowerBound,
            upper: ngramRange.upperBound)
        return caches.removeValue(forKey: key) != nil
    }

    /// Clear the entire registry. Mostly for tests at suite teardown.
    public func clear() {
        caches.removeAll()
    }
}
