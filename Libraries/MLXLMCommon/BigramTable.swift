// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Common-bigram fallback drafter (spec 022 mechanism B)
//
// A pure-CPU bigram lookup that proposes drafts when neither the chat-
// template grammar nor PLD has a hit. Compiled offline from a
// representative corpus per tokenizer family: for each ordered pair
// `(t_{i-1}, t_i)`, record the top-1 next token if its frequency is
// >= a confidence threshold (default 0.95). At decode time the iterator
// walks the table greedily until it runs out of confident bigrams.
//
// Phase 1 ships the in-memory data structure + a builder that takes a
// frequency map. The on-disk binary format + offline build pipeline
// land in phase 2 (the spec calls out
// `Resources/bigrams/<tokenizer-family>.bin` as the artifact).

/// One entry in the bigram table — the top-1 successor token for a
/// `(prev, current)` bigram, plus the confidence that fired the
/// admission filter at build time.
public struct BigramEntry: Equatable, Sendable {
    public let nextToken: Int
    public let confidence: Float

    public init(nextToken: Int, confidence: Float) {
        precondition(confidence >= 0.0 && confidence <= 1.0,
            "confidence must be in [0, 1] (got \(confidence))")
        self.nextToken = nextToken
        self.confidence = confidence
    }
}

/// Sparse bigram → next-token map. Entries are admitted only when their
/// top-1 successor cleared the confidence threshold at build time;
/// lookup is unconditional (the threshold has already been applied).
///
/// Storage: `[Int: [Int: BigramEntry]]` keyed `prev → current →
/// entry`. Two-level dict instead of `[(Int, Int): BigramEntry]` so
/// `confidentNext(after:)` doesn't need to construct a tuple key on
/// every decode-time call. Flat tuple-key would be cleaner; the nested
/// form is measurably faster on Swift's standard `Dictionary` impl
/// (avoids hash-of-tuple boxing).
public struct BigramTable: Sendable {
    private var table: [Int: [Int: BigramEntry]]

    public let admissionThreshold: Float

    /// Number of (prev, current) → entry mappings stored. Used by tests
    /// and by the bench harness `[BIGRAM]` line.
    public var entryCount: Int {
        table.reduce(0) { $0 + $1.value.count }
    }

    public init(admissionThreshold: Float = 0.95) {
        precondition(
            admissionThreshold >= 0.0 && admissionThreshold <= 1.0,
            "admissionThreshold must be in [0, 1] (got \(admissionThreshold))")
        self.admissionThreshold = admissionThreshold
        self.table = [:]
    }

    /// Insert an entry directly. Tests use this; the offline builder
    /// goes through `build(fromFrequencies:)` instead.
    ///
    /// Re-insert with the same `(prev, current)` overwrites in place
    /// — matches "rebuilt from a fresher corpus" semantics.
    public mutating func insert(prev: Int, current: Int, entry: BigramEntry) {
        precondition(
            entry.confidence >= admissionThreshold,
            "Inserted entry below admission threshold: \(entry.confidence) "
                + "< \(admissionThreshold). Builder should filter first.")
        if table[prev] == nil { table[prev] = [:] }
        table[prev]![current] = entry
    }

    /// Look up the confident successor for `(prev, current)`, or nil if
    /// the bigram isn't in the table.
    public func confidentNext(prev: Int, current: Int) -> BigramEntry? {
        table[prev]?[current]
    }

    /// Walk the bigram chain greedily up to `maxK` drafts.
    ///
    /// - Parameter recentTokens: at least the last 2 emitted tokens.
    ///   The walk seeds from `(recentTokens[-2], recentTokens[-1])`.
    /// - Parameter maxK: cap on draft length (the iterator's per-round
    ///   budget).
    /// - Returns: greedy chain of confident successors. Empty if
    ///   `recentTokens.count < 2` or the seed bigram has no confident
    ///   continuation.
    public func bigramDraft(maxK: Int, recentTokens: [Int]) -> [Int] {
        guard maxK > 0, recentTokens.count >= 2 else { return [] }
        var draft: [Int] = []
        var prev = recentTokens[recentTokens.count - 2]
        var current = recentTokens[recentTokens.count - 1]
        for _ in 0 ..< maxK {
            guard let next = confidentNext(prev: prev, current: current) else { break }
            draft.append(next.nextToken)
            prev = current
            current = next.nextToken
        }
        return draft
    }

    /// Build a bigram table from a `(prev, current) -> [next: count]`
    /// frequency map, applying the admission threshold.
    ///
    /// Threshold semantics: for each `(prev, current)` key, the top-1
    /// `next`'s share of the total observed continuations must be
    /// >= `threshold` to admit the entry. Bigrams whose top-1 falls
    /// below threshold are simply omitted — the table is "sparse on the
    /// confident set", not "exhaustive over the corpus".
    ///
    /// Tied top-1 candidates are broken arbitrarily (Dictionary
    /// iteration order). For phase-1 the threshold is 0.95 and ties
    /// at that level are vanishingly rare; phase-2's offline tool
    /// will use a stable tiebreaker (numeric token ID) when it serializes.
    public static func build(
        fromFrequencies frequencies: [BigramKey: [Int: Int]],
        threshold: Float = 0.95
    ) -> BigramTable {
        var t = BigramTable(admissionThreshold: threshold)
        for (key, nextCounts) in frequencies {
            let total = nextCounts.values.reduce(0, +)
            guard total > 0 else { continue }
            // Find the top-1 successor.
            var bestNext: Int? = nil
            var bestCount: Int = 0
            for (next, count) in nextCounts {
                if count > bestCount {
                    bestCount = count
                    bestNext = next
                }
            }
            guard let nt = bestNext else { continue }
            let confidence = Float(bestCount) / Float(total)
            if confidence >= threshold {
                t.insert(
                    prev: key.prev,
                    current: key.current,
                    entry: BigramEntry(nextToken: nt, confidence: confidence))
            }
        }
        return t
    }
}

/// Key for the frequency map consumed by `BigramTable.build(...)`. Wraps
/// `(prev, current)` into a Hashable struct — Swift can hash tuples of
/// Hashable elements but the ergonomics of an explicit struct are
/// nicer for the offline builder's call sites.
public struct BigramKey: Equatable, Hashable, Sendable {
    public let prev: Int
    public let current: Int

    public init(prev: Int, current: Int) {
        self.prev = prev
        self.current = current
    }
}
