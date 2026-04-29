// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - ANE draft registry (Mirror Speculative Decoding)
//
// Mirror SD pairs each supported MLX target with a Core ML draft model
// running on the ANE. The pairing is fixed per target — you don't get to
// run Mirror SD with an arbitrary draft because the verify-side correct-
// ness depends on tokenizer alignment (see ``ANEVocabEquivalence``).
// This file defines the registry shape + pure-Swift registration logic.
//
// The registry is intentionally **data-only** — no Core ML imports, no
// model loading. Phase 1B will add a `CoreMLANEDraftBackend` that takes
// a registry entry and constructs the backend; for now, the registry just
// answers "is this target Mirror-SD eligible?" and returns a bundle path
// the future loader will consume.

/// One entry in the ANE draft registry.
public struct ANEDraftRegistryEntry: Equatable, Hashable, Sendable {
    /// Target HuggingFace model ID this draft is registered for. Match is
    /// exact; family-base matching is the caller's job (e.g. "match
    /// `mlx-community/Qwen3.5-27B-Instruct-4bit` and any quantization
    /// variant of it" is a routing decision, not a registry fact).
    public let targetID: String

    /// Local file URL of the Core ML `.mlpackage` (or compiled `.mlmodelc`)
    /// containing the draft model. The registry stores the URL but does
    /// not validate its existence — callers do that when they actually
    /// load. Lets the registry survive partial deployments where some
    /// drafts haven't been downloaded yet.
    public let draftBundleURL: URL

    /// Number of candidate tokens this draft is configured to emit per
    /// cycle. Mirror SD's K is per-draft because the draft's batch shape
    /// is baked into the Core ML graph at conversion time.
    public let draftLength: Int

    /// HuggingFace tokenizer ID expected for *both* draft and target. The
    /// vocabulary-equivalence gate compares the actual target tokenizer
    /// against this expectation at iterator-construction time.
    public let expectedTokenizerID: String

    public init(
        targetID: String,
        draftBundleURL: URL,
        draftLength: Int,
        expectedTokenizerID: String
    ) {
        precondition(draftLength >= 1, "draftLength must be >= 1 (got \(draftLength))")
        self.targetID = targetID
        self.draftBundleURL = draftBundleURL
        self.draftLength = draftLength
        self.expectedTokenizerID = expectedTokenizerID
    }
}

/// Pure-Swift registry mapping target IDs → ANE draft bundle metadata.
///
/// The default registry ships empty in Phase 1A — drafts get added as
/// they're converted (Phase 1B onward). Tests construct ad-hoc registries
/// to exercise lookup behavior; production callers can extend the default
/// instance via ``register(_:)`` at app startup.
public final class ANEDraftRegistry: @unchecked Sendable {
    private var entries: [String: ANEDraftRegistryEntry] = [:]

    /// Process-wide shared registry. Apps populate this at launch with
    /// any drafts they've shipped with the bundle. Tests construct local
    /// registries instead of using this to keep side effects out of the
    /// shared instance.
    public static let shared = ANEDraftRegistry()

    public init() {}

    /// Insert or replace an entry. Replacement is intentional — useful
    /// for hot-swapping in tests and for late-binding draft URLs (e.g.
    /// after on-demand download).
    public func register(_ entry: ANEDraftRegistryEntry) {
        entries[entry.targetID] = entry
    }

    public func remove(targetID: String) {
        entries.removeValue(forKey: targetID)
    }

    public func entry(for targetID: String) -> ANEDraftRegistryEntry? {
        entries[targetID]
    }

    /// Whether the given target has a registered ANE draft. Used as the
    /// `ane-draft-eligible` predicate's first gate during auto-routing
    /// in `MLXLMCommon.generate(...)` (see spec 021 §5).
    public func isRegistered(targetID: String) -> Bool {
        entries[targetID] != nil
    }

    /// All registered target IDs, for diagnostic UIs and bench harness
    /// listings. Order is not stable across calls.
    public func registeredTargetIDs() -> [String] {
        Array(entries.keys)
    }

    /// Number of registered drafts. Mostly a test convenience.
    public var count: Int { entries.count }
}
