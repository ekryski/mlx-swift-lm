// Copyright © 2026 ekryski.
//
// Multi-token prediction (MTP) — variant B: companion EAGLE-style
// "assistant" drafts. Maps a target HF id to an assistant-draft HF id
// so `MLXLMCommon.generate(...)` can auto-resolve the draft and route
// through the existing `SpeculativeTokenIterator`.
//
// Spec: specs/030-multi-token-prediction.md (variant B).
//
// Initial coverage: Google's gemma-4-*-it-assistant family. These are
// EAGLE-style standalone draft transformers (78M / 78.8M / 0.4B / 0.5B
// params) that share the target's tokenizer and (per the model card)
// produce byte-identical output at temperature 0 with --draft-block-size
// 6 single / 3 batched.

import Foundation

/// Mapping from a target HF id to the recommended assistant draft id +
/// recommended `numDraftTokens`. The match function is a longest-prefix
/// resolver: a target id like
/// `mlx-community/gemma-4-26B-A4B-it-bf16` matches the registry
/// entry for `mlx-community/gemma-4-26B-A4B-it`.
public enum AssistantDraftRegistry {

    /// One mapping rule.
    public struct Entry: Sendable {
        /// Target HF id prefix. Matched against
        /// `ModelConfiguration.name` via `hasPrefix`.
        public let targetPrefix: String
        /// Draft model HF id to load when the target matches.
        public let draftId: String
        /// Recommended `numDraftTokens`. Per the gemma-4 assistant
        /// model card, 6 is best for single-batch decode; 3 for
        /// batched. Mirrors HF Transformers' `--draft-block-size`.
        public let recommendedNumDraftTokens: Int
        /// Optional human-readable note printed in diagnostic logs.
        public let note: String?

        public init(
            targetPrefix: String,
            draftId: String,
            recommendedNumDraftTokens: Int,
            note: String? = nil
        ) {
            self.targetPrefix = targetPrefix
            self.draftId = draftId
            self.recommendedNumDraftTokens = recommendedNumDraftTokens
            self.note = note
        }
    }

    /// Built-in registry. Listed longest-prefix-first so the matcher
    /// picks the most specific entry.
    public static let entries: [Entry] = [
        // Gemma 4 assistants (Google's EAGLE-style drafts; mirrored to
        // mlx-community).
        Entry(
            targetPrefix: "mlx-community/gemma-4-26B-A4B-it",
            draftId: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
            recommendedNumDraftTokens: 6,
            note: "Gemma-4 26B-A4B EAGLE-style assistant draft (~0.4B)"),
        Entry(
            targetPrefix: "mlx-community/gemma-4-31B-it",
            draftId: "mlx-community/gemma-4-31B-it-assistant-bf16",
            recommendedNumDraftTokens: 6,
            note: "Gemma-4 31B EAGLE-style assistant draft (~0.5B)"),
        Entry(
            targetPrefix: "mlx-community/gemma-4-E2B-it",
            draftId: "mlx-community/gemma-4-E2B-it-assistant-bf16",
            recommendedNumDraftTokens: 6,
            note: "Gemma-4 E2B EAGLE-style assistant draft (~78M)"),
        Entry(
            targetPrefix: "mlx-community/gemma-4-E4B-it",
            draftId: "mlx-community/gemma-4-E4B-it-assistant-bf16",
            recommendedNumDraftTokens: 6,
            note: "Gemma-4 E4B EAGLE-style assistant draft (~78.8M)"),
    ]

    /// Resolve an assistant draft for the given target HF id (or
    /// `ModelConfiguration.name`). Returns `nil` if no entry matches.
    ///
    /// Matching:
    ///   - Exact prefix match against `entries[i].targetPrefix`. The
    ///     entries list is searched in declared order; first match
    ///     wins. Order entries longest-prefix-first.
    ///   - Org-strip fallback: if no entry matches the full id, retry
    ///     after stripping a leading `<org>/` prefix. Lets local-only
    ///     bundles (e.g. `LocalDir/gemma-4-26B-A4B-it`) match the
    ///     `mlx-community/gemma-4-26B-A4B-it` rule.
    public static func resolve(targetId: String) -> Entry? {
        for entry in entries {
            if targetId.hasPrefix(entry.targetPrefix) { return entry }
        }
        // Org-strip fallback.
        if let slashIdx = targetId.firstIndex(of: "/") {
            let stripped = String(targetId[targetId.index(after: slashIdx)...])
            for entry in entries {
                let entrySlashIdx = entry.targetPrefix.firstIndex(of: "/")
                let entryStripped: String
                if let entrySlashIdx {
                    entryStripped = String(
                        entry.targetPrefix[
                            entry.targetPrefix.index(after: entrySlashIdx)...])
                } else {
                    entryStripped = entry.targetPrefix
                }
                if stripped.hasPrefix(entryStripped) { return entry }
            }
        }
        return nil
    }
}
