// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Stable prefix policy (spec 017)
//
// The prefix KV cache (`PrefixKVCache`) stores snapshots of the target's
// KV state at the *end* of a stable prefix — the largest token prefix
// that's expected to recur byte-stably across requests. For
// completion-style workloads that's the entire request prefix; for chat
// workloads the trailing chat-template boilerplate (e.g.
// `<|im_start|>assistant\n`) regenerates every turn and must be excluded
// to maximise hit rate.

/// A policy that decides where the **stable prefix** of a request ends.
/// Returns `n` such that `tokens[0..<n]` is the largest prefix expected
/// to recur unchanged across subsequent requests in the same session.
public protocol StablePrefixPolicy: Sendable {
    /// - Parameter tokens: the request's prompt tokens.
    /// - Returns: number of leading tokens that form the stable prefix
    ///   (`0` if no prefix is stable, `tokens.count` if the entire prompt
    ///   is stable).
    func stablePrefixLen(_ tokens: [Int]) -> Int
}

/// Trivial policy: the entire prompt is treated as stable. Right answer
/// for completion-style workloads where the prompt is request-final
/// (no trailing chat-template boilerplate).
public struct IdentityPolicy: StablePrefixPolicy {
    public init() {}
    public func stablePrefixLen(_ tokens: [Int]) -> Int { tokens.count }
}

/// Trim a fixed number of trailing tokens from the prompt. Useful as a
/// crude phase-1 alternative for chat workloads where the trailing
/// chat-template suffix length is known to be constant (e.g. exactly N
/// tokens for `<|im_start|>assistant\n`). Phase 2's
/// ``LastAssistantOpenerPolicy`` scans for the boundary tokens directly
/// instead, but this gives bench harnesses a way to reach for a chat-
/// style cache hit without depending on tokenizer specifics.
public struct FixedTrimPolicy: StablePrefixPolicy {
    public let trimSuffix: Int

    public init(trimSuffix: Int) {
        precondition(trimSuffix >= 0, "trimSuffix must be >= 0 (got \(trimSuffix))")
        self.trimSuffix = trimSuffix
    }

    public func stablePrefixLen(_ tokens: [Int]) -> Int {
        Swift.max(0, tokens.count - trimSuffix)
    }
}

// MARK: - Phase 2: chat-aware boundary detection

/// Chat-aware stable-prefix policy. Scans the prompt for the **last
/// occurrence** of a model-family-specific "assistant opener" sentinel
/// (e.g. `<|im_start|>assistant\n` for Qwen; `<start_of_turn>model\n`
/// for Gemma 4; `<|start|>assistant<|channel|>` for GPT-OSS) and treats
/// everything before it as stable.
///
/// The opener is supplied as a **token sequence** (pre-encoded at policy
/// construction). The protocol's `stablePrefixLen(_ tokens: [Int])`
/// takes only `[Int]`, so the caller is responsible for encoding the
/// opener via the model's tokenizer before passing it here. This keeps
/// the policy pure-Swift (no `Tokenizer` dependency at runtime) and
/// makes it cheap to share across model loads.
///
/// The match is **exact-sequence**: we look for the rightmost index `i`
/// such that `tokens[i..<i + opener.count] == opener`, then return `i`.
/// If no match is found, the policy falls back to the configured
/// `fallback` length (default `tokens.count` — identity).
public struct LastAssistantOpenerPolicy: StablePrefixPolicy {
    /// The opener token sequence to scan for. Typically the chat
    /// template's "assistant opener" suffix.
    public let opener: [Int]

    /// Length to return when the opener isn't found in the prompt.
    /// Defaults to `tokens.count` (i.e. treat the whole prompt as
    /// stable) so completion-style workloads that don't use the chat
    /// template still hit the cache.
    public enum NoMatchFallback: Sendable, Equatable {
        /// Return `tokens.count` — treat the entire prompt as stable.
        case identity
        /// Return 0 — refuse to cache when no opener is found.
        case refuse
        /// Return `tokens.count - trimSuffix`, floored at 0.
        case fixedTrim(suffix: Int)
    }

    public let fallback: NoMatchFallback

    public init(opener: [Int], fallback: NoMatchFallback = .identity) {
        precondition(!opener.isEmpty, "opener must be non-empty (got [])")
        self.opener = opener
        self.fallback = fallback
    }

    public func stablePrefixLen(_ tokens: [Int]) -> Int {
        guard !tokens.isEmpty, opener.count <= tokens.count else {
            return applyFallback(tokens)
        }
        // Rightmost match: scan from the end of `tokens` backwards.
        // Bounded by `tokens.count - opener.count + 1` start positions.
        let maxStart = tokens.count - opener.count
        var i = maxStart
        while i >= 0 {
            var matched = true
            for j in 0 ..< opener.count where tokens[i + j] != opener[j] {
                matched = false
                break
            }
            if matched { return i }
            i -= 1
        }
        return applyFallback(tokens)
    }

    private func applyFallback(_ tokens: [Int]) -> Int {
        switch fallback {
        case .identity: return tokens.count
        case .refuse: return 0
        case .fixedTrim(let s): return Swift.max(0, tokens.count - s)
        }
    }
}

// MARK: - Pre-encoded model-family openers

/// Pre-encoded chat-template openers for the model families we ship.
/// These are the **token sequences** the model's chat template emits as
/// the "assistant opener" — the trailing region of every chat prompt
/// that regenerates on each turn and therefore should NOT be part of a
/// stable prefix snapshot.
///
/// Each enum value carries the **string** form of the opener; callers
/// resolve the per-tokenizer token sequence via
/// ``encodedTokens(using:)`` at policy construction time. Storing the
/// strings (rather than hard-coded token IDs) keeps the policy robust
/// against vocab differences across model checkpoints.
public enum AssistantOpener: Sendable, Equatable {
    /// Qwen chat template (`<|im_start|>assistant\n`). Covers Qwen 2 /
    /// 2.5 / 3 / 3.5 / 3.6 chat models.
    case qwenChatML
    /// Gemma 4 chat template (`<start_of_turn>model\n`).
    case gemma4
    /// GPT-OSS harmony chat template (`<|start|>assistant<|channel|>`).
    /// The `<|channel|>` token closes the opener; `analysis` /
    /// `commentary` / `final` then follow.
    case gptOSSHarmony
    /// Custom opener supplied verbatim by the caller.
    case custom(String)

    /// String form of the opener. Suitable for `tokenizer.encode(...)`.
    public var rawString: String {
        switch self {
        case .qwenChatML: return "<|im_start|>assistant\n"
        case .gemma4: return "<start_of_turn>model\n"
        case .gptOSSHarmony: return "<|start|>assistant<|channel|>"
        case .custom(let s): return s
        }
    }

    /// Encode this opener via a tokenizer. Special tokens must be
    /// preserved — these are chat-template sentinels, not ordinary
    /// text — so the call uses `addSpecialTokens: false` (the encoder
    /// emits the special-token IDs from the string verbatim without
    /// also prepending a BOS).
    ///
    /// - Parameter tokenizer: any conforming ``Tokenizer``.
    /// - Returns: the token sequence, or `nil` if the encoder produced
    ///   no tokens (likely a tokenizer that doesn't recognise the
    ///   sentinels — caller should fall back to ``FixedTrimPolicy``).
    public func encodedTokens(using tokenizer: any Tokenizer) -> [Int]? {
        let tokens = tokenizer.encode(text: rawString, addSpecialTokens: false)
        return tokens.isEmpty ? nil : tokens
    }

    /// Detect the right opener for a given model based on its
    /// ``ModelContext/configuration``'s `name` field (HF repo ID or
    /// local directory name).
    ///
    /// Returns `nil` for unknown families; callers should fall back to
    /// ``IdentityPolicy``. Catalogue is intentionally conservative:
    /// adding a family is one substring + one test case, but
    /// mis-detecting causes silently sub-optimal cache hits, so we
    /// prefer false-negatives over false-positives.
    ///
    /// Currently catalogued families (case-insensitive substring match):
    ///
    /// | Substring | Opener | Coverage |
    /// |---|---|---|
    /// | `qwen` / `qwq` | ``qwenChatML`` | Qwen 1.x – 3.6, QwQ |
    /// | `gemma` | ``gemma4`` | Gemma 1 / 2 / 3 / 4 — all share `<start_of_turn>model\n` |
    /// | `gpt-oss` / `gpt_oss` | ``gptOSSHarmony`` | GPT-OSS harmony chat template |
    ///
    /// - Parameter modelID: model identifier (typically
    ///   `ModelContext.configuration.name`, e.g.
    ///   `"mlx-community/Qwen3.5-9B-Instruct"`).
    /// - Returns: matching opener, or nil if no family in the catalogue
    ///   matched.
    public static func detect(forModelID modelID: String) -> AssistantOpener? {
        let lower = modelID.lowercased()
        if lower.contains("qwen") || lower.contains("qwq") {
            return .qwenChatML
        }
        if lower.contains("gemma") {
            // Gemma 1/2/3/4 all use the same `<start_of_turn>` opener.
            // The `.gemma4` case name is historical (added when only
            // Gemma 4 was supported) and is kept stable for API
            // continuity.
            return .gemma4
        }
        if lower.contains("gpt-oss") || lower.contains("gpt_oss") {
            return .gptOSSHarmony
        }
        return nil
    }
}

extension LastAssistantOpenerPolicy {
    /// Convenience initialiser that resolves the opener via the
    /// tokenizer. Returns `nil` if the tokenizer can't encode the
    /// opener; callers should then fall back to ``FixedTrimPolicy`` or
    /// ``IdentityPolicy``.
    public init?(
        opener: AssistantOpener,
        tokenizer: any Tokenizer,
        fallback: NoMatchFallback = .identity
    ) {
        guard let tokens = opener.encodedTokens(using: tokenizer) else { return nil }
        self.init(opener: tokens, fallback: fallback)
    }
}
