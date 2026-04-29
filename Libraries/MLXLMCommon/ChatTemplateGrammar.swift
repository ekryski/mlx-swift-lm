// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Chat-template grammar (spec 022 mechanism A — deterministic stretch)
//
// Modern chat templates emit deterministic token sequences at structural
// boundaries: after a channel marker, after a thinking-end tag, at the
// turn opener after the assistant header. By construction these
// continuations don't depend on model output — they're forced by the
// template grammar. A small per-family state machine can predict them
// 100% of the time and feed them through the speculative-decode verify
// path as zero-error drafts.
//
// Phase 1 (this file): the protocol + state types + a `NoOpChatGrammar`
// that always returns nil. Per-family implementations
// (Qwen35Grammar, Gemma4Grammar, GPTOSSGrammar, LlamaGrammar) land in
// follow-up PRs once we wire in the tokenizer-resolved special-token IDs.
//
// The state machine is queried in the speculative iterator's draft
// ladder *before* PLD lookup (deterministic stretches always beat
// probabilistic ones) and *after* the model has just emitted a token —
// the grammar inspects the recent token history + current phase and
// decides whether to inject a deterministic continuation.

/// High-level conversation phase the grammar uses to scope its
/// transition rules. Values are deliberately coarse — fine-grained
/// state lives in the recent-tokens window and the per-family grammar's
/// internal logic.
public enum ChatTemplatePhase: Equatable, Hashable, Sendable {
    case userTurn
    case assistantTurn
    case thinking
    case channelMarker
    case toolCall
    case other
}

/// Tokenizer-resolved chat-template special token IDs. The grammar
/// implementations consume this to convert text-level patterns
/// (`<|channel|>`, `<|im_end|>`, etc.) into token-ID-level rules
/// without re-tokenizing at decode time.
///
/// Phase 1 ships with a small set of fields covering the four
/// supported families. Per-family grammars can ignore fields they
/// don't use; nil-valued fields signal "this template doesn't have
/// that boundary".
public struct ChatTemplateConfig: Equatable, Sendable {
    /// Begin-of-turn / opener token (e.g. `<|im_start|>` for Qwen,
    /// `<start_of_turn>` for Gemma 4).
    public let turnOpenerToken: Int?

    /// End-of-turn / closer token (e.g. `<|im_end|>`, `<end_of_turn>`).
    public let turnCloserToken: Int?

    /// Begin-thinking marker (`<think>` for Qwen 3.5).
    public let thinkBeginToken: Int?

    /// End-thinking marker (`</think>` for Qwen 3.5).
    public let thinkEndToken: Int?

    /// Channel marker (`<|channel|>` for GPT-OSS harmony format).
    public let channelMarkerToken: Int?

    /// Message marker (`<|message|>` for GPT-OSS harmony).
    public let messageMarkerToken: Int?

    /// Newline token IDs — varies per tokenizer (some have a single
    /// `\n`, others have `\n\n` as a separate token). Stored as a set
    /// because matching against any newline-shaped token is fine for
    /// the phase-1 grammars.
    public let newlineTokens: Set<Int>

    public init(
        turnOpenerToken: Int? = nil,
        turnCloserToken: Int? = nil,
        thinkBeginToken: Int? = nil,
        thinkEndToken: Int? = nil,
        channelMarkerToken: Int? = nil,
        messageMarkerToken: Int? = nil,
        newlineTokens: Set<Int> = []
    ) {
        self.turnOpenerToken = turnOpenerToken
        self.turnCloserToken = turnCloserToken
        self.thinkBeginToken = thinkBeginToken
        self.thinkEndToken = thinkEndToken
        self.channelMarkerToken = channelMarkerToken
        self.messageMarkerToken = messageMarkerToken
        self.newlineTokens = newlineTokens
    }
}

/// State the grammar inspects to decide whether to fire a deterministic
/// draft on this round. Iterator constructs it fresh per call; grammars
/// are stateless across calls (any state they need to accumulate is
/// derived from `recentTokens`).
public struct ChatTemplateState: Equatable, Sendable {
    public let phase: ChatTemplatePhase
    public let recentTokens: [Int]
    public let chatTemplateConfig: ChatTemplateConfig

    public init(
        phase: ChatTemplatePhase,
        recentTokens: [Int],
        chatTemplateConfig: ChatTemplateConfig
    ) {
        self.phase = phase
        self.recentTokens = recentTokens
        self.chatTemplateConfig = chatTemplateConfig
    }
}

/// Per-family grammar. Implementations decide whether the just-emitted
/// token has put us at a deterministic-stretch boundary; if so, return
/// the next K forced tokens (typically 1-4); otherwise return nil so
/// the iterator falls through to PLD / bigram / AR.
public protocol ChatTemplateGrammar: Sendable {
    /// - Parameter justEmittedToken: the token that was emitted in the
    ///   last decode step.
    /// - Parameter state: current chat-template state.
    /// - Returns: deterministic continuation tokens, or nil if no
    ///   deterministic stretch applies here.
    func deterministicContinuation(
        afterToken justEmittedToken: Int,
        state: ChatTemplateState
    ) -> [Int]?
}

/// Trivial fallback grammar — always returns nil. Used by the iterator
/// when the target's chat template doesn't match any registered family
/// (and as a default in tests). Drop-in compatible with the protocol so
/// we don't need optional handling in the iterator hot path.
public struct NoOpChatGrammar: ChatTemplateGrammar {
    public init() {}
    public func deterministicContinuation(
        afterToken justEmittedToken: Int,
        state: ChatTemplateState
    ) -> [Int]? {
        nil
    }
}

/// Test / generic grammar that fires a fixed continuation after seeing
/// a specific trigger token. Lets tests exercise the iterator's
/// draft-ladder routing without standing up a real per-family grammar.
public struct TriggerTokenGrammar: ChatTemplateGrammar {
    public let triggerToken: Int
    public let continuation: [Int]

    public init(triggerToken: Int, continuation: [Int]) {
        precondition(!continuation.isEmpty,
            "TriggerTokenGrammar requires non-empty continuation")
        self.triggerToken = triggerToken
        self.continuation = continuation
    }

    public func deterministicContinuation(
        afterToken justEmittedToken: Int,
        state: ChatTemplateState
    ) -> [Int]? {
        justEmittedToken == triggerToken ? continuation : nil
    }
}
