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
//
// Phase 1 ships ``IdentityPolicy`` only. ``LastAssistantOpenerPolicy``
// (chat-aware) lands in phase 2 alongside the per-tokenizer template
// scanner.

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
/// `LastAssistantOpenerPolicy` will scan for the boundary tokens directly
/// instead, but this gives bench harnesses a way to reach for a chat-
/// style cache hit without waiting on phase 2.
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
