// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Public types

/// Per-step output from a batched generation stream.
///
/// `BatchTokenIterator` advances B sequences in lockstep, so each step yields
/// exactly one new token per sequence. The trailing `.info` event carries
/// aggregate timing for the whole batch.
public enum BatchedGeneration: Sendable {
    /// One decode step. `tokens[i]` is the token sampled for sequence `i`
    /// at this step. `tokens.count == B`.
    case step(tokens: [Int])

    /// Aggregate timing emitted once after the last step.
    case info(BatchedGenerateCompletionInfo)

    /// Per-step tokens, or `nil` for `.info`.
    public var tokens: [Int]? {
        if case .step(let tokens) = self { return tokens }
        return nil
    }

    /// Completion info, or `nil` for `.step`.
    public var info: BatchedGenerateCompletionInfo? {
        if case .info(let info) = self { return info }
        return nil
    }
}

/// Aggregate timing for a batched generation run.
///
/// The bench harness consumes `aggregateTokensPerSecond` (total tokens emitted
/// across the whole batch divided by wall time) as the headline throughput
/// number; per-sequence latency is `tokensPerSecond / batchSize`.
public struct BatchedGenerateCompletionInfo: Sendable {
    /// Number of sequences run concurrently.
    public let batchSize: Int

    /// Prompt token count per sequence (all sequences must have equal-length
    /// prompts in v1, so this is a single scalar).
    public let promptTokenCount: Int

    /// Tokens emitted per sequence (lockstep — `B × generationTokenCountPerSequence`
    /// total tokens were generated).
    public let generationTokenCountPerSequence: Int

    /// Wall time (s) for the prompt pass — the first model forward over the
    /// stacked `[B, L]` input.
    public let promptTime: TimeInterval

    /// Wall time (s) for decode (excludes prompt time).
    public let generateTime: TimeInterval

    /// Total tokens emitted across the batch divided by `generateTime`.
    public var aggregateTokensPerSecond: Double {
        guard generateTime > 0 else { return 0 }
        return Double(batchSize * generationTokenCountPerSequence) / generateTime
    }

    /// Per-sequence decode throughput.
    public var tokensPerSecondPerSequence: Double {
        guard generateTime > 0 else { return 0 }
        return Double(generationTokenCountPerSequence) / generateTime
    }

    /// Prompt-phase throughput (tokens × batchSize / promptTime).
    public var promptTokensPerSecond: Double {
        guard promptTime > 0 else { return 0 }
        return Double(batchSize * promptTokenCount) / promptTime
    }

    public init(
        batchSize: Int,
        promptTokenCount: Int,
        generationTokenCountPerSequence: Int,
        promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.batchSize = batchSize
        self.promptTokenCount = promptTokenCount
        self.generationTokenCountPerSequence = generationTokenCountPerSequence
        self.promptTime = promptTime
        self.generateTime = generateTime
    }
}

// MARK: - Public API

/// Generate tokens for B prompts concurrently via a real B-dim forward pass.
///
/// Stacks B equal-length prompts into a single `[B, L]` tensor and runs the
/// model once per decode step over the batched input. The B sequences advance
/// in lockstep — one new token per sequence per step — so every emission is
/// a `[B]` token vector.
///
/// In v1 prompts must have identical token length. For variable-length serving
/// (per-sequence prompt offsets, per-sequence EOS) see #136 follow-ups.
///
/// - Parameters:
///   - inputs: Prepared B prompts. All must be 1-D and the same length.
///   - cache: Optional pre-built per-layer cache. When `nil`, the model's
///     default cache is created via `model.newCache(parameters:)` — for
///     `RotatingKVCache` (set when `parameters.maxKVSize` is non-nil) the
///     cache allocates a `[B, kvHeads, maxKVSize, headDim]` buffer per layer
///     on first update.
///   - parameters: Generation parameters. Tool-call parsing, perplexity
///     tracking, and per-sequence EOS detection are intentionally not wired
///     in v1.
///   - context: Model context (model + tokenizer + configuration).
///   - wiredMemoryTicket: Optional wired-memory ticket. The bench harness
///     should pass one sized for `batchSize × maxTokens` (see
///     `WiredMemoryUtils.resolveTicket`).
/// - Returns: AsyncStream emitting one `.step(tokens:)` per decode step,
///   followed by a single `.info(...)` event.
public func generateBatched(
    inputs: [LMInput],
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<BatchedGeneration> {
    precondition(!inputs.isEmpty, "generateBatched requires at least one input")

    let batchSize = inputs.count
    let promptTokenCount = inputs[0].text.tokens.dim(0)
    for (i, input) in inputs.enumerated() {
        let tokens = input.text.tokens
        precondition(
            tokens.ndim == 1 && tokens.dim(0) == promptTokenCount,
            "generateBatched v1 requires 1-D prompts of equal length; "
                + "input \(i) has shape \(tokens.shape) vs expected [\(promptTokenCount)]"
        )
    }

    let promptStart = Date()
    let iterator = try BatchTokenIterator(
        inputs: inputs,
        model: context.model,
        cache: cache,
        parameters: parameters
    )
    let promptTime = Date().timeIntervalSince(promptStart)

    let (stream, continuation) = AsyncStream.makeStream(of: BatchedGeneration.self)
    let boxedIterator = SendableBox(iterator)

    let task = Task {
        let runIteration = {
            var iterator = boxedIterator.consume()
            let generateStart = Date()
            var tokensPerSequence = 0
            while !Task.isCancelled, let step = iterator.next() {
                tokensPerSequence += 1
                continuation.yield(.step(tokens: step))
            }

            let generateTime = Date().timeIntervalSince(generateStart)
            Stream().synchronize()

            continuation.yield(
                .info(
                    BatchedGenerateCompletionInfo(
                        batchSize: batchSize,
                        promptTokenCount: promptTokenCount,
                        generationTokenCountPerSequence: tokensPerSequence,
                        promptTime: promptTime,
                        generateTime: generateTime
                    )
                )
            )
            continuation.finish()
        }

        if let wiredMemoryTicket {
            await WiredMemoryTicket.withWiredLimit(wiredMemoryTicket) {
                runIteration()
            }
        } else {
            runIteration()
        }
    }

    continuation.onTermination = { termination in
        if case .cancelled = termination {
            task.cancel()
        }
    }

    return stream
}
