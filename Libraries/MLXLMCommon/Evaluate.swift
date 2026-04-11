// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import os.signpost
import Tokenizers

// MARK: - Generation Stream
// Persistent GPU stream matching Python mlx-lm's `generation_stream = mx.new_stream(...)`.
// Enables prefill pipelining — 3-4x prefill improvement.
private let _generationStreamLock = NSLock()
private var _generationStream: MLX.Stream?
var generationStream: MLX.Stream {
    _generationStreamLock.lock()
    defer { _generationStreamLock.unlock() }
    if let s = _generationStream { return s }
    let s = MLX.Stream(Device.gpu)
    _generationStream = s
    return s
}

/// Stream context matching Python's `with mx.stream(s):`.
/// Sets C-level default stream via setAsDefault, restores Stream.gpu on exit.
func withGenerationStream<R>(_ body: () throws -> R) rethrows -> R {
    generationStream.setAsDefault()
    defer { MLX.Stream.gpu.setAsDefault() }
    return try body()
}

// MARK: - CPU Profiling via os_signpost

/// Signpost log for decode loop CPU profiling.
/// Zero overhead when not recording via Instruments.
/// Capture with: `xcrun xctrace record --template 'Time Profiler'`
private let decodeLog = OSLog(subsystem: "com.mlx.lm", category: "decode")
private let decodeSignpost = OSSignpostID(log: decodeLog)

/// Lightweight wall-clock accumulator for per-step CPU timing.
/// Enabled by MLX_CPU_PROFILE=1 env var. Prints summary after generation.
enum DecodeCPUProfiler {
    nonisolated(unsafe) static var enabled = ProcessInfo.processInfo.environment["MLX_CPU_PROFILE"] == "1"
    nonisolated(unsafe) static var modelForwardNs: UInt64 = 0
    nonisolated(unsafe) static var convertTokenNs: UInt64 = 0
    nonisolated(unsafe) static var asyncEvalNs: UInt64 = 0
    nonisolated(unsafe) static var itemSyncNs: UInt64 = 0
    nonisolated(unsafe) static var otherNs: UInt64 = 0
    nonisolated(unsafe) static var tokenCount: Int = 0

    // Sub-step breakdown within convertToToken
    nonisolated(unsafe) static var logitSliceNs: UInt64 = 0
    nonisolated(unsafe) static var processorNs: UInt64 = 0
    nonisolated(unsafe) static var samplerNs: UInt64 = 0
    nonisolated(unsafe) static var perplexityNs: UInt64 = 0
    nonisolated(unsafe) static var didSampleNs: UInt64 = 0

    static func reset() {
        modelForwardNs = 0; convertTokenNs = 0; asyncEvalNs = 0
        itemSyncNs = 0; otherNs = 0; tokenCount = 0
        logitSliceNs = 0; processorNs = 0; samplerNs = 0
        perplexityNs = 0; didSampleNs = 0
    }

    static func report() {
        guard enabled, tokenCount > 0 else { return }
        let total = modelForwardNs + convertTokenNs + asyncEvalNs + itemSyncNs + otherNs
        let perToken = Double(total) / Double(tokenCount) / 1_000_000  // ms

        func pct(_ v: UInt64) -> String { String(format: "%.0f", Double(v) / Double(total) * 100) }
        func ms(_ v: UInt64) -> String { String(format: "%.2f", Double(v) / Double(tokenCount) / 1_000_000) }

        print("\n[CPU-PROFILE] Decode loop breakdown (\(tokenCount) tokens, \(String(format: "%.1f", perToken))ms/token):")
        print("[CPU-PROFILE]   model forward:   \(ms(modelForwardNs))ms (\(pct(modelForwardNs))%)")
        print("[CPU-PROFILE]   convertToToken:  \(ms(convertTokenNs))ms (\(pct(convertTokenNs))%)")
        print("[CPU-PROFILE]     logit slice:   \(ms(logitSliceNs))ms")
        print("[CPU-PROFILE]     processor:     \(ms(processorNs))ms")
        print("[CPU-PROFILE]     sampler:       \(ms(samplerNs))ms")
        print("[CPU-PROFILE]     perplexity:    \(ms(perplexityNs))ms")
        print("[CPU-PROFILE]     didSample:     \(ms(didSampleNs))ms")
        print("[CPU-PROFILE]   asyncEval:       \(ms(asyncEvalNs))ms (\(pct(asyncEvalNs))%)")
        print("[CPU-PROFILE]   .item() sync:    \(ms(itemSyncNs))ms (\(pct(itemSyncNs))%)")
        print("[CPU-PROFILE]   other:           \(ms(otherNs))ms (\(pct(otherNs))%)")
        print("")
    }
}

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
public protocol LogitSampler {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let y = sampler.sample(logits: logits)
/// processor?.didSample(token: y)
/// ```
///
/// See also: ``LogitSampler``
public protocol LogitProcessor {

    /// called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.
public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize: Int

    /// Maximum tokens to generate
    public var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries (except the first 4 tokens) will be overwritten.
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple`` (which is unbounded)
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// KV cache compression scheme. nil = use kvBits (affine quantization) if set.
    /// "turbo1" through "turbo4" = TurboQuant compression at 1-4 bits.
    /// When set, kvBits is ignored for cache creation.
    public var kvScheme: String?

    /// sampling temperature
    public var temperature: Float

    /// top p sampling
    public var topP: Float

    /// top k sampling (0 disables)
    public var topK: Int

    /// min p sampling threshold relative to the highest probability token (0 disables)
    public var minP: Float

    /// penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// additive penalty for tokens that appear in recent context
    public var presencePenalty: Float?

    /// number of tokens to consider for presence penalty
    public var presenceContextSize: Int

    /// additive penalty that scales with token frequency in recent context
    public var frequencyPenalty: Float?

    /// number of tokens to consider for frequency penalty
    public var frequencyContextSize: Int

    /// additional logit processors (e.g., ThinkingEOSSuppressionProcessor)
    public var additionalProcessors: [LogitProcessor]

    /// reasoning effort hint (e.g., "low", "medium", "high")
    public var reasoningEffort: String?

    /// N-gram size for prompt-lookup speculative decoding. When > 0, the iterator
    /// searches for matching n-grams in the prompt text and uses continuations as
    /// draft tokens, verifying them in a single batched forward pass.
    /// Typical value: 3 (trigram matching). Set to 0 to disable.
    public var ngramSize: Int

    /// Maximum draft tokens per n-gram speculation round.
    public var maxNgramDraftTokens: Int

    /// Token ID marking the start of a thinking phase (e.g., <think> token).
    /// When set with thinkEndTokenId, log probabilities are tracked separately
    /// for thinking vs. generation phases in GenerateCompletionInfo.
    public var thinkStartTokenId: Int32?

    /// Token ID marking the end of a thinking phase (e.g., </think> token).
    public var thinkEndTokenId: Int32?

    /// When true, the iterator starts already inside the thinking phase.
    /// Use this when <think> was prepended as an assistant prefix in the prompt
    /// rather than being generated — so the iterator never sees the start token.
    public var thinkingPhasePrefilled: Bool

    /// When true, per-token log probs, token IDs, and phase labels are stored
    /// in the TokenIterator for downstream KLD computation.
    public var collectPerTokenData: Bool

    /// When true, accumulate log probabilities for perplexity computation.
    /// Default: true (backward compatible). Set to false for production inference
    /// where perplexity isn't needed — saves a full-vocab softmax+log per token
    /// and prevents the lazy compute graph from retaining logits buffers.
    public var trackPerplexity: Bool

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefillStepSize: Int = 512,
        additionalProcessors: [LogitProcessor] = [],
        reasoningEffort: String? = nil,
        ngramSize: Int = 0,
        maxNgramDraftTokens: Int = 5,
        kvScheme: String? = nil,
        thinkStartTokenId: Int32? = nil,
        thinkEndTokenId: Int32? = nil,
        thinkingPhasePrefilled: Bool = false,
        collectPerTokenData: Bool = false,
        trackPerplexity: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.kvScheme = kvScheme
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefillStepSize = prefillStepSize
        self.additionalProcessors = additionalProcessors
        self.reasoningEffort = reasoningEffort
        self.ngramSize = ngramSize
        self.maxNgramDraftTokens = maxNgramDraftTokens
        self.kvScheme = kvScheme
        self.thinkStartTokenId = thinkStartTokenId
        self.thinkEndTokenId = thinkEndTokenId
        self.thinkingPhasePrefilled = thinkingPhasePrefilled
        self.collectPerTokenData = collectPerTokenData
        self.trackPerplexity = trackPerplexity
    }

    public func sampler() -> LogitSampler {
        let usesTopP = topP > 0 && topP < 1
        let usesTopK = topK > 0
        let usesMinP = minP > 0

        if temperature == 0 {
            return ArgMaxSampler()
        } else if usesTopP || usesTopK || usesMinP {
            return TopPSampler(temperature: temperature, topP: topP, topK: topK, minP: minP)
        } else {
            return CategoricalSampler(temperature: temperature)
        }
    }

    public func processor() -> LogitProcessor? {
        var all: [LogitProcessor] = []

        if let repetitionPenalty, repetitionPenalty != 0, repetitionContextSize > 0 {
            all.append(
                RepetitionContext(
                    repetitionPenalty: repetitionPenalty,
                    repetitionContextSize: repetitionContextSize))
        }

        if let presencePenalty, presencePenalty != 0, presenceContextSize > 0 {
            all.append(
                PresencePenaltyContext(
                    presencePenalty: presencePenalty,
                    presenceContextSize: presenceContextSize))
        }

        if let frequencyPenalty, frequencyPenalty != 0, frequencyContextSize > 0 {
            all.append(
                FrequencyPenaltyContext(
                    frequencyPenalty: frequencyPenalty,
                    frequencyContextSize: frequencyContextSize))
        }

        all.append(contentsOf: additionalProcessors)

        switch all.count {
        case 0: return nil
        case 1: return all[0]
        default: return CompositeLogitProcessor(processors: all)
        }
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    public init() {}

    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
/// to sample the logits.  Works in log-space to avoid numerical issues.
/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
/// to sample the logits.
///
/// Filters are applied in the same order as Python mlx-lm: top_p → min_p → top_k.
/// Each filter operates on the full vocabulary in original token order, masking
/// rejected tokens with `-inf`. This matches the composable filter chain in
/// `mlx_lm.sample_utils.make_sampler`.
///
/// Performance optimizations (matching PR #147):
/// - Log-space throughout (`logSoftmax`) — avoids redundant softmax calls
/// - `argPartition` for top-k — O(V) partial sort vs O(V log V) full sort
/// - `-Float.infinity` for proper masking in categorical sampler
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0) {
        self.temp = MLXArray(temperature)
        if topP > 0 && topP < 1 {
            self.topP = MLXArray(topP)
        } else {
            self.topP = nil
        }
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.negInf = MLXArray(-Float.infinity)
        self.randomState = MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)

            if let topP {
                logprobs = applyTopP(logprobs, topP: topP)
            }
            if let minP {
                logprobs = applyMinP(logprobs, minP: minP)
            }
            if let topK {
                logprobs = applyTopK(logprobs, topK: topK)
            }

            return categorical(logprobs * (1 / temp))
        }
    }

    /// Keep tokens whose cumulative probability exceeds `1 - topP` (nucleus sampling).
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP (log-space).
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens (O(V) argPartition).
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// Sampler that uses `temperature` to sample the logits.
public struct CategoricalSampler: LogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float) {
        self.temp = MLXArray(temperature)
        self.randomState = MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// GPU-resident ring buffer of recent token IDs.
/// Uses MLX.where mask operations for GPU-only updates (no CPU<-GPU sync),
/// preserving asyncEval() pipelining in TokenIterator.
struct TokenRing {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    mutating func loadPrompt(_ prompt: MLXArray) {
        let promptTokens = prompt.reshaped(-1).asType(.int32)
        let n = promptTokens.dim(0)
        if n <= capacity {
            if n < capacity {
                let padding = MLXArray.zeros([capacity - n], type: Int32.self)
                buffer = concatenated([promptTokens.reshaped(-1), padding])
            } else {
                buffer = promptTokens.reshaped(-1)
            }
            count = n
            writeIndex = n % capacity
        } else {
            buffer = promptTokens[(-capacity)...].reshaped(-1)
            count = capacity
            writeIndex = 0
        }
    }

    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Processor that implements a `repetitionPenalty`
public struct RepetitionContext: LogitProcessor {
    var ring: TokenRing

    /// penalty factor for repeating tokens
    let repetitionPenalty: Float

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        precondition(repetitionContextSize > 0)
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }
        let indices = validTokens.asType(.uint32)
        var selectedLogits = logits[0..., indices]

        selectedLogits = MLX.where(
            selectedLogits .< 0, selectedLogits * repetitionPenalty,
            selectedLogits / repetitionPenalty)

        logits[0..., indices] = selectedLogits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive presence penalty to tokens in a recent context window.
public struct PresencePenaltyContext: LogitProcessor {
    var ring: TokenRing

    let presencePenalty: Float

    public init(presencePenalty: Float, presenceContextSize: Int) {
        precondition(presenceContextSize > 0)
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }
        let indices = validTokens.asType(.uint32)
        logits[0..., indices] = logits[0..., indices] - presencePenalty
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
public struct FrequencyPenaltyContext: LogitProcessor {
    var ring: TokenRing

    let frequencyPenalty: Float

    public init(frequencyPenalty: Float, frequencyContextSize: Int) {
        precondition(frequencyContextSize > 0)
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }
        let vocabSize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
            .at[validTokens.asType(.int32)].add(ones)
        return logits - histogram * frequencyPenalty
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Composes multiple logit processors into a single processor.
public struct CompositeLogitProcessor: LogitProcessor {
    var processors: [LogitProcessor]

    public init(processors: [LogitProcessor]) {
        self.processors = processors
    }

    mutating public func prompt(_ prompt: MLXArray) {
        for i in 0 ..< processors.count {
            processors[i].prompt(prompt)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        for processor in processors {
            logits = processor.process(logits: logits)
        }
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        for i in 0 ..< processors.count {
            processors[i].didSample(token: token)
        }
    }
}

/// Suppresses EOS tokens for a window of tokens after a trigger sequence is generated.
///
/// This is designed for thinking models (e.g., Qwen3.5, Nemotron) where the model generates
/// `</think>` and then immediately emits `<|im_end|>` instead of continuing with
/// `<tool_call>` or response text. By suppressing EOS for a configurable number of tokens
/// after the trigger, the model is forced to generate content.
///
/// Supports two trigger modes:
/// - **Single token**: When `</think>` is a single special token in the vocabulary
/// - **Token sequence**: When `</think>` is tokenized as multiple tokens (e.g., `<`, `/think`, `>`)
///
/// Example usage (single token):
/// ```swift
/// let processor = ThinkingEOSSuppressionProcessor(
///     eosTokenIds: Set([248044, 248046]),
///     triggerTokenId: 248069,  // </think> as single special token
///     suppressionWindow: 15
/// )
/// ```
///
/// Example usage (token sequence):
/// ```swift
/// let thinkCloseTokens = tokenizer.encode(text: "</think>")
/// let processor = ThinkingEOSSuppressionProcessor(
///     eosTokenIds: Set([151643, 151645]),
///     triggerSequence: thinkCloseTokens,
///     suppressionWindow: 15
/// )
/// ```
public struct ThinkingEOSSuppressionProcessor: LogitProcessor {
    /// Token IDs to suppress (set logits to -inf) during the suppression window.
    let eosTokenIds: Set<Int>

    /// Single token ID trigger (used when `</think>` is one special token).
    let triggerTokenId: Int?

    /// Multi-token sequence trigger (used when `</think>` tokenizes to multiple tokens).
    /// The processor watches a rolling buffer and triggers when the last N tokens match.
    let triggerSequence: [Int]?

    /// Number of tokens after the trigger to suppress EOS for.
    /// 15 tokens covers `\n\n<tool_call>\n<function=execute_skill_action>\n`.
    let suppressionWindow: Int

    /// Tracks tokens generated since the trigger. nil = not triggered yet.
    var tokensSinceTrigger: Int?

    /// Rolling buffer of recent token IDs for sequence matching.
    var recentTokens: [Int]

    /// Single-token trigger initializer.
    public init(eosTokenIds: Set<Int>, triggerTokenId: Int, suppressionWindow: Int = 15) {
        self.eosTokenIds = eosTokenIds
        self.triggerTokenId = triggerTokenId
        self.triggerSequence = nil
        self.suppressionWindow = suppressionWindow
        self.tokensSinceTrigger = nil
        self.recentTokens = []
    }

    /// Multi-token sequence trigger initializer.
    /// - Parameter triggerSequence: The token IDs that form the trigger (e.g., tokenizer.encode("</think>")).
    ///   Must not be empty.
    public init(eosTokenIds: Set<Int>, triggerSequence: [Int], suppressionWindow: Int = 15) {
        precondition(!triggerSequence.isEmpty, "triggerSequence must not be empty")
        self.eosTokenIds = eosTokenIds
        self.triggerTokenId = nil
        self.triggerSequence = triggerSequence
        self.suppressionWindow = suppressionWindow
        self.tokensSinceTrigger = nil
        self.recentTokens = []
    }

    mutating public func prompt(_ prompt: MLXArray) {
        tokensSinceTrigger = nil
        recentTokens = []
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let count = tokensSinceTrigger, count < suppressionWindow else {
            return logits
        }
        // logits can be [V] (1D) or [1, V] (2D with batch). Index vocabulary dimension correctly.
        var logits = logits
        if logits.ndim == 1 {
            for eosId in eosTokenIds {
                logits[eosId] = MLXArray(-Float.infinity)
            }
        } else {
            for eosId in eosTokenIds {
                logits[0..., eosId] = MLXArray(-Float.infinity)
            }
        }
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        let tokenId = token.item(Int.self)

        // Single-token trigger mode
        if let triggerId = triggerTokenId {
            if tokenId == triggerId {
                tokensSinceTrigger = 0
            } else if tokensSinceTrigger != nil {
                tokensSinceTrigger! += 1
            }
            return
        }

        // Multi-token sequence trigger mode
        if let sequence = triggerSequence {
            recentTokens.append(tokenId)
            // Keep only as many tokens as the sequence length
            if recentTokens.count > sequence.count {
                recentTokens.removeFirst(recentTokens.count - sequence.count)
            }

            if tokensSinceTrigger == nil {
                // Check if the rolling buffer matches the trigger sequence
                if recentTokens == sequence {
                    tokensSinceTrigger = 0
                }
            } else {
                tokensSinceTrigger! += 1
            }
        }
    }
}

/// Suppresses EOS tokens for a window of tokens after an external trigger.
///
/// Unlike `ThinkingEOSSuppressionProcessor` which detects triggers by token ID matching,
/// this processor is triggered externally by setting `triggered = true`. This is designed
/// for models where `</think>` is tokenized as multiple regular tokens (not a single special
/// token), making token-level detection unreliable.
///
/// The calling code detects `</think>` by text matching (e.g., via a BudgetClassifier)
/// and sets the trigger. The processor then suppresses EOS for the next `suppressionWindow`
/// tokens, forcing the model to generate content (tool calls or response text).
///
/// This is a reference type (class) so the trigger can be set from the generation loop
/// while the processor runs inside the generate() function.
///
/// Example usage:
/// ```swift
/// let suppressor = EOSSuppressionTrigger(
///     eosTokenIds: Set([151643, 151645]),
///     suppressionWindow: 15
/// )
/// parameters.additionalProcessors.append(suppressor)
///
/// // In generation loop:
/// if thinkingJustEnded {
///     suppressor.triggered = true
/// }
/// ```
public final class EOSSuppressionTrigger: @unchecked Sendable, LogitProcessor {
    /// Token IDs to suppress (set logits to -inf) during the suppression window.
    let eosTokenIds: Set<Int>

    /// Number of tokens after trigger to suppress EOS for.
    public let suppressionWindow: Int

    /// Atomic trigger flag — set from the generation loop, read by the logit processor.
    /// Uses os_unfair_lock for thread safety between producer (generate) and consumer (loop).
    private var _triggered: Bool = false
    private let lock = NSLock()

    public var triggered: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _triggered }
        set { lock.lock(); _triggered = newValue; lock.unlock() }
    }

    /// Tracks tokens generated since trigger.
    private var _tokensSinceTrigger: Int = 0

    public init(eosTokenIds: Set<Int>, suppressionWindow: Int = 15) {
        self.eosTokenIds = eosTokenIds
        self.suppressionWindow = suppressionWindow
    }

    public func prompt(_ prompt: MLXArray) {
        lock.lock()
        _triggered = false
        _tokensSinceTrigger = 0
        lock.unlock()
    }

    public func process(logits: MLXArray) -> MLXArray {
        lock.lock()
        let isTriggered = _triggered
        let count = _tokensSinceTrigger
        lock.unlock()

        guard isTriggered, count < suppressionWindow else {
            return logits
        }
        print("[EOSSuppressor] Suppressing EOS tokens \(eosTokenIds) at count=\(count)/\(suppressionWindow), shape=\(logits.shape)")
        var logits = logits
        // logits can be [V] (1D) or [1, V] (2D with batch). Index vocabulary dimension correctly.
        if logits.ndim == 1 {
            for eosId in eosTokenIds {
                logits[eosId] = MLXArray(-Float.infinity)
            }
        } else {
            for eosId in eosTokenIds {
                logits[0..., eosId] = MLXArray(-Float.infinity)
            }
        }
        return logits
    }

    public func didSample(token: MLXArray) {
        let tokenId = token.item(Int.self)
        lock.lock()
        if _triggered {
            _tokensSinceTrigger += 1
            let count = _tokensSinceTrigger
            lock.unlock()
            print("[EOSSuppressor] didSample token=\(tokenId), count=\(count)/\(suppressionWindow)")
        } else {
            lock.unlock()
        }
    }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: generateParameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let kvScheme: String?

    // N-gram prompt lookup speculation
    let ngramSize: Int
    let maxNgramDraftTokens: Int
    var promptTokenIds: [Int32] = []
    var generatedTokenIds: [Int32] = []
    var pendingTokens: [Int] = []
    var ngramProposed = 0
    var ngramAccepted = 0
    var cachesTrimmable = false
    var ngramDisabled = false  // auto-disabled when acceptance rate drops
    var ngramAttempts = 0      // rolling window for acceptance tracking
    var ngramHits = 0          // hits in rolling window

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

    // Log probability accumulation for perplexity
    var logProbSum: MLXArray = MLXArray(Float(0))
    var logProbTokenCount: Int = 0

    // Phase-aware perplexity: thinking vs. generation
    var thinkStartTokenId: Int32?
    var thinkEndTokenId: Int32?
    var inThinkingPhase: Bool = false  // set to true at init when prefilled
    var thinkingLogProbSum: MLXArray = MLXArray(Float(0))
    var thinkingLogProbCount: Int = 0
    var generationLogProbSum: MLXArray = MLXArray(Float(0))
    var generationLogProbCount: Int = 0

    /// When true, per-token log probs, token IDs, and phase labels are stored
    /// for downstream KLD computation. Off by default to avoid memory overhead.
    var collectPerTokenData: Bool = false

    /// When true, accumulate log probabilities for perplexity computation.
    /// When false, skip the full-vocab softmax+log chain — saves GPU compute and
    /// prevents the lazy graph from retaining logits buffers across all tokens.
    var trackPerplexity: Bool = false
    var perTokenLogProbs: [Float] = []
    var perTokenIds: [Int] = []
    /// Phase label per token: "think", "gen", or "marker"
    var perTokenPhases: [String] = []

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:parameters:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.kvScheme = parameters.kvScheme

        self.ngramSize = parameters.ngramSize
        self.maxNgramDraftTokens = parameters.maxNgramDraftTokens
        if ngramSize > 0 {
            self.promptTokenIds = prompt.reshaped(-1).asArray(Int32.self)
        }

        self.thinkStartTokenId = parameters.thinkStartTokenId
        self.thinkEndTokenId = parameters.thinkEndTokenId
        self.inThinkingPhase = parameters.thinkingPhasePrefilled
        self.collectPerTokenData = parameters.collectPerTokenData
        self.trackPerplexity = parameters.trackPerplexity

        self.promptPrefillTime = try measure {
            try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
        }

        self.cachesTrimmable = self.cache.allSatisfy { $0.isTrimmable }
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.kvScheme = parameters.kvScheme

        self.ngramSize = parameters.ngramSize
        self.maxNgramDraftTokens = parameters.maxNgramDraftTokens
        if ngramSize > 0 {
            self.promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int32.self)
        }

        self.thinkStartTokenId = parameters.thinkStartTokenId
        self.thinkEndTokenId = parameters.thinkEndTokenId
        self.inThinkingPhase = parameters.thinkingPhasePrefilled
        self.collectPerTokenData = parameters.collectPerTokenData
        self.trackPerplexity = parameters.trackPerplexity

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }

        self.cachesTrimmable = self.cache.allSatisfy { $0.isTrimmable }
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefillStepSize: optional prefill step size
    ///   - maxTokens: maximum number of tokens to generate
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int = 512,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = 64
        self.quantizedKVStart = 0
        self.kvScheme = nil

        // No n-gram speculation for direct initialization
        self.ngramSize = 0
        self.maxNgramDraftTokens = 0

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: prefillStepSize)
        }

        self.cachesTrimmable = self.cache.allSatisfy { $0.isTrimmable }
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        try withGenerationStream {
            switch try model.prepare(input, cache: cache, windowSize: windowSize) {
            case .tokens(let tokens):
                y = tokens

                // Prime the pump with the last prompt token
                let token = step(previous: y)
                y = .init(tokens: token)

                // Sync-eval first decode token — asyncEval causes pad-token bug on Gemma 4
                eval(y.tokens)

            case .logits(let result):
                y = .init(tokens: convertToToken(logits: result.logits))
                asyncEval(y.tokens)
                MLX.Memory.clearCache()

                break
            }
        }
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var ct0 = DispatchTime.now().uptimeNanoseconds

        // process the logits (one hot array of possible tokens)
        var logits = logits[0..., -1, 0...]

        if DecodeCPUProfiler.enabled {
            let ct1 = DispatchTime.now().uptimeNanoseconds
            DecodeCPUProfiler.logitSliceNs += (ct1 - ct0)
            ct0 = ct1
        }

        logits = processor?.process(logits: logits) ?? logits

        if DecodeCPUProfiler.enabled {
            let ct1 = DispatchTime.now().uptimeNanoseconds
            DecodeCPUProfiler.processorNs += (ct1 - ct0)
            ct0 = ct1
        }

        // transform logits back to a token
        let y = sampler.sample(logits: logits)

        if DecodeCPUProfiler.enabled {
            let ct1 = DispatchTime.now().uptimeNanoseconds
            DecodeCPUProfiler.samplerNs += (ct1 - ct0)
            ct0 = ct1
        }

        // Accumulate log probability for perplexity computation.
        // PERF: When trackPerplexity is false, skip the full-vocab softmax+log chain
        // entirely. This saves GPU compute and prevents the lazy graph from retaining
        // logits buffers (~1MB per token for vocab=248K) across the entire generation.
        if trackPerplexity || collectPerTokenData {
            let logprobs = log(softmax(logits.asType(.float32)))
            let tokenLogProb = takeAlong(
                logprobs.reshaped([1, -1]),
                y.reshaped([1, 1]).asType(.int32),
                axis: 1
            ).reshaped([])
            logProbSum = logProbSum + tokenLogProb
            logProbTokenCount += 1

            // Phase-aware tracking: separate thinking vs. generation perplexity.
            //
            // PERF: .item() forces a GPU→CPU sync (~4ms) that breaks the async pipeline.
            // This sync is ONLY needed for per-phase perplexity reporting (separating
            // thinking PPL from generation PPL). It is NOT required for model correctness
            // — the model generates correctly without phase tracking.
            //
            // Gate behind collectPerTokenData (KLD computation needs it) OR
            // trackPerplexity when thinking tokens are configured. In production
            // inference (trackPerplexity=false), this sync is completely skipped
            // even for thinking models like Qwen3.5.
            if collectPerTokenData
                || (trackPerplexity && (thinkStartTokenId != nil || thinkEndTokenId != nil))
            {
                let tokenId = y.item(Int32.self)
                let phase: String
                if let startId = thinkStartTokenId, tokenId == startId {
                    inThinkingPhase = true
                    phase = "marker"
                } else if let endId = thinkEndTokenId, tokenId == endId {
                    inThinkingPhase = false
                    phase = "marker"
                } else if inThinkingPhase {
                    thinkingLogProbSum = thinkingLogProbSum + tokenLogProb
                    thinkingLogProbCount += 1
                    phase = "think"
                } else {
                    generationLogProbSum = generationLogProbSum + tokenLogProb
                    generationLogProbCount += 1
                    phase = "gen"
                }

                if collectPerTokenData {
                    perTokenLogProbs.append(tokenLogProb.item(Float.self))
                    perTokenIds.append(Int(tokenId))
                    perTokenPhases.append(phase)
                }
            } else {
                generationLogProbSum = generationLogProbSum + tokenLogProb
                generationLogProbCount += 1
            }
        }

        if DecodeCPUProfiler.enabled {
            let ct1 = DispatchTime.now().uptimeNanoseconds
            DecodeCPUProfiler.perplexityNs += (ct1 - ct0)
            ct0 = ct1
        }

        // PERF: didSample moved to next() AFTER asyncEval to avoid triggering
        // GPU eval prematurely. The penalty processor's ring.append uses MLX.where
        // which forces eval — by deferring it after asyncEval, the GPU starts
        // the forward pass evaluation asynchronously first.
        // processor?.didSample(token: y)  // MOVED to next()

        if DecodeCPUProfiler.enabled {
            DecodeCPUProfiler.didSampleNs += (DispatchTime.now().uptimeNanoseconds - ct0)
        }

        return y
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        var t0 = DispatchTime.now().uptimeNanoseconds

        os_signpost(.begin, log: decodeLog, name: "model_forward", signpostID: decodeSignpost)
        let result = withGenerationStream {
            model(previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        }
        self.state = result.state
        os_signpost(.end, log: decodeLog, name: "model_forward", signpostID: decodeSignpost)

        if DecodeCPUProfiler.enabled {
            let t1 = DispatchTime.now().uptimeNanoseconds
            DecodeCPUProfiler.modelForwardNs += (t1 - t0)
            t0 = t1
        }

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart,
            kvScheme: kvScheme
        )

        os_signpost(.begin, log: decodeLog, name: "convert_token", signpostID: decodeSignpost)
        let token = convertToToken(logits: result.logits)
        os_signpost(.end, log: decodeLog, name: "convert_token", signpostID: decodeSignpost)

        if DecodeCPUProfiler.enabled {
            DecodeCPUProfiler.convertTokenNs += (DispatchTime.now().uptimeNanoseconds - t0)
        }

        return token
    }

    // MARK: - N-gram Prompt Lookup Speculation

    /// Search prompt + generated tokens for a matching n-gram continuation.
    private func lookupNgramDraft() -> [Int32] {
        let allTokens = promptTokenIds + generatedTokenIds
        guard allTokens.count >= ngramSize else { return [] }

        // Query: last ngramSize tokens
        let queryStart = allTokens.count - ngramSize
        let query = Array(allTokens[queryStart...])

        // Search for matching n-gram (skip the query itself at the end)
        let searchEnd = allTokens.count - ngramSize
        guard searchEnd > 0 else { return [] }

        // Search backwards — more recent matches are more likely to be relevant
        var i = searchEnd - 1
        while i >= 0 {
            if i + ngramSize > searchEnd { i -= 1; continue }
            var matches = true
            for j in 0..<ngramSize {
                if allTokens[i + j] != query[j] { matches = false; break }
            }
            if matches {
                let contStart = i + ngramSize
                let contEnd = Swift.min(contStart + maxNgramDraftTokens, allTokens.count)
                if contStart < contEnd {
                    return Array(allTokens[contStart..<contEnd])
                }
            }
            i -= 1
        }
        return []
    }

    /// Run one n-gram speculation round with adaptive enable/disable.
    ///
    /// Skips speculation if:
    /// - Not enough generated tokens yet (need ngramSize to form a query)
    /// - Acceptance rate dropped below 30% (auto-disabled)
    /// - Caches aren't trimmable
    ///
    /// Returns accepted token IDs (includes the corrected token on mismatch).
    private mutating func ngramSpeculateRound() -> [Int] {
        guard cachesTrimmable else { return [] }
        guard !ngramDisabled else { return [] }

        // Need at least ngramSize generated tokens to form a lookup query
        guard generatedTokenIds.count >= ngramSize else { return [] }

        let draftTokens = lookupNgramDraft()
        guard !draftTokens.isEmpty else { return [] }

        let k = draftTokens.count
        ngramProposed += k

        // Try batched verification (K+1 tokens in one forward pass)
        let result = batchedNgramVerification(draftTokens: draftTokens, k: k)

        // Adaptive disable: track acceptance and disable if too low
        ngramAttempts += 1
        if !result.isEmpty {
            // At least one token was accepted (the corrected token counts)
            // Real "hits" are when draft tokens matched (result.count - 1 on mismatch, result.count on full match)
            let matchCount = Swift.min(result.count, k)  // cap at k (excludes bonus token)
            let actualMatches = result.count > k ? k : Swift.max(0, result.count - 1)
            ngramHits += actualMatches
        }

        // Check rolling acceptance rate every 10 attempts
        if ngramAttempts >= 10 {
            let rate = Double(ngramHits) / Double(ngramAttempts * maxNgramDraftTokens)
            if rate < 0.1 {  // less than 10% of proposed tokens accepted
                ngramDisabled = true
            }
            // Reset rolling window
            ngramAttempts = 0
            ngramHits = 0
        }

        return result
    }

    /// Batched verification: feed [currentToken, draft_0, ..., draft_{k-1}]
    /// in a single forward pass and compare each position's output.
    private mutating func batchedNgramVerification(draftTokens: [Int32], k: Int) -> [Int] {
        // Build verification input: [currentToken, draft_0, ..., draft_{k-1}]
        // y.tokens is the current token (scalar or [1])
        let currentToken = y.tokens.item(Int32.self)
        var allTokenIds = [currentToken] + draftTokens
        let verifyTokens = MLXArray(allTokenIds)  // [k+1]

        // Create LMInput.Text with batch dimension
        let verifyText = LMInput.Text(tokens: verifyTokens[.newAxis])  // [1, k+1]

        // Run model on all k+1 tokens in one forward pass
        let result = model(verifyText, cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        maybeQuantizeKVCache(
            cache: &cache, kvBits: kvBits,
            kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart)

        // result.logits shape: [1, k+1, vocab]
        // Position i gives logits for predicting the token AFTER position i
        // Position 0: logits for what comes after currentToken → should match draft[0]
        // Position k-1: logits for what comes after draft[k-2] → should match draft[k-1]
        // Position k: logits for what comes after draft[k-1] → next token
        let allLogits = result.logits

        var accepted: [Int] = []
        for i in 0..<k {
            // Get logits at position i (predicts token after position i)
            var logits_i = allLogits[0..., i, 0...]
            logits_i = processor?.process(logits: logits_i) ?? logits_i
            let sampled = sampler.sample(logits: logits_i)
            let sampledId = sampled.item(Int32.self)
            processor?.didSample(token: sampled)

            if sampledId == draftTokens[i] {
                accepted.append(Int(sampledId))
                ngramAccepted += 1
            } else {
                // Mismatch — use target's token as corrected output
                accepted.append(Int(sampledId))
                // Trim cache: we processed k+1 positions but only first (i+1) are valid
                // Need to trim (k - i) positions (the rejected draft tokens + the extra)
                let trimAmount = k - accepted.count
                if trimAmount > 0 {
                    for idx in 0..<cache.count {
                        cache[idx].trim(trimAmount)
                    }
                }
                // Set y to the corrected token for the next round
                y = .init(tokens: sampled)
                asyncEval(y.tokens)
                return accepted
            }
        }

        // All accepted — sample the bonus token from position k
        var logits_last = allLogits[0..., k, 0...]
        logits_last = processor?.process(logits: logits_last) ?? logits_last
        let sampled = sampler.sample(logits: logits_last)
        processor?.didSample(token: sampled)
        y = .init(tokens: sampled)
        asyncEval(y.tokens)

        return accepted
    }

    // MARK: - Iterator

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain pending tokens from last speculation round
        if !pendingTokens.isEmpty {
            let token = pendingTokens.removeFirst()
            tokenCount += 1
            if ngramSize > 0 { generatedTokenIds.append(Int32(token)) }
            return token
        }

        // Try n-gram speculation if enabled
        if ngramSize > 0 && cachesTrimmable {
            let accepted = ngramSpeculateRound()
            if !accepted.isEmpty {
                pendingTokens = Array(accepted.dropFirst())
                let firstToken = accepted[0]
                tokenCount += 1
                generatedTokenIds.append(Int32(firstToken))
                return firstToken
            }
        }

        // Standard single-token generation
        let previousY = y
        let profiling = DecodeCPUProfiler.enabled

        if profiling {
            os_signpost(.begin, log: decodeLog, name: "step", signpostID: decodeSignpost)
        }
        let token = step(previous: previousY)
        if profiling {
            os_signpost(.end, log: decodeLog, name: "step", signpostID: decodeSignpost)
        }

        y = .init(tokens: token)

        var t0: UInt64 = 0
        if profiling {
            t0 = DispatchTime.now().uptimeNanoseconds
        }
        asyncEval(token)

        if profiling {
            let t1 = DispatchTime.now().uptimeNanoseconds
            DecodeCPUProfiler.asyncEvalNs += (t1 - t0)
            t0 = t1
        }

        // Penalty processor ring update: deferred from convertToToken to here
        // so asyncEval triggers GPU eval first. The ring's MLX.where will find
        // the token already being evaluated (or evaluated), avoiding a premature
        // sync that blocks the pipeline.
        processor?.didSample(token: token)

        if profiling {
            DecodeCPUProfiler.didSampleNs += (DispatchTime.now().uptimeNanoseconds - t0)
            t0 = DispatchTime.now().uptimeNanoseconds
        }

        tokenCount += 1

        // Periodically clear GPU memory cache to prevent fragmentation
        // during long generations, matching Python mlx-lm behavior.
        if tokenCount % 256 == 0 {
            MLX.Memory.clearCache()
        }


        let tokenId = previousY.tokens.item(Int.self)

        if profiling {
            DecodeCPUProfiler.itemSyncNs += (DispatchTime.now().uptimeNanoseconds - t0)
            DecodeCPUProfiler.tokenCount += 1
        }

        if ngramSize > 0 { generatedTokenIds.append(Int32(tokenId)) }
        return tokenId
    }
}

/// Result of a call to a deprecated callback-based generate function.
public struct GenerateResult {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokens: The array of tokens generated.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokens = tokens
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    @available(*, deprecated, message: "use inputText")
    public var promptTokens: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    /// output tokens
    public let tokens: [Int]

    /// output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokens.count }

    /// time to process the prompt / generate the first token
    public let promptTime: TimeInterval

    /// time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokens.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in deprecated callback-based generate functions.
public enum GenerateDisposition: Sendable {
    /// keep producing tokens until an EOS token is produced
    case more

    /// stop producing tokens, e.g. a token limit has been hit
    case stop
}

private struct SynchronousGenerationLoopResult {
    let generatedTokens: [Int]
    let promptTime: TimeInterval
    let generateTime: TimeInterval
    let promptPrefillTime: TimeInterval
    let stopReason: GenerateStopReason
    let logProbSum: MLXArray
    let logProbTokenCount: Int
    let thinkingLogProbSum: MLXArray
    let thinkingLogProbCount: Int
    let generationLogProbSum: MLXArray
    let generationLogProbCount: Int
    let collectPerTokenData: Bool
    let perTokenLogProbs: [Float]
    let perTokenIds: [Int]
    let perTokenPhases: [String]
}

private func buildStopTokenIDs(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer
) -> Set<Int> {
    // Build complete EOS token set from all sources.
    var stopTokenIDs = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIDs.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIDs.insert(id)
        }
    }
    return stopTokenIDs
}

private func runSynchronousGenerationLoop(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: TokenIterator,
    didGenerate: (_ token: Int, _ generatedTokens: [Int]) -> GenerateDisposition
) -> SynchronousGenerationLoopResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let stopTokenIDs = buildStopTokenIDs(
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )

    var generatedTokens = [Int]()
    var iterator = iterator
    var stopReason: GenerateStopReason?

    while let token = iterator.next() {
        // Compute the timing for the prompt.
        if promptTime == 0 {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        // Check for end-of-sequence tokens.
        if token == tokenizer.unknownTokenId || stopTokenIDs.contains(token) {
            stopReason = .stop
            break
        }

        generatedTokens.append(token)

        if didGenerate(token, generatedTokens) == .stop {
            stopReason = .cancelled
            break
        }
    }

    // If the iterator ends naturally, the max-token limit was reached.
    if stopReason == nil {
        if let maxTokens = iterator.maxTokens, iterator.tokenCount >= maxTokens {
            stopReason = .length
        } else {
            stopReason = .cancelled
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // TokenIterator uses `asyncEval()` to keep the pipeline full. If the caller
    // exits the program right away, those tasks will still be executing and will
    // hit assertions as the mlx scheduler is torn down. Synchronize with the stream
    // to make sure it is complete.
    Stream().synchronize()

    // Print CPU decode profiling if enabled
    DecodeCPUProfiler.report()

    return SynchronousGenerationLoopResult(
        generatedTokens: generatedTokens,
        promptTime: promptTime,
        generateTime: generateTime,
        promptPrefillTime: iterator.promptPrefillTime,
        stopReason: stopReason ?? .cancelled,
        logProbSum: iterator.logProbSum,
        logProbTokenCount: iterator.logProbTokenCount,
        thinkingLogProbSum: iterator.thinkingLogProbSum,
        thinkingLogProbCount: iterator.thinkingLogProbCount,
        generationLogProbSum: iterator.generationLogProbSum,
        generationLogProbCount: iterator.generationLogProbCount,
        collectPerTokenData: iterator.collectPerTokenData,
        perTokenLogProbs: iterator.perTokenLogProbs,
        perTokenIds: iterator.perTokenIds,
        perTokenPhases: iterator.perTokenPhases
    )
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognize special tokens
///   - extraEOSTokens: any additional stop tokens
///   - didGenerate: visitor for the tokens as they are generated
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    // this is a compatibility cover -- create the required values
    // for the iteration
    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { _, generatedTokens in
        didGenerate(generatedTokens)
    }

    return GenerateResult(
        inputText: input.text, tokens: result.generatedTokens,
        output: context.tokenizer.decode(tokens: result.generatedTokens),
        promptTime: result.promptTime + result.promptPrefillTime,
        generateTime: result.generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { token, _ in
        didGenerate(token)
    }

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: result.generatedTokens.count,
        promptTime: result.promptTime + result.promptPrefillTime,
        generationTime: result.generateTime,
        stopReason: result.stopReason,
        averageLogProb: result.logProbTokenCount > 0
            ? Double(result.logProbSum.item(Float.self)) / Double(result.logProbTokenCount) : nil,
        thinkingAverageLogProb: result.thinkingLogProbCount > 0
            ? Double(result.thinkingLogProbSum.item(Float.self)) / Double(result.thinkingLogProbCount) : nil,
        generationAverageLogProb: result.generationLogProbCount > 0
            ? Double(result.generationLogProbSum.item(Float.self)) / Double(result.generationLogProbCount) : nil,
        perTokenLogProbs: result.collectPerTokenData ? result.perTokenLogProbs : nil,
        perTokenIds: result.collectPerTokenData ? result.perTokenIds : nil,
        perTokenPhases: result.collectPerTokenData ? result.perTokenPhases : nil
    )
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent text chunks, tool calls, or summary
/// completion information.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  This is typically OK for
/// one-shot calls, but for "chat session" type calls consider using
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:)``
/// so that the end of the generation task can be observed.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated text chunks (`.chunk`),
///   tool calls (`.toolCall`), and completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: generateParameters, context: context)
///
/// // Process the stream asynchronously to handle text chunks and completion info.
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     }
/// }
/// ```
public func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

@available(
    *, deprecated,
    message: "use a higher level generate() call or use generateTask() for fine grained control"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> AsyncStream<Generation> {
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Low-level token generation using a ``TokenIterator``, returning an
/// `AsyncStream<Generation>` and a `Task`.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  Callers can await
/// the `task` to observe when the use of the parameters is complete.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens and tool-call format)
///   - tokenizer: tokenizer (for EOS id, unknown token id, and detokenization)
///   - iterator: token iterator
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values and a `Task`
public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: tokenizer,
            format: modelConfiguration.toolCallFormat ?? .json
        )
    )
}

/// Generates raw token IDs asynchronously using the provided language model input, parameters, and context.
///
/// This is similar to `generate(input:cache:parameters:context:)`, but yields raw token IDs instead of decoded text/tool calls.
/// This is useful for downstream parsers that need access to token IDs directly (e.g. Harmony parsing).
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
    return stream
}

/// Generates raw token IDs asynchronously and returns the stream plus a `Task`.
///
/// Prefer this overload if you want to be able to observe when the underlying generation work is finished
/// (especially if the consumer terminates the stream early).
///
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values and a `Task`.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    return generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
}

/// Low-level raw token generation using a `TokenIterator`, returning an
/// `AsyncStream<TokenGeneration>` and a `Task`.
///
/// This is useful for parsers that need access to the token IDs directly (e.g. Harmony parsing)
/// without detokenization or tool-call parsing.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens)
///   - tokenizer: tokenizer (for EOS id and unknown token id)
///   - iterator: token iterator
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits token IDs and a final `.info`, plus a `Task`.
public func generateTokenTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        handler: RawTokenLoopHandler()
    )
}

private func generateLoopTask<Handler: TokenLoopHandler>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    includeStopToken: Bool = false,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<Void, Never>) {

    let (stream, continuation) = AsyncStream<Handler.Output>.makeStream()

    let iterator = SendableBox(iterator)
    let handler = SendableBox(handler)

    // Launch a Task to perform iteration asynchronously.
    let task = Task {
        let performIteration = {
            // Use `var` so next() mutates in-place. `for token in iterator` calls makeIterator()
            // which returns a copy for value types — the original iterator's perplexity counters
            // would be stale when read after the loop. `while let token = iterator.next()` avoids
            // the copy and reads the final state correctly.
            var iterator = iterator.consume()
            var handler = handler.consume()

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            let stopTokenIDs = buildStopTokenIDs(
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer
            )

            while let token = iterator.next() {
                // Check for cancellation on every loop iteration.
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                // Check for end-of-sequence tokens
                if token == tokenizer.unknownTokenId || stopTokenIDs.contains(token) {
                    if includeStopToken {
                        tokenCount += 1
                        if !handler.onStopToken(token, emit: continuation.yield) {
                            stopReason = .cancelled
                            break
                        }
                    }
                    stopReason = .stop
                    break
                }

                tokenCount += 1
                if !handler.onToken(token, emit: continuation.yield) {
                    stopReason = .cancelled
                    break
                }
            }

            if stopReason == nil {
                if Task.isCancelled {
                    stopReason = .cancelled
                } else if let maxTokens = iterator.maxTokens, iterator.tokenCount >= maxTokens {
                    stopReason = .length
                } else {
                    stopReason = .cancelled
                }
            }

            handler.onGenerationEnd(emit: continuation.yield)

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iterator.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled,
                averageLogProb: iterator.logProbTokenCount > 0
                    ? Double(iterator.logProbSum.item(Float.self)) / Double(iterator.logProbTokenCount) : nil,
                thinkingAverageLogProb: iterator.thinkingLogProbCount > 0
                    ? Double(iterator.thinkingLogProbSum.item(Float.self)) / Double(iterator.thinkingLogProbCount) : nil,
                generationAverageLogProb: iterator.generationLogProbCount > 0
                    ? Double(iterator.generationLogProbSum.item(Float.self)) / Double(iterator.generationLogProbCount) : nil,
                perTokenLogProbs: iterator.collectPerTokenData ? iterator.perTokenLogProbs : nil,
                perTokenIds: iterator.collectPerTokenData ? iterator.perTokenIds : nil,
                perTokenPhases: iterator.collectPerTokenData ? iterator.perTokenPhases : nil
            )
            _ = continuation.yield(handler.infoEvent(info))

            // Synchronize with the stream to ensure tasks are completed
            MLX.Stream().synchronize()

            // Print CPU decode profiling if enabled
            DecodeCPUProfiler.report()

            // Finalize the stream
            continuation.finish()
        }

        if let ticket = wiredMemoryTicket {
            await WiredMemoryTicket.withWiredLimit(ticket) {
                performIteration()
            }
        } else {
            performIteration()
        }
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { termination in
        if case .cancelled = termination {
            task.cancel()
        }
    }

    return (stream, task)
}

/// Measures the execution time of a closure.
private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}

// MARK: - Generation structs

/// Reason why token generation stopped.
public enum GenerateStopReason: Sendable {
    /// Generation stopped because an EOS/unknown stop token was encountered.
    case stop

    /// Generation stopped because the configured max token limit was reached.
    case length

    /// Generation stopped due to explicit task cancellation or early stream termination.
    case cancelled
}

/// Represents metadata and statistics related to token generation.
///
/// Provides information about the number of tokens processed during both the prompt and generation phases, as well as the time taken for each phase.
public struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    public let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    public let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    public let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    public let generateTime: TimeInterval

    /// Reason generation stopped.
    public let stopReason: GenerateStopReason

    /// Per-token average log probability (negative). nil if not computed.
    public let averageLogProb: Double?

    /// Per-token average log probability during the thinking phase (negative). nil if phase tracking not configured.
    public let thinkingAverageLogProb: Double?

    /// Per-token average log probability during the generation phase after </think> (negative). nil if phase tracking not configured.
    public let generationAverageLogProb: Double?

    /// Per-token log probs, token IDs, and phase labels. Only populated when
    /// `GenerateParameters.collectPerTokenData` is true.
    public let perTokenLogProbs: [Float]?
    public let perTokenIds: [Int]?
    public let perTokenPhases: [String]?

    /// Perplexity of all generated tokens: exp(-averageLogProb). nil if not computed.
    public var perplexity: Double? {
        guard let avgLogProb = averageLogProb else { return nil }
        return exp(-avgLogProb)
    }

    /// Perplexity of thinking-phase tokens (between <think> and </think>). nil if phase tracking not configured.
    public var thinkingPerplexity: Double? {
        guard let avg = thinkingAverageLogProb else { return nil }
        return exp(-avg)
    }

    /// Perplexity of generation-phase tokens (after </think>). nil if phase tracking not configured.
    public var generationPerplexity: Double? {
        guard let avg = generationAverageLogProb else { return nil }
        return exp(-avg)
    }

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: GenerateStopReason = .stop,
        averageLogProb: Double? = nil,
        thinkingAverageLogProb: Double? = nil,
        generationAverageLogProb: Double? = nil,
        perTokenLogProbs: [Float]? = nil,
        perTokenIds: [Int]? = nil,
        perTokenPhases: [String]? = nil
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
        self.stopReason = stopReason
        self.averageLogProb = averageLogProb
        self.thinkingAverageLogProb = thinkingAverageLogProb
        self.generationAverageLogProb = generationAverageLogProb
        self.perTokenLogProbs = perTokenLogProbs
        self.perTokenIds = perTokenIds
        self.perTokenPhases = perTokenPhases
    }

    public func summary() -> String {
        var s = """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
        if let ppl = perplexity {
            s += "\nPerplexity: \(String(format: "%.2f", ppl))"
        }
        return s
    }
}

/// Represents the different stages or outputs of the token generation process.
///
/// This enum distinguishes between the following:
/// - `.chunk`: A decoded string from one or more tokens generated by the language model.
/// - `.toolCall`: A tool call parsed from the generated output.
/// - `.info`: Metadata and performance statistics about the generation process.
public enum Generation: Sendable {
    /// A generated text chunk as a String.
    case chunk(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// A tool call from the language model.
    case toolCall(ToolCall)

    /// Generated text or nil
    public var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .info: nil
        case .toolCall: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .info(let info): info
        case .toolCall: nil
        }
    }

    /// Tool call or nil
    public var toolCall: ToolCall? {
        switch self {
        case .chunk: nil
        case .info: nil
        case .toolCall(let toolCall): toolCall
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}

/// Represents the different stages or outputs of raw-token generation.
///
/// This mirrors `Generation`, but yields raw token IDs instead of decoded text/tool calls.
public enum TokenGeneration: Sendable {
    /// A generated token ID.
    case token(Int)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// Token ID or nil
    public var token: Int? {
        switch self {
        case .token(let token): token
        case .info: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .token: nil
        case .info(let info): info
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [TokenGeneration]?, _ element: TokenGeneration)
        -> [TokenGeneration]
    {
        (batch ?? []) + [element]
    }
}

// MARK: - TokenLoopHandlers

private protocol TokenLoopHandler: Sendable {
    associatedtype Output

    /// Return false to stop the loop early.
    mutating func onToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called only when includeStopToken == true and a stop token was hit.
    mutating func onStopToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called after the token loop finishes, before the info event.
    mutating func onGenerationEnd(
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    )

    func infoEvent(_ info: GenerateCompletionInfo) -> Output
}

private struct TextToolTokenLoopHandler: TokenLoopHandler, @unchecked Sendable {
    typealias Output = Generation

    var detokenizer: NaiveStreamingDetokenizer
    let toolCallProcessor: ToolCallProcessor

    init(tokenizer: Tokenizer, format: ToolCallFormat) {
        detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        toolCallProcessor = ToolCallProcessor(format: format)
    }

    mutating func onToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            // Process chunk through the tool call processor.
            if let textToYield = toolCallProcessor.processChunk(chunk) {
                if case .terminated = emit(.chunk(textToYield)) {
                    return false
                }
            }

            // Check if we have a complete tool call.
            if let toolCall = toolCallProcessor.toolCalls.popLast() {
                if case .terminated = emit(.toolCall(toolCall)) {
                    return false
                }
            }
        }

        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        true
    }

    mutating func onGenerationEnd(
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) {
        toolCallProcessor.processEOS()

        for toolCall in toolCallProcessor.toolCalls {
            if case .terminated = emit(.toolCall(toolCall)) {
                break
            }
        }
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> Generation {
        .info(info)
    }
}

private struct RawTokenLoopHandler: TokenLoopHandler {
    typealias Output = TokenGeneration

    mutating func onToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onGenerationEnd(
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) {}

    func infoEvent(_ info: GenerateCompletionInfo) -> TokenGeneration {
        .info(info)
    }
}
