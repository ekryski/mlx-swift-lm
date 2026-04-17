// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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

    /// Called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// Called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// Called to provide the sampled token
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
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// Sampling temperature
    public var temperature: Float

    /// Top-p sampling
    public var topP: Float

    /// Top-k sampling (0 disables)
    public var topK: Int

    /// Min-p sampling threshold relative to the highest probability token (0 disables)
    public var minP: Float

    /// Penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// Number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// additive penalty for tokens that appear in recent context
    public var presencePenalty: Float?

    /// number of tokens to consider for presence penalty
    public var presenceContextSize: Int

    /// additive penalty that scales with token frequency in recent context
    public var frequencyPenalty: Float?

    /// number of tokens to consider for frequency penalty
    public var frequencyContextSize: Int

    /// KV cache compression scheme. nil = use kvBits (affine quantization) if set.
    /// "turbo1" through "turbo4" = TurboQuant compression at 1-4 bits.
    /// When set, kvBits is ignored for cache creation.
    public var kvScheme: String?

    /// Additional logit processors applied after built-in penalty processors.
    /// Use this to inject custom processors (e.g., EOS suppression for thinking models).
    public nonisolated(unsafe) var additionalProcessors: [any LogitProcessor]

    /// Reasoning effort hint (e.g., "low", "medium", "high")
    public var reasoningEffort: String?

    /// N-gram size for prompt-lookup speculative decoding. When > 0, the iterator
    /// searches for matching n-grams in the prompt text and uses continuations as
    /// draft tokens, verifying them in a single batched forward pass.
    ///
    /// **Default: 0 (disabled).** The speculation path is a net-win only when the
    /// generated text has enough repetition that the trigram table hits with a
    /// reasonable acceptance rate. For novel prose / summarisation / chat, accept
    /// rate is often low and the draft work becomes pure overhead. Enable
    /// explicitly (`ngramSize: 3`, or higher) when your workload has been
    /// validated to benefit — e.g. code-heavy output or repetitive templates.
    public var ngramSize: Int

    /// Maximum draft tokens per n-gram speculation round. Defaults to 0 so that
    /// a bare `GenerateParameters()` disables speculation end-to-end; must be
    /// set in tandem with `ngramSize` when enabling.
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

    /// Token ID of the harmony channel marker, typically `<|channel|>` (e.g. 200005
    /// on the GPT-OSS tokenizer). When set, phase labeling uses a harmony state
    /// machine: the token immediately following the marker selects the phase via
    /// ``harmonyThinkingChannelTokenIds`` / ``harmonyGenerationChannelTokenIds``.
    /// Takes precedence over ``thinkStartTokenId`` / ``thinkEndTokenId``.
    public var harmonyChannelMarkerTokenId: Int32?

    /// Token IDs whose appearance immediately after the harmony channel marker
    /// transitions the phase to "think" (e.g. `analysis` → 35644 on GPT-OSS).
    public var harmonyThinkingChannelTokenIds: [Int32]

    /// Token IDs whose appearance immediately after the harmony channel marker
    /// transitions the phase to "gen" (e.g. `final` → 17196 on GPT-OSS).
    public var harmonyGenerationChannelTokenIds: [Int32]

    /// When true, per-token log probs, token IDs, and phase labels are stored
    /// in the TokenIterator for downstream KLD computation.
    public var collectPerTokenData: Bool

    /// When true, accumulate log probabilities for perplexity computation.
    /// Default: false. Set to true when perplexity tracking is needed.
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
        kvScheme: String? = nil,
        additionalProcessors: [any LogitProcessor] = [],
        reasoningEffort: String? = nil,
        ngramSize: Int = 0,
        maxNgramDraftTokens: Int = 0,
        thinkStartTokenId: Int32? = nil,
        thinkEndTokenId: Int32? = nil,
        thinkingPhasePrefilled: Bool = false,
        harmonyChannelMarkerTokenId: Int32? = nil,
        harmonyThinkingChannelTokenIds: [Int32] = [],
        harmonyGenerationChannelTokenIds: [Int32] = [],
        collectPerTokenData: Bool = false,
        trackPerplexity: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
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
        self.kvScheme = kvScheme
        self.additionalProcessors = additionalProcessors
        self.reasoningEffort = reasoningEffort
        self.ngramSize = ngramSize
        self.maxNgramDraftTokens = maxNgramDraftTokens
        self.thinkStartTokenId = thinkStartTokenId
        self.thinkEndTokenId = thinkEndTokenId
        self.thinkingPhasePrefilled = thinkingPhasePrefilled
        self.harmonyChannelMarkerTokenId = harmonyChannelMarkerTokenId
        self.harmonyThinkingChannelTokenIds = harmonyThinkingChannelTokenIds
        self.harmonyGenerationChannelTokenIds = harmonyGenerationChannelTokenIds
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

    public func processor() -> (any LogitProcessor)? {
        var all: [any LogitProcessor] = []

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
/// to sample the logits.
///
/// Filters are applied in the same order as Python mlx-lm: top_p → min_p → top_k.
/// Each filter operates on the full vocabulary in original token order, masking
/// rejected tokens with `-inf`. This matches the composable filter chain in
/// `mlx_lm.sample_utils.make_sampler`.
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

            // Apply filters in Python mlx-lm order: top_p → min_p → top_k.
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
    /// Matches `apply_top_p` from `mlx_lm/sample_utils.py`.
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Mask low-probability tail in sorted order, scatter back to original vocab order.
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP.
    /// Matches `apply_min_p` from `mlx_lm/sample_utils.py`.
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP)
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens.
    /// Mirrors `apply_top_k` from `mlx_lm/sample_utils.py`.
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        // O(V) partition on negated logprobs so top-k land at [0, topK).
        // Indices at [topK, V) are the tokens to mask out.
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
///
/// Shared by penalty processors to avoid duplicating ring buffer logic.
/// Uses `MLX.where` mask operations for GPU-only updates (no CPU←GPU sync),
/// preserving `asyncEval()` pipelining in `TokenIterator`.
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

    /// The valid portion of the ring (all of it once full), or `nil` if empty.
    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    /// Bulk-load from a prompt. Keeps the last `capacity` tokens.
    ///
    /// Accepts either a 1D `[seqLen]` or 2D `[1, seqLen]` prompt — `LMInput.text.tokens`
    /// is typically 2D after `container.prepare(input:)`. Flatten first so `n` reflects
    /// the sequence length regardless of whether a batch dimension is present.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let promptTokens = prompt.asType(.int32).reshaped(-1)
        let n = promptTokens.dim(0)
        if n <= capacity {
            if n < capacity {
                let padding = MLXArray.zeros([capacity - n], type: Int32.self)
                buffer = concatenated([promptTokens, padding])
            } else {
                buffer = promptTokens
            }
            count = n
            writeIndex = n % capacity
        } else {
            buffer = promptTokens[(-capacity)...]
            count = capacity
            writeIndex = 0
        }
    }

    /// Append a single token using GPU-only mask write (no CPU←GPU sync).
    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Processor that implements a `repetitionPenalty`.
public struct RepetitionContext: LogitProcessor {
    private var ring: TokenRing
    let repetitionPenalty: Float

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
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
///
/// The penalty is applied once per unique token via scatter-write (writing the
/// same value to the same index multiple times is idempotent).
public struct PresencePenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let presencePenalty: Float

    public init(presencePenalty: Float, presenceContextSize: Int) {
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        logits[0..., indices] = logits[0..., indices] - presencePenalty
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
///
/// Frequency counting is performed on GPU via `scatter_add` to build a histogram
/// of token occurrences, avoiding CPU←GPU synchronization.
public struct FrequencyPenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let frequencyPenalty: Float

    public init(frequencyPenalty: Float, frequencyContextSize: Int) {
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

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that composes penalty processors in Python mlx-lm order.
public struct PenaltyProcessor: LogitProcessor {
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?

    public init(
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?
    ) {
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
    }

    mutating public func prompt(_ prompt: MLXArray) {
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
    }
}

/// Composes multiple logit processors into a single processor.
public struct CompositeLogitProcessor: LogitProcessor {
    var processors: [any LogitProcessor]

    public init(processors: [any LogitProcessor]) {
        self.processors = processors
    }

    mutating public func prompt(_ prompt: MLXArray) {
        for i in 0..<processors.count {
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
        for i in 0..<processors.count {
            processors[i].didSample(token: token)
        }
    }
}

// MARK: - Per-Token Data Capture (opt-in)

/// Per-token data produced during generation when
/// ``GenerateParameters/trackPerplexity`` or ``GenerateParameters/collectPerTokenData``
/// is set.
///
/// Surfaces on ``GenerateCompletionInfo``. Downstream consumers such as the
/// benchmark harness use `tokenIds` + `logProbs` to compute KL divergence
/// against a baseline model via forced decode; `thinkingPerplexity` and
/// `generationPerplexity` are the model's own per-phase perplexity over
/// the generated tokens it sampled.
public struct PerTokenData: Sendable {
    public let tokenIds: [Int]
    public let logProbs: [Float]
    public let phases: [String]
    public let thinkingPerplexity: Float?
    public let generationPerplexity: Float?

    public init(
        tokenIds: [Int],
        logProbs: [Float],
        phases: [String],
        thinkingPerplexity: Float?,
        generationPerplexity: Float?
    ) {
        self.tokenIds = tokenIds
        self.logProbs = logProbs
        self.phases = phases
        self.thinkingPerplexity = thinkingPerplexity
        self.generationPerplexity = generationPerplexity
    }
}

/// Reference-type box holding the lazy MLXArray handles accumulated during a
/// generation run.
///
/// `TokenIterator` is a struct and `for token in iterator { … }` copies it into
/// the for-in machinery before iterating. A reference-type capture box keeps
/// the data the copy appends visible to the outer iterator so the final flush
/// can happen after the loop returns.
///
/// Entries are lazy — no GPU→CPU transfer happens per step; all transfers are
/// batched in ``TokenIterator/finalizePerTokenData()`` after
/// `Stream().synchronize()`.
final class PerTokenDataCapture {
    var logProbs: [MLXArray] = []
    var tokenIds: [MLXArray] = []
}

extension GenerateParameters {
    /// True when the generation loop must emit per-token logprobs for PPL or KLD.
    /// When false, `TokenIterator` leaves its capture slot nil and runs the
    /// inference path unchanged.
    var needsPerTokenCapture: Bool {
        trackPerplexity || collectPerTokenData
    }
}

/// Common properties shared by token-generating iterators.
protocol TokenIteratorProtocol: Sequence, IteratorProtocol where Element == Int {
    var maxTokens: Int? { get }
    var tokenCount: Int { get }
    var promptPrefillTime: TimeInterval { get }

    /// Flush lazy per-token data to CPU, label tokens with phase, and compute
    /// per-phase perplexity. Returns nil when per-token capture was not enabled.
    ///
    /// Must be called only after `Stream().synchronize()` so all lazy logprob
    /// MLXArrays have completed. Implementations must perform a single batched
    /// GPU→CPU transfer rather than per-token `.item()` calls.
    func finalizePerTokenData() -> PerTokenData?
}

extension TokenIteratorProtocol {
    /// Default: iterators that don't capture per-token data return nil.
    public func finalizePerTokenData() -> PerTokenData? { nil }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:cache:parameters:context:wiredMemoryTicket:)`` returning `AsyncStream<Generation>`.
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
public struct TokenIterator: TokenIteratorProtocol {
    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: (any LogitProcessor)?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let kvScheme: String?

    // Phase tracking for per-token data capture (cheap Ints / Sets / Bool,
    // only read at finalize time — no inference-loop cost). Three mutually
    // exclusive modes:
    //   • harmonyChannelMarkerTokenId set → harmony channel transitions.
    //   • thinkStartTokenId + thinkEndTokenId set → bracket pair.
    //   • neither set → all tokens labelled "gen".
    let thinkStartTokenId: Int?
    let thinkEndTokenId: Int?
    let thinkingPhasePrefilled: Bool
    let harmonyChannelMarkerTokenId: Int?
    let harmonyThinkingChannelTokenIds: Set<Int>
    let harmonyGenerationChannelTokenIds: Set<Int>

    /// Lazy per-token logprob/id capture. Non-nil only when parameters enable
    /// ``GenerateParameters/trackPerplexity`` or
    /// ``GenerateParameters/collectPerTokenData``. The `if let` check in
    /// ``convertToToken(logits:)`` is the only inference-path cost when
    /// capture is disabled.
    let perTokenCapture: PerTokenDataCapture?

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

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

        self.thinkStartTokenId = parameters.thinkStartTokenId.map { Int($0) }
        self.thinkEndTokenId = parameters.thinkEndTokenId.map { Int($0) }
        self.thinkingPhasePrefilled = parameters.thinkingPhasePrefilled
        self.harmonyChannelMarkerTokenId = parameters.harmonyChannelMarkerTokenId.map { Int($0) }
        self.harmonyThinkingChannelTokenIds = Set(parameters.harmonyThinkingChannelTokenIds.map { Int($0) })
        self.harmonyGenerationChannelTokenIds = Set(parameters.harmonyGenerationChannelTokenIds.map { Int($0) })
        self.perTokenCapture = parameters.needsPerTokenCapture ? PerTokenDataCapture() : nil

        self.promptPrefillTime = try measure {
            try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
        }
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:maxTokens:)``
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

        self.thinkStartTokenId = parameters.thinkStartTokenId.map { Int($0) }
        self.thinkEndTokenId = parameters.thinkEndTokenId.map { Int($0) }
        self.thinkingPhasePrefilled = parameters.thinkingPhasePrefilled
        self.harmonyChannelMarkerTokenId = parameters.harmonyChannelMarkerTokenId.map { Int($0) }
        self.harmonyThinkingChannelTokenIds = Set(parameters.harmonyThinkingChannelTokenIds.map { Int($0) })
        self.harmonyGenerationChannelTokenIds = Set(parameters.harmonyGenerationChannelTokenIds.map { Int($0) })
        self.perTokenCapture = parameters.needsPerTokenCapture ? PerTokenDataCapture() : nil

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
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
        processor: (any LogitProcessor)?, sampler: LogitSampler, prefillStepSize: Int = 512,
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

        // The manual init does not carry thinking config or per-token capture.
        // Callers that need PPL/KLD should use the `parameters:` init instead.
        self.thinkStartTokenId = nil
        self.thinkEndTokenId = nil
        self.thinkingPhasePrefilled = false
        self.harmonyChannelMarkerTokenId = nil
        self.harmonyThinkingChannelTokenIds = []
        self.harmonyGenerationChannelTokenIds = []
        self.perTokenCapture = nil

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: prefillStepSize)
        }
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)

            // Sync-eval first decode token — asyncEval causes pad-token bug on Gemma 4
            // (second decode forward pass starts before prefill's KV writes commit).
            eval(y.tokens)

        case .logits(let result):
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)

            break
        }
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        // process the logits (one hot array of possible tokens)
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits

        // transform logits back to a token
        let y = sampler.sample(logits: logits)

        // Opt-in per-token data capture for PPL / KLD.
        //
        // Zero inference-path overhead when `perTokenCapture == nil` — the
        // guard is a single pointer nil-check the optimizer can hoist.
        //
        // When enabled, we emit a scalar-output kernel chain per step:
        //   1. `logSumExp(logits, axes: [-1])` — one reduction producing [1].
        //   2. `takeAlong(logits, y_exp, axis: -1)` — scalar gather producing [1, 1].
        //   3. subtract to get logprob[y] = logits[y] - logSumExp(logits).
        //
        // This avoids materialising a full-vocab logSoftmax output tensor
        // (V can be 250k+ tokens → ~1MB per step on decode). Both branches
        // depend on `logits` only — no data dependency on `y` for the
        // reduction — so the reduction runs concurrently with sampling.
        // `asyncEval(selected)` detaches the scalar subgraph from the
        // token-eval pipeline so the next forward pass is not blocked on
        // the logprob.
        if let capture = perTokenCapture {
            let logSumExpLogits = logSumExp(logits, axes: [-1])
            let yExpanded = y.expandedDimensions(axis: -1)
            let yLogit = takeAlong(logits, yExpanded, axis: -1)
            let selected = yLogit - logSumExpLogits
            asyncEval(selected)
            capture.logProbs.append(selected)
            capture.tokenIds.append(y)
        }

        processor?.didSample(token: y)

        return y
    }

    /// Flush lazy per-token data collected during generation.
    ///
    /// Called by the generation loop after `Stream().synchronize()` so all
    /// captured logprob and token-id MLXArrays have finished evaluating on
    /// the GPU. Performs one batched GPU→CPU transfer for the whole run,
    /// assigns phase labels ("think" / "gen") based on the
    /// model's configured `thinkStartTokenId` / `thinkEndTokenId`, and
    /// computes `exp(-mean(logprobs))` per phase.
    ///
    /// Returns `nil` when per-token capture was not enabled for this run.
    public func finalizePerTokenData() -> PerTokenData? {
        guard let capture = perTokenCapture, !capture.logProbs.isEmpty else {
            return nil
        }

        // Batch the GPU→CPU transfer: concatenate all the 1-element logprob
        // tensors into a single [N] tensor, then pull it to CPU once.
        let reshapedLogProbs = capture.logProbs.map { $0.reshaped(1) }
        let reshapedIds = capture.tokenIds.map { $0.reshaped(1) }
        let stackedLogprobs = concatenated(reshapedLogProbs).asType(.float32)
        let stackedIds = concatenated(reshapedIds).asType(.int32)

        let floatLogprobs = stackedLogprobs.asArray(Float.self)
        let intIds = stackedIds.asArray(Int32.self).map { Int($0) }

        // Phase labels — pure Swift, zero GPU work. Three modes dispatched
        // by which fields were populated at init time:
        //   1. Harmony channel mode (GPT-OSS):  <|channel|> <channel-name>
        //      <|message|> ...  where the channel-name token flips phase.
        //   2. Bracket mode (Qwen, Gemma 4):    <think> ... </think>
        //      where a single token opens and another closes the block.
        //   3. None:                            every token labelled
        //      "gen".
        let phases: [String] = labelPhases(intIds)

        func perplexity(for phase: String) -> Float? {
            var sum: Double = 0
            var count = 0
            for i in intIds.indices where phases[i] == phase {
                sum += Double(floatLogprobs[i])
                count += 1
            }
            guard count > 0 else { return nil }
            return Float(exp(-sum / Double(count)))
        }

        return PerTokenData(
            tokenIds: intIds,
            logProbs: floatLogprobs,
            phases: phases,
            thinkingPerplexity: perplexity(for: "think"),
            generationPerplexity: perplexity(for: "gen")
        )
    }

    /// Assign a phase label ("think" / "gen") to every generated token.
    ///
    /// Harmony mode (GPT-OSS and other harmony-format models) is a 2-state
    /// state machine: on each token we check whether it is the channel marker
    /// (e.g. `<|channel|>` → 200005). The very next token is the channel name
    /// (`analysis` → thinking, `final` / `commentary` → generation). The
    /// marker, name, and subsequent message tokens are all attributed to the
    /// resolved phase. Unknown channel names keep the prior phase.
    ///
    /// Bracket mode (Qwen, Gemma 4) is a single-token boundary: seeing the
    /// configured `thinkStartTokenId` opens the block, seeing the
    /// `thinkEndTokenId` closes it; both markers themselves stay in the
    /// "think" bucket.
    private func labelPhases(_ intIds: [Int]) -> [String] {
        if let markerId = harmonyChannelMarkerTokenId {
            var phases: [String] = []
            phases.reserveCapacity(intIds.count)
            var inThinking = thinkingPhasePrefilled
            var awaitingChannelName = false
            for id in intIds {
                if id == markerId {
                    awaitingChannelName = true
                    // Attribute marker to current phase — transition hasn't
                    // been resolved yet.
                    phases.append(inThinking ? "think" : "gen")
                    continue
                }
                if awaitingChannelName {
                    awaitingChannelName = false
                    if harmonyThinkingChannelTokenIds.contains(id) {
                        inThinking = true
                    } else if harmonyGenerationChannelTokenIds.contains(id) {
                        inThinking = false
                    }
                    // Unknown channel names fall through with phase unchanged.
                    phases.append(inThinking ? "think" : "gen")
                    continue
                }
                phases.append(inThinking ? "think" : "gen")
            }
            return phases
        }

        // Bracket mode.
        var phases: [String] = []
        phases.reserveCapacity(intIds.count)
        var inThinking = thinkStartTokenId != nil && thinkingPhasePrefilled
        for id in intIds {
            if !inThinking, let startId = thinkStartTokenId, id == startId {
                inThinking = true
                phases.append("think")
                continue
            }
            if inThinking, let endId = thinkEndTokenId, id == endId {
                inThinking = false
                // The end-of-think token itself is part of the thinking phase.
                phases.append("think")
                continue
            }
            phases.append(inThinking ? "think" : "gen")
        }
        return phases
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        let result = model(
            previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )

        return convertToToken(logits: result.logits)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // save current value -- this will be returned
        let previousY = y

        // compute the next state and async eval the next token
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)

        tokenCount += 1

        return previousY.tokens.item(Int.self)
    }
}

// MARK: - N-gram speculative decoding

/// Prompt-lookup speculative draft source.
///
/// Builds a rolling map from the last-N-tokens suffix → list of positions
/// where that n-gram occurs in the token history. On each speculation round,
/// the iterator takes the last `ngramSize` tokens of its generated prefix,
/// looks them up, and proposes the continuation as draft tokens.
///
/// Stays entirely on CPU (Swift dictionary + arrays). Rolling: generated
/// tokens get added to the history so self-repetition during generation
/// produces hits too.
final class NGramLookup {
    /// Token history — prompt followed by accepted generated tokens.
    private var tokens: [Int]
    /// Map from FNV-1a hash of the last-`ngramSize` suffix → token positions
    /// in `tokens` where that n-gram ends. Collisions are handled on the
    /// verify side (a bad draft is just rejected — correctness preserved).
    private var table: [UInt64: [Int]]
    let ngramSize: Int

    init(promptTokens: [Int], ngramSize: Int) {
        precondition(ngramSize >= 1, "ngramSize must be >= 1")
        self.tokens = promptTokens
        self.table = [:]
        self.ngramSize = ngramSize
        rebuildTable()
    }

    /// FNV-1a 64-bit hash over the last `ngramSize` tokens ending at `endIdx`
    /// (inclusive). Rolling update is not worth it at these sizes.
    private func hashNgramEndingAt(_ endIdx: Int) -> UInt64? {
        let start = endIdx - ngramSize + 1
        guard start >= 0 else { return nil }
        var h: UInt64 = 14_695_981_039_346_656_037  // FNV-1a offset basis
        let prime: UInt64 = 1_099_511_628_211
        for i in start ... endIdx {
            var t = UInt64(bitPattern: Int64(tokens[i]))
            for _ in 0 ..< 8 {
                h ^= (t & 0xff)
                h = h &* prime
                t >>= 8
            }
        }
        return h
    }

    private func rebuildTable() {
        table.removeAll(keepingCapacity: true)
        // For each position `i` where an n-gram can END, store i.
        // Continuations = tokens[i+1 ..< i+1+k].
        guard tokens.count >= ngramSize else { return }
        for i in (ngramSize - 1) ..< tokens.count {
            if let h = hashNgramEndingAt(i) {
                table[h, default: []].append(i)
            }
        }
    }

    /// Append newly-accepted tokens to the history and extend the table so
    /// future lookups see the updated prefix.
    func extend(with newTokens: [Int]) {
        let startIdx = tokens.count
        tokens.append(contentsOf: newTokens)
        // New n-grams end at indices `[startIdx, tokens.count)` — for each,
        // compute the hash and append to the table. This amortises O(k) work
        // per appended token rather than re-scanning the whole history.
        let firstNewEnd = max(startIdx, ngramSize - 1)
        guard firstNewEnd < tokens.count else { return }
        for i in firstNewEnd ..< tokens.count {
            if let h = hashNgramEndingAt(i) {
                table[h, default: []].append(i)
            }
        }
    }

    /// Propose up to `maxDraft` continuation tokens given the last
    /// `ngramSize` tokens of `tokens`. Returns an empty array on miss.
    ///
    /// On a hit, we return the continuation of the **most recent** occurrence
    /// (last entry in the positions list) — recent context is likely a better
    /// prior than ancient context.
    func proposeDraft(maxDraft: Int) -> [Int] {
        guard tokens.count >= ngramSize, maxDraft > 0 else { return [] }
        let lastEnd = tokens.count - 1
        guard let h = hashNgramEndingAt(lastEnd),
              let positions = table[h],
              let mostRecentBeforeEnd = positions.last(where: { $0 < lastEnd })
        else {
            return []
        }
        // Continuation starts at mostRecentBeforeEnd + 1.
        let continuationStart = mostRecentBeforeEnd + 1
        let continuationEnd = min(continuationStart + maxDraft, tokens.count)
        if continuationStart >= continuationEnd {
            return []
        }
        return Array(tokens[continuationStart ..< continuationEnd])
    }
}

/// Generator of tokens with **n-gram prompt-lookup speculative decoding**.
///
/// Unlike ``SpeculativeTokenIterator`` which needs a separate draft model,
/// this iterator sources draft tokens from the token history itself — prompt
/// tokens and already-generated tokens. Works best when the generation has
/// repetitive structure (boilerplate, code, templates, factual regurgitation
/// of the prompt).
///
/// Enable via `GenerateParameters.ngramSize > 0` and
/// `maxNgramDraftTokens > 0`. With both zero (default), construction throws —
/// callers should switch to ``TokenIterator`` for non-speculative decode.
///
/// Greedy-equivalent: verification accepts a drafted token only when it
/// matches the main model's argmax at the same position, so the output
/// stream is identical to running ``TokenIterator`` at temperature=0.
public struct NGramSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let mainModel: any LanguageModel
    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    let sampler: LogitSampler
    var tokenCount = 0
    let maxTokens: Int?

    // N-gram config
    let ngramSize: Int
    let maxNgramDraftTokens: Int
    var lookup: NGramLookup

    // Per-round emission buffer
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    /// Prompt prefill time (ms), measured once at init.
    public private(set) var promptPrefillTime: TimeInterval = 0.0

    /// Tokens accepted from n-gram lookup (for acceptance-rate tracking).
    public private(set) var ngramAcceptedCount = 0

    /// Total n-gram tokens proposed.
    public private(set) var ngramProposedCount = 0

    /// Acceptance rate = accepted / proposed. Zero when no rounds have hit.
    public var ngramAcceptanceRate: Double {
        guard ngramProposedCount > 0 else { return 0 }
        return Double(ngramAcceptedCount) / Double(ngramProposedCount)
    }

    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        precondition(
            parameters.ngramSize >= 1 && parameters.maxNgramDraftTokens >= 1,
            "NGramSpeculativeTokenIterator requires ngramSize >= 1 and "
                + "maxNgramDraftTokens >= 1. Use TokenIterator for "
                + "non-speculative decode.")

        self.y = input.text
        self.mainModel = mainModel

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache) else {
            throw KVCacheError(
                message: "N-gram speculative decoding requires trimmable KV caches.")
        }

        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.ngramSize = parameters.ngramSize
        self.maxNgramDraftTokens = parameters.maxNgramDraftTokens

        let promptTokens = input.text.tokens.asArray(Int.self)
        self.lookup = NGramLookup(
            promptTokens: promptTokens, ngramSize: parameters.ngramSize)

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Prefill the main model with the prompt, sample the first token.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            y = .init(tokens: token)
            mainState = result.state
        }
    }

    /// One speculation round: look up draft tokens, verify with main model,
    /// emit accepted tokens to `pendingTokens`.
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? maxNgramDraftTokens
        let budget = Swift.min(remaining, maxNgramDraftTokens)
        guard budget > 0 else { return }

        let draftInts = lookup.proposeDraft(maxDraft: budget)

        if draftInts.isEmpty {
            // Miss — pure autoregressive step.
            let result = mainModel(y[text: .newAxis], cache: mainCache, state: mainState)
            quantizeKVCache(&mainCache)
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            asyncEval(token)
            mainState = result.state
            let tokenInt = token.item(Int.self)
            pendingTokens.append(tokenInt)
            lookup.extend(with: [tokenInt])
            y = .init(tokens: token)
            return
        }

        let numDraft = draftInts.count
        let draftArray = MLXArray(draftInts.map { Int32($0) })

        // Verification: main model processes [y, draft_1 ... draft_k] in one pass.
        let verifyTokens = concatenated([y.tokens, draftArray])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: mainCache, state: mainState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        // Argmax per position. This is identical to what the sampler would
        // produce under temperature=0; non-greedy samplers would need a
        // per-position resample loop instead (tracked as follow-up).
        let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
        let mainTokens = sampler.sample(logits: verifyLogits)
        eval(mainTokens)
        let mainList = mainTokens.asArray(Int.self)

        var accepted = 0
        for i in 0 ..< numDraft where mainList[i] == draftInts[i] {
            pendingTokens.append(mainList[i])
            accepted += 1
        }

        ngramAcceptedCount += accepted
        ngramProposedCount += numDraft

        // The main model's token at position `accepted` is always emitted —
        // either the correction after the first rejected draft, or the
        // bonus token after a full-accept. Counts as a non-draft emission.
        let finalTokenInt = mainList[accepted]
        pendingTokens.append(finalTokenInt)

        // Trim the KV cache by the number of rejected draft tokens — their
        // K/V rows must be undone so the cache offset matches what the
        // outer world thinks it has.
        let rejected = numDraft - accepted
        trimPromptCache(mainCache, numTokens: rejected)
        quantizeKVCache(&mainCache)

        // Extend lookup with all the real tokens we just committed.
        let emitted = pendingTokens.suffix(accepted + 1)
        lookup.extend(with: Array(emitted))

        // Next round starts from the final emitted token.
        y = .init(tokens: mainTokens[accepted ... accepted])
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let t = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return t
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        guard !pendingTokens.isEmpty else { return nil }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return t
    }
}

/// Generator of tokens using speculative decoding.
///
/// This is typically used via a call to ``generate(input:cache:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)``
/// returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let mainModel: LanguageModel
/// let draftModel: LanguageModel
///
/// let iterator = try SpeculativeTokenIterator(
///     input: input, mainModel: mainModel, draftModel: draftModel,
///     parameters: generateParameters, numDraftTokens: 2)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `speculative_generate_step()` from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
public struct SpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    var draftY: LMInput.Text

    let mainModel: any LanguageModel
    let draftModel: any LanguageModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    var draftCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: (any LogitProcessor)?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?
    let numDraftTokens: Int

    // Buffer of accepted tokens from the current speculation round
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

    /// Tokens accepted from draft model (for acceptance rate tracking).
    public private(set) var draftAcceptedCount = 0

    /// Total draft tokens proposed (for acceptance rate tracking).
    public private(set) var draftProposedCount = 0

    /// The acceptance rate of draft tokens (0.0 to 1.0).
    public var acceptanceRate: Double {
        guard draftProposedCount > 0 else { return 0 }
        return Double(draftAcceptedCount) / Double(draftProposedCount)
    }

    /// Initialize a `SpeculativeTokenIterator` with the given input.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - mainModel: the main (verifier) ``LanguageModel``
    ///   - draftModel: the draft ``LanguageModel`` (must share the same tokenizer)
    ///   - mainCache: optional ``KVCache`` for the main model
    ///   - draftCache: optional ``KVCache`` for the draft model
    ///   - parameters: the generation parameters
    ///   - numDraftTokens: number of tokens the draft model proposes per round
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int
    ) throws {
        self.y = input.text
        self.draftY = input.text
        self.mainModel = mainModel
        self.draftModel = draftModel

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        self.draftCache = draftCache ?? draftModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache), canTrimPromptCache(self.draftCache) else {
            throw KVCacheError(message: "Speculative decoding requires trimmable KV caches.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()

        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Prefill both main and draft models with the prompt, priming caches for generation
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        // Prefill main model
        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
        }

        // Prefill draft model, don't call didSample here -- processor tracks main model's accepted sequence only
        switch try draftModel.prepare(input, cache: draftCache, windowSize: windowSize) {
        case .tokens(let tokens):
            draftY = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            draftY = .init(tokens: token)
            asyncEval(draftY.tokens)
        }
    }

    /// Run one round of speculative decoding: draft, verify, accept/reject
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else {
            return
        }

        // Snapshot MambaCache layers before speculation (SSM state is cumulative,
        // not trimmable, so we must restore on rejection)
        snapshotMambaCaches(mainCache)
        snapshotMambaCaches(draftCache)

        // Draft generation: autoregressive loop with draft model
        var draftProcessor = processor  // Copy to discard later
        var draftTokens = [MLXArray]()
        for _ in 0 ..< numDraft {
            let draftResult = draftModel(draftY[text: .newAxis], cache: draftCache, state: nil)
            quantizeKVCache(&draftCache)
            var draftLogits = draftResult.logits[0..., -1, 0...]
            draftLogits = draftProcessor?.process(logits: draftLogits) ?? draftLogits
            let draftToken = sampler.sample(logits: draftLogits)
            draftProcessor?.didSample(token: draftToken)
            asyncEval(draftToken)
            draftTokens.append(draftToken)
            draftY = .init(tokens: draftToken)
        }

        // Verification: main model processes proposals in one pass
        let verifyTokens = [y.tokens] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(verifyInput[text: .newAxis], cache: mainCache, state: mainState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let mainTokens: MLXArray
        if var verifyProcessor = processor {
            // Process each position sequentially so that the processor sees tokens sampled at earlier positions
            var sampled = [MLXArray]()
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            // Batch-sample all verify tokens from main model in one operation
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        // Compare and accept proposed tokens
        eval(mainTokens, draftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = concatenated(draftTokens).asArray(Int.self)
        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else {
                break
            }

            processor?.didSample(token: draftTokens[i])
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        // Update acceptance metrics
        draftAcceptedCount += accepted
        draftProposedCount += numDraft

        // Always emit the main model's token at position `accepted`
        // (either the correction token or the bonus token if all drafts matched)
        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])

        // Rewind trimmable caches for rejected tokens
        trimPromptCache(mainCache, numTokens: numDraft - accepted)
        trimPromptCache(draftCache, numTokens: Swift.max(numDraft - accepted - 1, 0))

        // Restore MambaCache layers on rejection, discard snapshots on full acceptance
        if accepted == numDraft {
            discardMambaSnapshots(mainCache)
            discardMambaSnapshots(draftCache)
        } else {
            restoreMambaCaches(mainCache)
            restoreMambaCaches(draftCache)
        }

        // Apply dynamic cache quantization after rewind
        quantizeKVCache(&mainCache)
        quantizeKVCache(&draftCache)

        // Set y/draftY for the next round
        y = .init(tokens: finalToken)
        draftY = .init(tokens: finalToken)

        // If all draft tokens were accepted, the draft model hasn't processed
        // the last accepted draft token yet. Feed it through to keep caches in sync.
        if accepted == numDraft {
            draftY = .init(
                tokens: concatenated([
                    draftTokens[numDraft - 1].reshaped([1]),
                    finalToken,
                ])
            )
        }
    }

    // MARK: - MambaCache Helpers

    /// Snapshot all MambaCache layers for potential restoration on rejection.
    private func snapshotMambaCaches(_ cache: [KVCache]) {
        for c in cache {
            if let mambaCache = c as? MambaCache {
                mambaCache.snapshot()
            }
        }
    }

    /// Restore all MambaCache layers from their snapshots (draft tokens rejected).
    private func restoreMambaCaches(_ cache: [KVCache]) {
        for c in cache {
            if let mambaCache = c as? MambaCache {
                mambaCache.restore()
            }
        }
    }

    /// Discard MambaCache snapshots without restoring (all draft tokens accepted).
    private func discardMambaSnapshots(_ cache: [KVCache]) {
        for c in cache {
            if let mambaCache = c as? MambaCache {
                mambaCache.discardSnapshot()
            }
        }
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        // Run a new speculation round
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

/// Result of a call to a deprecated callback-based generate function.
public struct GenerateResult {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokenIds: The array of generated token IDs.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    public init(
        inputText: LMInput.Text, tokenIds: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokenIds = tokenIds
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    @available(*, deprecated, renamed: "init(inputText:tokenIds:output:promptTime:generateTime:)")
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.init(
            inputText: inputText, tokenIds: tokens, output: output, promptTime: promptTime,
            generateTime: generateTime)
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    /// The token IDs of the input prompt.
    public var promptTokenIds: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    @available(*, deprecated, renamed: "promptTokenIds")
    public var promptTokens: [Int] { promptTokenIds }

    /// Generated token IDs
    public let tokenIds: [Int]

    @available(*, deprecated, renamed: "tokenIds")
    public var tokens: [Int] { tokenIds }

    /// Output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokenIds.count }

    /// Time to process the prompt (generate the first token)
    public let promptTime: TimeInterval

    /// Time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokenIds.count) / generateTime
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
    /// Keep producing tokens until an EOS token is produced
    case more

    /// Stop producing tokens, e.g. a token limit has been hit
    case stop
}

private struct SynchronousGenerationLoopResult {
    let generatedTokenIds: [Int]
    let promptTime: TimeInterval
    let generateTime: TimeInterval
    let promptPrefillTime: TimeInterval
    let stopReason: GenerateStopReason
}

private func buildStopTokenIds(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer
) -> Set<Int> {
    // Build complete EOS token set from all sources.
    var stopTokenIds = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIds.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    return stopTokenIds
}

private func runSynchronousGenerationLoop(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: TokenIterator,
    didGenerate: (_ token: Int, _ generatedTokenIds: [Int]) -> GenerateDisposition
) -> SynchronousGenerationLoopResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let stopTokenIds = buildStopTokenIds(
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )

    var generatedTokenIds = [Int]()
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
        if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            stopReason = .stop
            break
        }

        generatedTokenIds.append(token)

        if didGenerate(token, generatedTokenIds) == .stop {
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

    return SynchronousGenerationLoopResult(
        generatedTokenIds: generatedTokenIds,
        promptTime: promptTime,
        generateTime: generateTime,
        promptPrefillTime: iterator.promptPrefillTime,
        stopReason: stopReason ?? .cancelled
    )
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:cache:parameters:context:wiredMemoryTicket:)`` returning `AsyncStream<Generation>` is the preferred call.
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
/// Prefer using ``generate(input:cache:parameters:context:wiredMemoryTicket:)`` returning `AsyncStream<Generation>` instead.
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
/// ``generate(input:cache:parameters:context:wiredMemoryTicket:)`` returning `AsyncStream<Generation>` is the preferred call.
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
        inputText: input.text, tokenIds: result.generatedTokenIds,
        output: context.tokenizer.decode(tokenIds: result.generatedTokenIds),
        promptTime: result.promptTime + result.promptPrefillTime,
        generateTime: result.generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:wiredMemoryTicket:)`` returning `AsyncStream<Generation>` instead.
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
/// ``generate(input:cache:parameters:context:wiredMemoryTicket:)`` returning `AsyncStream<Generation>` is the preferred call.
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

    // runSynchronousGenerationLoop already calls Stream().synchronize() so any
    // lazy logprobs captured during the loop are safe to flush here. The
    // capture box is a reference held by a class so the appends the loop made
    // (through its own mutable copy of the iterator) are visible to us.
    let perTokenData = iterator.finalizePerTokenData()

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: result.generatedTokenIds.count,
        promptTime: result.promptTime + result.promptPrefillTime,
        generationTime: result.generateTime,
        stopReason: result.stopReason,
        perTokenLogProbs: perTokenData?.logProbs,
        perTokenIds: perTokenData?.tokenIds,
        perTokenPhases: perTokenData?.phases,
        generationPerplexity: perTokenData?.generationPerplexity,
        thinkingPerplexity: perTokenData?.thinkingPerplexity
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
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:wiredMemoryTicket:)``
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

/// Generates text and tool calls asynchronously using speculative decoding with a draft model.
///
/// This function uses a smaller draft model to propose tokens that are verified in batch
/// by the main model, potentially accelerating generation. The resulting stream yields
/// decoded text chunks, tool calls, and completion information. It has the same output as the
/// non-speculative ``generate(input:cache:parameters:context:wiredMemoryTicket:)``.
///
/// Both models must share the same tokenizer.
///
/// ### Example Usage:
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let mainContext: ModelContext
/// let draftModel: LanguageModel
///
/// let lmInput = try mainContext.processor.prepare(input: input)
///
/// let stream = try generate(
///     input: lmInput, parameters: generateParameters,
///     context: mainContext, draftModel: draftModel)
///
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
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values.
/// - Throws: An error if the iterator initialization fails.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: context.tokenizer,
            format: context.configuration.toolCallFormat ?? .json
        )
    )
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

/// Generates raw token IDs asynchronously using speculative decoding with a draft model.
///
/// This is similar to `generate(input:cache:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)`,
/// but yields raw token IDs instead of decoded text/tool calls.
///
/// Both models must share the same tokenizer.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
/// - Throws: An error if the iterator initialization fails.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: RawTokenLoopHandler()
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
    iterator: consuming any TokenIteratorProtocol,
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
            let iterator = iterator.consume()
            var handler = handler.consume()

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            let stopTokenIds = buildStopTokenIds(
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer
            )

            for token in iterator {
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
                if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
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

            // Synchronize before flushing per-token data so the logprob
            // MLXArrays captured during the loop have all been evaluated.
            // The synchronize is required for asyncEval-driven safety
            // anyway; we simply moved it ahead of the info-event emit.
            Stream().synchronize()

            let perTokenData = iterator.finalizePerTokenData()

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iterator.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled,
                perTokenLogProbs: perTokenData?.logProbs,
                perTokenIds: perTokenData?.tokenIds,
                perTokenPhases: perTokenData?.phases,
                generationPerplexity: perTokenData?.generationPerplexity,
                thinkingPerplexity: perTokenData?.thinkingPerplexity
            )
            _ = continuation.yield(handler.infoEvent(info))

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

    /// Per-token log probabilities collected during generation (when collectPerTokenData is true).
    public var perTokenLogProbs: [Float]?

    /// Per-token IDs collected during generation (when collectPerTokenData is true).
    public var perTokenIds: [Int]?

    /// Per-token phase labels ("think" or "gen") collected during generation.
    public var perTokenPhases: [String]?

    /// Perplexity computed over the generation (non-thinking) phase.
    public var generationPerplexity: Float?

    /// Perplexity computed over the thinking phase.
    public var thinkingPerplexity: Float?

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
        perTokenLogProbs: [Float]? = nil,
        perTokenIds: [Int]? = nil,
        perTokenPhases: [String]? = nil,
        generationPerplexity: Float? = nil,
        thinkingPerplexity: Float? = nil
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
        self.stopReason = stopReason
        self.perTokenLogProbs = perTokenLogProbs
        self.perTokenIds = perTokenIds
        self.perTokenPhases = perTokenPhases
        self.generationPerplexity = generationPerplexity
        self.thinkingPerplexity = thinkingPerplexity
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
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
