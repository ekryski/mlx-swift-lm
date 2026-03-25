// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers

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
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

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

    /// Additional logit processors to apply during generation.
    /// These are chained after the built-in repetition and presence penalty processors.
    /// Use this to inject custom processors like EOS suppression.
    /// Note: LogitProcessor is not Sendable (uses MLXArray), but GenerateParameters
    /// is only mutated before generation starts, so this is safe.
    public nonisolated(unsafe) var additionalProcessors: [LogitProcessor]

    /// GPT-OSS reasoning effort: "low", "medium", or "high". nil for non-GPT-OSS or default.
    public var reasoningEffort: String?

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
        reasoningEffort: String? = nil
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
        self.additionalProcessors = additionalProcessors
        self.reasoningEffort = reasoningEffort
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
        let repetitionContext: RepetitionContext?
        if let repetitionPenalty, repetitionPenalty != 0, repetitionContextSize > 0 {
            repetitionContext = RepetitionContext(
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize
            )
        } else {
            repetitionContext = nil
        }

        let presenceContext: PresencePenaltyContext?
        if let presencePenalty, presencePenalty != 0, presenceContextSize > 0 {
            presenceContext = PresencePenaltyContext(
                presencePenalty: presencePenalty,
                presenceContextSize: presenceContextSize
            )
        } else {
            presenceContext = nil
        }

        let frequencyContext: FrequencyPenaltyContext?
        if let frequencyPenalty, frequencyPenalty != 0, frequencyContextSize > 0 {
            frequencyContext = FrequencyPenaltyContext(
                frequencyPenalty: frequencyPenalty,
                frequencyContextSize: frequencyContextSize
            )
        } else {
            frequencyContext = nil
        }

        let penaltyProcessor: PenaltyProcessor?
        if repetitionContext == nil && presenceContext == nil && frequencyContext == nil {
            penaltyProcessor = nil
        } else {
            penaltyProcessor = PenaltyProcessor(
                repetitionContext: repetitionContext,
                presenceContext: presenceContext,
                frequencyContext: frequencyContext
            )
        }

        // Combine penalty processor with additional processors
        var all: [LogitProcessor] = []
        if let penaltyProcessor {
            all.append(penaltyProcessor)
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
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: Float
    let topK: Int
    let minP: Float
    let randomState: MLXRandom.RandomState

    private static let negInf = Float(-1e9)

    public init(temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0) {
        self.temp = MLXArray(temperature)
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.randomState = MLXRandom.RandomState()
    }

    /// Apply top-p (nucleus) filtering in log-space.
    /// Keeps the smallest set of tokens whose cumulative probability exceeds topP.
    private func applyTopP(_ logits: MLXArray, sortedIndices: MLXArray) -> MLXArray {
        guard topP > 0 && topP < 1 else { return logits }
        let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(softmax(sortedLogits, axis: -1), axis: -1)
        // Mask tokens whose cumulative prob is below (1 - topP), except always keep the last (highest)
        let mask = cumulativeProbs .> (1 - MLXArray(topP))
        let filteredLogits = MLX.where(mask, sortedLogits, MLXArray(Self.negInf))
        // Scatter back to original positions
        return putAlong(broadcast(MLXArray(Self.negInf), to: logits.shape), sortedIndices, values: filteredLogits, axis: -1)
    }

    /// Apply min-p filtering: remove tokens below minP * max_probability.
    private func applyMinP(_ logits: MLXArray) -> MLXArray {
        guard minP > 0 else { return logits }
        let probs = softmax(logits, axis: -1)
        let maxProb = MLX.max(probs, axis: -1, keepDims: true)
        let mask = probs .>= (maxProb * MLXArray(minP))
        return MLX.where(mask, logits, MLXArray(Self.negInf))
    }

    /// Apply top-k filtering: keep only the k highest-probability tokens.
    private func applyTopK(_ logits: MLXArray) -> MLXArray {
        guard topK > 0 else { return logits }
        let vocabSize = logits.dim(-1)
        guard topK < vocabSize else { return logits }
        // Find the k-th largest value as threshold
        let kthValues = sorted(logits, axis: -1)[0..., vocabSize - topK]
        let threshold = expandedDimensions(kthValues, axis: -1)
        return MLX.where(logits .>= threshold, logits, MLXArray(Self.negInf))
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            // Python mlx-lm order: top_p -> min_p -> top_k
            let sortedIndices = argSort(logits, axis: -1)
            logits = applyTopP(logits, sortedIndices: sortedIndices)
            logits = applyMinP(logits)
            logits = applyTopK(logits)

            return categorical(logits * (1 / temp))
        }
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
        let n = prompt.dim(0)
        let promptTokens = prompt.asType(.int32)
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

/// Chains multiple `LogitProcessor`s, calling each in sequence.
public struct CompositeLogitProcessor: LogitProcessor {
    var processors: [LogitProcessor]

    public init(processors: [LogitProcessor]) {
        self.processors = processors
    }

    mutating public func prompt(_ prompt: MLXArray) {
        for i in processors.indices {
            processors[i].prompt(prompt)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        var result = logits
        for processor in processors {
            result = processor.process(logits: result)
        }
        return result
    }

    mutating public func didSample(token: MLXArray) {
        for i in processors.indices {
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
        // Suppress all EOS tokens by setting their logits to -infinity.
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

        self.promptPrefillTime = try measure {
            try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
        }
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
            asyncEval(y.tokens)

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

        processor?.didSample(token: y)

        return y
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

    return SynchronousGenerationLoopResult(
        generatedTokens: generatedTokens,
        promptTime: promptTime,
        generateTime: generateTime,
        promptPrefillTime: iterator.promptPrefillTime,
        stopReason: stopReason ?? .cancelled
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
        stopReason: result.stopReason
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
            let iterator = iterator.consume()
            var handler = handler.consume()

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            let stopTokenIDs = buildStopTokenIDs(
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
                stopReason: stopReason ?? .cancelled
            )
            _ = continuation.yield(handler.infoEvent(info))

            // Synchronize with the stream to ensure tasks are completed
            Stream().synchronize()

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
        stopReason: GenerateStopReason = .stop
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
        self.stopReason = stopReason
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
