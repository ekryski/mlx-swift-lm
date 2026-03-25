// Copyright 2026 Eric Kryski. All rights reserved.

import Foundation
import MLX
import MLXNN
import Tokenizers

/// Token iterator that implements speculative decoding for faster inference.
///
/// Speculative decoding uses a small "draft" model to generate K candidate tokens
/// autoregressively, then the larger "target" model verifies all K+1 tokens in a
/// single forward pass. Matching tokens are accepted; on first mismatch, the
/// target model's token is used instead. This gives near-draft-model speed with
/// target-model quality.
///
/// Both models must share the same tokenizer and have trimmable KV caches.
///
/// Usage:
/// ```swift
/// let iterator = try SpeculativeTokenIterator(
///     input: lmInput,
///     targetModel: targetModel,
///     draftModel: draftModel,
///     targetCache: nil,
///     draftCache: nil,
///     parameters: generateParameters,
///     numDraftTokens: 4
/// )
///
/// for token in iterator {
///     // token is an Int (token ID)
/// }
/// ```
///
/// Reference: Python mlx-lm `speculative_generate_step()` in generate.py
public struct SpeculativeTokenIterator: Sequence, IteratorProtocol {

    let targetModel: any LanguageModel
    let draftModel: any LanguageModel

    var targetCache: [KVCache]
    var draftCache: [KVCache]

    var processor: LogitProcessor?
    let sampler: LogitSampler

    /// Number of draft tokens to generate per speculation round.
    let numDraftTokens: Int

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

    /// Tokens accepted from draft model (for acceptance rate tracking).
    public private(set) var draftAcceptedCount = 0

    /// Total draft tokens proposed (for acceptance rate tracking).
    public private(set) var draftProposedCount = 0

    /// Buffer of accepted tokens waiting to be yielded.
    /// Speculative decoding can accept multiple tokens per round,
    /// but IteratorProtocol yields one at a time.
    private var pendingTokens: [Int] = []

    /// The next token to feed into the draft model for the next round.
    /// After verification, this is the target model's sampled token at
    /// the rejection point (or the token after the last accepted draft token).
    private var nextY: LMInput.Text

    /// When all draft tokens are accepted, the last draft token hasn't been
    /// processed by the draft model's cache yet, so we must prepend it
    /// to the next draft input.
    private var draftNeedsExtraToken: Bool = false
    private var lastDraftToken: MLXArray?

    /// Initialize a speculative token iterator.
    ///
    /// - Parameters:
    ///   - input: language model input (prompt)
    ///   - targetModel: the large target model for verification
    ///   - draftModel: the small draft model for candidate generation
    ///   - targetCache: optional pre-created KV cache for target model
    ///   - draftCache: optional pre-created KV cache for draft model
    ///   - parameters: generation parameters (shared between both models)
    ///   - numDraftTokens: number of draft tokens per speculation round (default: 4)
    public init(
        input: LMInput,
        targetModel: any LanguageModel,
        draftModel: any LanguageModel,
        targetCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int = 4
    ) throws {
        self.targetModel = targetModel
        self.draftModel = draftModel
        self.targetCache = targetCache ?? targetModel.newCache(parameters: parameters)
        self.draftCache = draftCache ?? draftModel.newCache(parameters: parameters)
        self.numDraftTokens = numDraftTokens

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        // Temporary placeholder; will be set during prepare()
        self.nextY = input.text

        // Validate that both caches are trimmable
        Self.validateCachesAreTrimmable(self.targetCache, label: "target")
        Self.validateCachesAreTrimmable(self.draftCache, label: "draft")

        self.promptPrefillTime = try measure {
            try prepareBothModels(
                input: input, windowSize: parameters.prefillStepSize
            )
        }
    }

    // MARK: - Prefill

    /// Prefill both models with the prompt.
    ///
    /// Both models must process the same prompt to align their caches.
    /// After prefill, the first token is sampled from the target model.
    private mutating func prepareBothModels(
        input: LMInput, windowSize: Int?
    ) throws {
        processor?.prompt(input.text.tokens)

        // Prefill draft model
        switch try draftModel.prepare(input, cache: draftCache, windowSize: windowSize) {
        case .tokens(let tokens):
            // Run the remaining tokens through the draft model to prime its cache
            let _ = draftModel(
                tokens[text: .newAxis], cache: draftCache.isEmpty ? nil : draftCache, state: nil
            )
        case .logits:
            // Draft model already processed everything during prepare
            break
        }

        // Prefill target model and get first token
        switch try targetModel.prepare(input, cache: targetCache, windowSize: windowSize) {
        case .tokens(let tokens):
            let result = targetModel(
                tokens[text: .newAxis],
                cache: targetCache.isEmpty ? nil : targetCache,
                state: nil
            )
            let token = convertToToken(logits: result.logits)
            nextY = .init(tokens: token)
            asyncEval(nextY.tokens)

        case .logits(let result):
            let token = convertToToken(logits: result.logits)
            nextY = .init(tokens: token)
            asyncEval(nextY.tokens)
        }
    }

    // MARK: - Core Speculative Loop

    /// Run one speculation round: draft K tokens, verify with target, accept/reject.
    ///
    /// Returns an array of accepted token IDs (may include the corrected token).
    private mutating func speculateRound() -> [Int] {
        let effectiveDraftCount: Int
        if let maxTokens {
            effectiveDraftCount = Swift.min(maxTokens - tokenCount, numDraftTokens)
        } else {
            effectiveDraftCount = numDraftTokens
        }

        // Step 1: Generate K draft tokens autoregressively
        let draftTokens = generateDraftTokens(count: effectiveDraftCount)

        // Step 2: Build the verification input: [nextY, draft_0, draft_1, ..., draft_{K-1}]
        let verificationInput: MLXArray
        if draftTokens.dim(0) == 0 {
            verificationInput = nextY.tokens.reshaped(-1)
        } else {
            verificationInput = concatenated(
                [nextY.tokens.reshaped(-1), draftTokens],
                axis: 0
            )
        }

        // Step 3: Run target model on all K+1 tokens in a single forward pass
        let targetResult = targetModel(
            LMInput.Text(tokens: verificationInput)[text: .newAxis],
            cache: targetCache.isEmpty ? nil : targetCache,
            state: nil
        )

        // Apply cache quantization to target
        maybeQuantizeKVCache(
            cache: &targetCache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )

        // Step 4: Sample from each position's logits
        // targetResult.logits shape: [1, K+1, vocab_size]
        // Position i gives logits for what should follow token i in the input
        let allLogits = targetResult.logits.squeezed(axis: 0)  // [K+1, vocab_size]

        var targetTokens: [Int] = []
        for i in 0 ..< (effectiveDraftCount + 1) {
            var logits = allLogits[i]
            logits = processor?.process(logits: logits) ?? logits
            let sampled = sampler.sample(logits: logits)
            processor?.didSample(token: sampled)
            targetTokens.append(sampled.item(Int.self))
        }

        // Step 5: Compare draft tokens with target tokens
        // targetTokens[i] is what the target thinks should follow the i-th input token
        // draftTokens[i] is what the draft proposed as the (i+1)-th token
        let draftTokenList: [Int]
        if draftTokens.size > 0 {
            eval(draftTokens)
            draftTokenList = (0 ..< draftTokens.dim(0)).map { draftTokens[$0].item(Int.self) }
        } else {
            draftTokenList = []
        }

        var acceptedCount = 0
        var accepted: [Int] = []

        for i in 0 ..< effectiveDraftCount {
            if targetTokens[i] == draftTokenList[i] {
                acceptedCount += 1
                accepted.append(targetTokens[i])
            } else {
                break
            }
        }

        // The target model's token at the rejection point (or after all accepted)
        accepted.append(targetTokens[acceptedCount])

        // Update metrics
        draftAcceptedCount += acceptedCount
        draftProposedCount += effectiveDraftCount

        // Step 6: Rewind caches to the correct position
        // Target cache consumed K+1 tokens but we only accepted `acceptedCount + 1`
        // So we need to trim (K - acceptedCount) tokens from target cache
        let targetTrim = effectiveDraftCount - acceptedCount
        if targetTrim > 0 {
            trimCache(targetCache, count: targetTrim)
        }

        // Draft cache consumed K tokens but we only accepted `acceptedCount`
        // Draft needs to be trimmed back. Also, we need to account for the
        // corrected/next token that will be the start of the next round.
        // Draft trim: max(K - acceptedCount - 1, 0)
        // But if acceptedCount == K (all accepted), the last draft token
        // hasn't been "verified" by the draft model, so in the next round
        // we feed [lastDraftToken, correctedToken] to draft.
        let draftTrim = Swift.max(effectiveDraftCount - acceptedCount - 1, 0)
        if draftTrim > 0 {
            trimCache(draftCache, count: draftTrim)
        }

        // Set up for next round
        let correctedToken = MLXArray(Int32(targetTokens[acceptedCount]))
        nextY = .init(tokens: correctedToken)

        if acceptedCount == effectiveDraftCount && !draftTokenList.isEmpty {
            // All draft tokens accepted: the last draft token hasn't been
            // processed by the draft cache yet, so prepend it next round
            draftNeedsExtraToken = true
            lastDraftToken = MLXArray(Int32(draftTokenList.last!))
        } else {
            draftNeedsExtraToken = false
            lastDraftToken = nil
        }

        return accepted
    }

    /// Generate K draft tokens autoregressively using the draft model.
    private mutating func generateDraftTokens(count: Int) -> MLXArray {
        guard count > 0 else {
            return MLXArray.zeros([0], type: Int32.self)
        }

        var draftInput: LMInput.Text
        if draftNeedsExtraToken, let extraToken = lastDraftToken {
            // Feed [lastDraftToken, nextY] to draft model
            let combined = concatenated(
                [extraToken.reshaped(1), nextY.tokens.reshaped(1)],
                axis: 0
            )
            draftInput = .init(tokens: combined)
            draftNeedsExtraToken = false
            lastDraftToken = nil
        } else {
            draftInput = nextY
        }

        var draftTokens: [MLXArray] = []

        for _ in 0 ..< count {
            let result = draftModel(
                draftInput[text: .newAxis],
                cache: draftCache.isEmpty ? nil : draftCache,
                state: nil
            )

            // Apply cache quantization to draft
            maybeQuantizeKVCache(
                cache: &draftCache,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart
            )

            // Sample from draft model (greedy for simplicity)
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let y = sampler.sample(logits: logits)
            asyncEval(y)

            draftTokens.append(y)
            draftInput = .init(tokens: y)
        }

        if draftTokens.isEmpty {
            return MLXArray.zeros([0], type: Int32.self)
        }

        return concatenated(draftTokens, axis: 0)
    }

    // MARK: - IteratorProtocol

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Yield from pending buffer first
        if !pendingTokens.isEmpty {
            let token = pendingTokens.removeFirst()
            tokenCount += 1

            // Periodically clear GPU memory cache
            if tokenCount % 256 == 0 {
                MLX.Memory.clearCache()
            }

            return token
        }

        // Run a new speculation round
        let accepted = speculateRound()

        guard !accepted.isEmpty else {
            return nil
        }

        // Put all but the first into the pending buffer
        if accepted.count > 1 {
            pendingTokens = Array(accepted.dropFirst())
        }

        tokenCount += 1

        // Periodically clear GPU memory cache
        if tokenCount % 256 == 0 {
            MLX.Memory.clearCache()
        }

        return accepted[0]
    }

    // MARK: - Helpers

    /// Convert logits to a sampled token, applying processor and sampler.
    private mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let y = sampler.sample(logits: logits)
        processor?.didSample(token: y)
        return y
    }

    /// Trim all caches in an array by the given count.
    private func trimCache(_ cache: [KVCache], count: Int) {
        for c in cache {
            c.trim(count)
        }
    }

    /// Validate that all caches in the array are trimmable.
    private static func validateCachesAreTrimmable(_ cache: [KVCache], label: String) {
        for (i, c) in cache.enumerated() {
            if !c.isTrimmable {
                print(
                    "Warning: \(label) cache layer \(i) is not trimmable. "
                    + "Speculative decoding requires trimmable caches."
                )
            }
        }
    }

    /// The acceptance rate of draft tokens (0.0 to 1.0).
    public var acceptanceRate: Double {
        guard draftProposedCount > 0 else { return 0 }
        return Double(draftAcceptedCount) / Double(draftProposedCount)
    }
}

// MARK: - Time measurement (mirrors Evaluate.swift)

private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    let end = Date.timeIntervalSinceReferenceDate
    return end - start
}
