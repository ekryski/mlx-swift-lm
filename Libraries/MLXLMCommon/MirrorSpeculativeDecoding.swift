// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Mirror Speculative Decoding iterator (spec 021 phase 1A scaffold)
//
// Mirror SD pairs a small ANE draft (Core ML) with a GPU target (MLX).
// The two compute units are physically independent so the cycles can run
// in parallel: ANE drafts K tokens while the GPU verifies the previous
// K-block. The structural difference vs. the existing
// ``SpeculativeTokenIterator`` (which uses an MLX draft):
//
//   1. The draft lives behind ``ANEDraftBackend`` — a protocol that
//      doesn't depend on MLX at all. The iterator never holds draft KV
//      state; the backend manages it.
//   2. The verify path is identical to the DFlash and SpeculativeToken
//      iterators — single-pass [y, draft_1...draft_K] forward, argmax
//      compare, accept-prefix walk, cache trim.
//
// **Phase 1A status:** this iterator runs the draft + verify *sequentially*
// — submit draft, wait, run verify, repeat. The "true" Mirror SD path
// (overlap draft and verify across the cycle) is Phase 2 and requires
// the actual Core ML backend + IPC primitives. Sequential is correct,
// just not throughput-optimal.

/// Generator of tokens with **Mirror Speculative Decoding** (ANE draft +
/// GPU target). Currently sequential per cycle; Phase 2 will pipeline
/// draft and verify across compute units.
///
/// Drop-in replacement for ``TokenIterator`` when the caller has an
/// ANE-eligible target (registered in ``ANEDraftRegistry``) and
/// `temperature == 0`. Output is byte-identical to greedy
/// ``TokenIterator`` decode at temperature 0 (the verifier compares
/// against the target's argmax — same contract as DFlash).
///
/// Construction throws if the configured draft vocabulary doesn't agree
/// with the target's tokenizer up to the
/// ``aneVocabDefaultMaxSizeDifference`` / ``aneVocabDefaultMaxDisagreements``
/// gate — Mirror SD relies on token-ID equivalence across frameworks.
public struct MirrorSpeculativeTokenIterator: TokenIteratorProtocol {

    // MARK: - State

    var y: LMInput.Text

    let target: any LanguageModel
    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var draftBackend: any ANEDraftBackend

    let sampler: LogitSampler
    var tokenCount = 0
    let maxTokens: Int?

    /// Full prefix of accepted tokens (prompt + everything emitted so far).
    /// Mirror SD's draft backend wants the full committed sequence to
    /// resync after partial-accept, since the ANE backend's KV state may
    /// not be trim-able the same way the MLX target's is. Keeping it on
    /// the iterator side lets the backend stay stateless across rejections
    /// if it wants to.
    private var committedTokens: [Int] = []

    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    // MARK: - Metrics

    public private(set) var promptPrefillTime: TimeInterval = 0.0

    public private(set) var mirrorAcceptedCount = 0
    public private(set) var mirrorProposedCount = 0
    public private(set) var mirrorCycleCount = 0

    public var mirrorAcceptanceRate: Double {
        guard mirrorProposedCount > 0 else { return 0 }
        return Double(mirrorAcceptedCount) / Double(mirrorProposedCount)
    }

    public var specDecodeProposed: Int { mirrorProposedCount }
    public var specDecodeAccepted: Int { mirrorAcceptedCount }

    public var kvCacheMemoryBytes: Int? {
        let main = mainCache.isEmpty ? 0 : mainCache.reduce(0) { $0 + $1.memoryBytes }
        let draft = draftBackend.draftCacheMemoryBytes ?? 0
        return main + draft
    }

    private static var debugTracing: Bool {
        ProcessInfo.processInfo.environment["MLX_MIRROR_DEBUG"] == "1"
    }

    // MARK: - Init

    /// Initialize a `MirrorSpeculativeTokenIterator` over a target + ANE
    /// draft backend pairing.
    ///
    /// - Parameter input: language model input.
    /// - Parameter target: the GPU target (MLX language model).
    /// - Parameter mainCache: optional pre-allocated target KV cache.
    /// - Parameter draftBackend: the ANE-side draft backend.
    /// - Parameter parameters: generation parameters. Must specify
    ///   `temperature == 0` for the greedy verify gate to be sound.
    /// - Parameter vocabReport: optional pre-computed vocab equivalence
    ///   report. When non-nil and `isCompatible == false`, init throws.
    ///   When nil, the gate is skipped — caller takes responsibility.
    public init(
        input: LMInput,
        target: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftBackend: any ANEDraftBackend,
        parameters: GenerateParameters,
        vocabReport: ANEVocabEquivalenceReport? = nil
    ) throws {
        if let report = vocabReport, !report.isCompatible {
            throw KVCacheError(
                message: "Mirror SD vocabulary equivalence gate failed: "
                    + "size diff = \(report.sizeDifference), "
                    + "disagreements = \(report.disagreementCount) "
                    + "(\(String(format: "%.2f%%", report.disagreementRate * 100))). "
                    + "Mirror SD requires the draft and target tokenizers "
                    + "to agree token-for-token; pair a different draft.")
        }

        self.y = input.text
        self.target = target
        self.mainCache = mainCache ?? target.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache) else {
            throw KVCacheError(
                message: "Mirror SD requires a trimmable target KV cache. "
                    + "Hybrid GDN/Mamba targets need the tape-replay path "
                    + "(spec 020); Mirror SD's correctness is upstream of "
                    + "that fix.")
        }

        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens
        self.draftBackend = draftBackend

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        // Seed committedTokens with the prompt — backends that want full
        // history have it from the start.
        self.committedTokens = input.text.tokens.asArray(Int.self)

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }

        if Self.debugTracing {
            print("[MIRROR] iterator engaged: K=\(draftBackend.draftLength) "
                + "prompt=\(input.text.tokens.size)")
        }
    }

    /// Prefill the target + prime the pump. Same shape as the n-gram and
    /// DFlash iterators. The trailing `eval(token)` is the Gemma-4
    /// async-prefill commit barrier (see comment on
    /// ``NGramSpeculativeTokenIterator/prepare``).
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        switch try target.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            let result = target(tokens[text: .newAxis], cache: mainCache, state: nil)
            quantizeKVCache(&mainCache)
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            mainState = result.state
            eval(token)
            let tokenInt = token.item(Int.self)
            y = .init(tokens: token)
            pendingTokens.append(tokenInt)
            committedTokens.append(tokenInt)
        case .logits(let result):
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            eval(token)
            let tokenInt = token.item(Int.self)
            y = .init(tokens: token)
            mainState = result.state
            pendingTokens.append(tokenInt)
            committedTokens.append(tokenInt)
        }
    }

    // MARK: - Cycle

    /// One Mirror SD cycle: ANE draft → GPU verify → accept → trim.
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? draftBackend.draftLength
        guard remaining > 0 else { return }

        let lastToken = committedTokens.last ?? 0
        let draftInts = draftBackend.draftBlock(
            committedTokens: committedTokens,
            lastCommittedToken: lastToken)

        // Empty draft → AR fallback (single-step). Same shape as the
        // DFlash iterator; not pipelined yet.
        if draftInts.isEmpty {
            let result = target(y[text: .newAxis], cache: mainCache, state: mainState)
            quantizeKVCache(&mainCache)
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            eval(token)
            let tokenInt = token.item(Int.self)
            mainState = result.state
            pendingTokens.append(tokenInt)
            committedTokens.append(tokenInt)
            y = .init(tokens: token)
            mirrorCycleCount += 1
            if Self.debugTracing {
                print("[MIRROR] cycle=\(mirrorCycleCount) draft=0 ar=\(tokenInt)")
            }
            return
        }

        // Clamp to (remaining - 1) so we always leave a slot for the
        // bonus token. Same logic as the DFlash iterator.
        let budget = Swift.max(0, remaining - 1)
        let clampedDraft = budget < draftInts.count
            ? Array(draftInts.prefix(budget))
            : draftInts
        let numDraft = clampedDraft.count

        if numDraft == 0 {
            let result = target(y[text: .newAxis], cache: mainCache, state: mainState)
            quantizeKVCache(&mainCache)
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            eval(token)
            let tokenInt = token.item(Int.self)
            mainState = result.state
            pendingTokens.append(tokenInt)
            committedTokens.append(tokenInt)
            y = .init(tokens: token)
            mirrorCycleCount += 1
            return
        }

        let draftArray = MLXArray(clampedDraft.map { Int32($0) })
        let verifyTokens = concatenated([y.tokens, draftArray])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = target(verifyInput[text: .newAxis], cache: mainCache, state: mainState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
        let mainTokens = sampler.sample(logits: verifyLogits)
        eval(mainTokens)
        let mainTokensList = mainTokens.asArray(Int.self)

        // Reuse the DFlash accept-prefix helper — same pure-Swift logic;
        // no need for two copies. The contract is identical: longest
        // matching prefix between draft and target argmax.
        let accepted = dflashAcceptedPrefixLength(
            draft: clampedDraft,
            targetArgmax: mainTokensList)

        for i in 0 ..< accepted {
            pendingTokens.append(mainTokensList[i])
            committedTokens.append(mainTokensList[i])
        }
        let bonus = mainTokensList[accepted]
        pendingTokens.append(bonus)
        committedTokens.append(bonus)

        // Cache trim for rejected tokens.
        let rejected = numDraft - accepted
        if rejected > 0 {
            trimPromptCache(mainCache, numTokens: rejected)
        }
        quantizeKVCache(&mainCache)

        mirrorAcceptedCount += accepted
        mirrorProposedCount += numDraft
        mirrorCycleCount += 1

        y = .init(tokens: mainTokens[accepted ... accepted])

        if Self.debugTracing {
            print("[MIRROR] cycle=\(mirrorCycleCount) draft=\(numDraft) "
                + "accepted=\(accepted) rejected=\(rejected) bonus=\(bonus)")
        }
    }

    // MARK: - IteratorProtocol

    public mutating func next() -> Int? {
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
