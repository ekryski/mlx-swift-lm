// Copyright © 2026 ekryski.
//
// Multi-token prediction (MTP) self-speculative decoding — variant A iterator.
//
// Spec: specs/030-multi-token-prediction.md
// Reference: github.com/youssofal/MTPLX (Python MLX, canonical algorithm).
//
// Cycle (greedy mode, temp == 0):
//
//   1. AR forward on the last committed token. Returns (logits, hidden);
//      sample bonus = argmax(logits). Trunk cache advances by 1.
//   2. MTP draft: head 0 takes (hidden, bonus); produces (hidden_0,
//      logits_0); proposal_0 = argmax(logits_0). Head i+1 takes
//      (hidden_i, proposal_i); produces (hidden_{i+1}, logits_{i+1}).
//      After k heads: proposals[0..<k]. Per-head MTP cache advances by 1
//      per head.
//   3. Verify forward on [bonus, proposal_0, ..., proposal_{k-1}] (length
//      k+1). Trunk cache advances by k+1. Returns logits at each
//      position; verify_logits[i] predicts the token at trunk position
//      `bonus + i + 1` (i.e. one past proposal_i). For acceptance we
//      compare verify_argmax[i] against proposal_i.
//   4. Accept m: verify_argmax[i] == proposal_i for i < m, mismatch at
//      m. Commit: bonus + proposal_0..proposal_{m-1} (1 + m tokens).
//      The model's argmax at verify position m (the rejected position)
//      becomes the next cycle's AR input — call it `fallback`.
//      Special case m == k: all accepted; verify_argmax[k] is the next
//      AR input.
//   5. Trim trunk cache by (k - m) — i.e. drop the rejected proposals.
//      Trim per-head MTP caches by (k - m) — they advanced once per
//      head past the accepted prefix. (Per-head advance is 1 per head,
//      but only the first m heads' state is "valid" — the heads
//      m..k-1 wrote state that's downstream of the rejected proposal,
//      which is now invalid.)
//
// Hybrid GDN models (Qwen 3.5 / 3.6 / Qwen3-Next) require spec 020
// tape-replay rollback for the trunk cache trim. This iterator declines
// to engage on hybrid caches via `canTrimPromptCache(...)` until that
// lands.

import Foundation
import MLX

// MARK: - Route decision

/// Outcome of the MTP-route decision: whether to engage
/// `MTPSelfSpeculativeTokenIterator` and (if env-var-driven) the patched
/// parameters that should be passed to it.
public struct MTPRouteDecision: Sendable {
    /// Engage `MTPSelfSpeculativeTokenIterator` (cache-trimmability
    /// willing).
    public let shouldEngage: Bool
    /// Parameters the iterator should be constructed with.
    public let parameters: GenerateParameters
}

/// MTP env-default knobs (applied when `MLX_MTP_ENABLED=1` is set
/// without explicit Swift parameters).
///
/// `mtpEnvDefaultDraftCount = 1` matches DeepSeek-V3/V4's published k=1
/// MTP head; per-family overrides should set this to 2 or higher only
/// when the bundle ships `>1` heads.
public let mtpEnvDefaultDraftCount: Int = 1

/// Decide whether `MLXLMCommon.generate(...)` should auto-route to the
/// MTP self-speculative iterator (variant A).
///
/// **Opt-in modes**:
///   1. **Swift parameters.** `parameters.mtpEnabled == true`.
///   2. **Env var.** `MLX_MTP_ENABLED=1`. Applies sensible defaults
///      (`mtpDraftCount = 1`); explicit Swift values still win.
///
/// **Disqualifiers** (any one declines the route):
///   - `temperature != 0` — phase 1 is greedy-only. Stochastic
///     accept/reject is spec 030 phase 3.
///
/// The model-conformance check (`is MTPInjector` + `isMTPLoaded`) and
/// the cache-trimmability check happen later, inside `generate(...)`,
/// after the model context is in hand. A non-conforming model falls
/// back to `TokenIterator` cleanly.
public func mtpRouteDecision(parameters: GenerateParameters) -> MTPRouteDecision {
    if parameters.temperature != 0 {
        return MTPRouteDecision(shouldEngage: false, parameters: parameters)
    }

    let optedInBySwift = parameters.mtpEnabled
    if optedInBySwift {
        var patched = parameters
        if patched.mtpDraftCount < 1 {
            patched.mtpDraftCount = mtpEnvDefaultDraftCount
        }
        return MTPRouteDecision(shouldEngage: true, parameters: patched)
    }

    let envEnabled =
        ProcessInfo.processInfo.environment["MLX_MTP_ENABLED"] == "1"
    if envEnabled {
        var patched = parameters
        patched.mtpEnabled = true
        if patched.mtpDraftCount < 1 {
            patched.mtpDraftCount = mtpEnvDefaultDraftCount
        }
        return MTPRouteDecision(shouldEngage: true, parameters: patched)
    }

    return MTPRouteDecision(shouldEngage: false, parameters: parameters)
}

// MARK: - Iterator

/// Self-speculative iterator backed by a target's in-trunk MTP heads.
///
/// One model serves both target and draft via the `MTPInjector`
/// protocol — the heads share the trunk's embedding + LM head and
/// predict tokens at offsets +1..+k from a captured trunk hidden state.
///
/// Phase 1 supports greedy mode only (`temperature == 0`). Stochastic
/// accept/reject is spec 030 phase 3.
///
/// Hybrid GDN trunks (Qwen 3.5 / 3.6 / Qwen3-Next) require spec 020
/// tape-replay rollback before they're safe with this iterator. The
/// init refuses to engage on non-trimmable caches.
public struct MTPSelfSpeculativeTokenIterator: TokenIteratorProtocol {

    /// Composite type — the model must be both a `LanguageModel`
    /// (for `prepare(...)`, `defaultPrefillStepSize`, etc.) and an
    /// `MTPInjector` (for the capture forward + MTP head forwards).
    public typealias MTPModel = LanguageModel & MTPInjector

    var y: LMInput.Text
    let model: any MTPModel
    var trunkCache: [KVCache]
    var headCaches: [KVCache]

    var processor: (any LogitProcessor)?
    let sampler: LogitSampler

    public var tokenCount = 0
    let maxTokens: Int?
    /// Number of MTP heads engaged per cycle. Constrained to
    /// `[1, model.mtpContract.maxHeads]` at init.
    let draftCount: Int

    // Per-cycle emission buffer.
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    public private(set) var promptPrefillTime: TimeInterval = 0.0

    /// MTP draft tokens accepted across the run.
    public private(set) var mtpAcceptedCount = 0
    /// MTP draft tokens proposed across the run.
    public private(set) var mtpProposedCount = 0

    public var acceptanceRate: Double {
        guard mtpProposedCount > 0 else { return 0 }
        return Double(mtpAcceptedCount) / Double(mtpProposedCount)
    }

    public var specDecodeProposed: Int { mtpProposedCount }
    public var specDecodeAccepted: Int { mtpAcceptedCount }

    public var kvCacheMemoryBytes: Int? {
        let trunkBytes = trunkCache.reduce(0) { $0 + $1.memoryBytes }
        let headBytes = headCaches.reduce(0) { $0 + $1.memoryBytes }
        return trunkBytes + headBytes
    }

    /// Initialize the iterator.
    ///
    /// - Throws: ``MTPIteratorError`` when the model is not MTP-loaded
    ///   or its trunk cache is not fully trimmable (hybrid GDN before
    ///   spec 020 lands).
    public init(
        input: LMInput,
        model: any MTPModel,
        trunkCache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        guard model.isMTPLoaded else {
            throw MTPIteratorError(
                message:
                    "MTP iterator requires a model with MTP heads loaded. "
                    + "Set GenerateParameters.mtpEnabled = true (or "
                    + "MLX_MTP_ENABLED=1) before model load to keep MTP "
                    + "weights through the sanitize step.")
        }
        guard parameters.temperature == 0 else {
            throw MTPIteratorError(
                message:
                    "MTP iterator is greedy-only in phase 1 (temperature "
                    + "== 0). Stochastic accept/reject is spec 030 phase 3.")
        }

        self.y = input.text
        self.model = model
        self.trunkCache = trunkCache ?? model.newCache(parameters: parameters)
        guard canTrimPromptCache(self.trunkCache) else {
            throw MTPIteratorError(
                message:
                    "MTP iterator requires a fully trimmable trunk KV "
                    + "cache. Hybrid GDN models block on spec 020 "
                    + "tape-replay rollback.")
        }

        self.headCaches = model.mtpHeadCaches(parameters: parameters)
        let availableHeads = self.headCaches.count
        precondition(
            availableHeads == model.mtpContract.maxHeads,
            "MTPInjector.mtpHeadCaches must return mtpContract.maxHeads entries"
        )
        let requested = Swift.max(1, parameters.mtpDraftCount)
        self.draftCount = Swift.min(requested, availableHeads)

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()
        self.maxTokens = parameters.maxTokens

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Prefill the trunk with the prompt. MTP head caches stay empty
    /// during prefill — the heads are only invoked after the trunk has
    /// produced a hidden state for the first decode position.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        let resolvedWindow = windowSize ?? model.defaultPrefillStepSize
        switch try model.prepare(
            input, cache: trunkCache, windowSize: resolvedWindow)
        {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
        }
    }

    /// One MTP cycle: AR + draft + verify + accept/trim.
    mutating func cycle() {
        let remainingBudget = maxTokens.map { $0 - tokenCount } ?? Int.max
        if remainingBudget <= 0 { return }

        // ---- 1. AR forward on the trailing token. -----------------------
        let arInput = y[text: .newAxis]  // [1, L]
        let (arLogits, arHidden) = model.runCaptureForward(
            inputs: arInput.tokens, cache: trunkCache)

        var bonusLogits = arLogits[0..., -1, 0...]  // [1, V]
        bonusLogits = processor?.process(logits: bonusLogits) ?? bonusLogits
        let bonusTokenArr = sampler.sample(logits: bonusLogits)  // [1]
        processor?.didSample(token: bonusTokenArr)
        let bonusToken = bonusTokenArr.item(Int.self)
        pendingTokens.append(bonusToken)

        if remainingBudget == 1 {
            // No room for spec-decode acceptance this cycle. Set up next
            // cycle's AR input from the bonus and exit.
            y = .init(tokens: bonusTokenArr)
            return
        }

        // Trim draftCount to the remaining budget so we don't propose
        // more tokens than the caller will consume.
        let k = Swift.min(draftCount, remainingBudget - 1)

        // ---- 2. MTP heads draft -----------------------------------------
        // Trunk hidden at the last AR position is the seed for head 0.
        // `[0..., -1, 0...]` collapses the L axis; expand it back to keep
        // the [B, 1, H] shape that the heads expect.
        let seedHidden = arHidden[0..., -1, 0...]
            .expandedDimensions(axis: 1)  // [1, 1, H]

        var prevHidden = seedHidden
        var prevTokenArr = bonusTokenArr.expandedDimensions(axis: 0)  // [1, 1]
        var draftProcessor = processor  // shadow — real processor only advances on accept
        var proposals = [Int]()
        var proposalArrays = [MLXArray]()
        proposals.reserveCapacity(k)
        proposalArrays.reserveCapacity(k)
        for headIdx in 0 ..< k {
            let out = model.runMTPHead(
                headIndex: headIdx,
                prevHidden: prevHidden,
                prevToken: prevTokenArr,
                headCache: headCaches[headIdx])

            var logits = out.logits[0..., -1, 0...]  // [1, V]
            logits = draftProcessor?.process(logits: logits) ?? logits
            let tokenArr = sampler.sample(logits: logits)
            draftProcessor?.didSample(token: tokenArr)
            asyncEval(tokenArr)
            let tok = tokenArr.item(Int.self)
            proposals.append(tok)
            proposalArrays.append(tokenArr)

            prevHidden = out.nextHidden
            prevTokenArr = tokenArr.expandedDimensions(axis: 0)
        }

        if k == 0 {
            y = .init(tokens: bonusTokenArr)
            return
        }

        // ---- 3. Verify forward ------------------------------------------
        // [bonus, proposal_0, ..., proposal_{k-1}] — total length k+1.
        let verifyTokens = concatenated(
            [bonusTokenArr] + proposalArrays.map { $0 })  // [k+1]
        let verifyInput = verifyTokens.expandedDimensions(axis: 0)  // [1, k+1]
        let (verifyLogits, _) = model.runCaptureForward(
            inputs: verifyInput, cache: trunkCache)
        // verifyLogits[0, i, :] = trunk's predicted next-token distribution
        // at position `i`. We compare verifyLogits[0, 0, :] argmax against
        // proposal_0, verifyLogits[0, 1, :] argmax against proposal_1, etc.
        // verifyLogits[0, k, :] is the token AFTER the last proposal — the
        // "free" fall-through that becomes the next cycle's AR input.

        let verifyStart = 0  // verify input was a fresh batch of k+1
        var fallbackTokenArr = MLXArray(0)  // placeholder, set below
        var accepted = 0
        if var verifyProcessor = processor {
            // Sequential sampling so the processor sees prior tokens.
            for i in 0 ... k {
                var logits = verifyLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                if i < k {
                    let tokInt = token.item(Int.self)
                    if tokInt == proposals[i] {
                        accepted += 1
                    } else {
                        fallbackTokenArr = token
                        break
                    }
                } else {
                    // All k accepted — position k is the fallthrough.
                    fallbackTokenArr = token
                }
            }
        } else {
            // Batched argmax.
            let batched = verifyLogits[0..., verifyStart ... verifyStart + k, 0...]
                .squeezed(axis: 0)  // [k+1, V]
            let argmax = sampler.sample(logits: batched)  // [k+1]
            eval(argmax)
            let argmaxList = argmax.asArray(Int.self)
            for i in 0 ..< k {
                if argmaxList[i] == proposals[i] {
                    accepted += 1
                } else {
                    break
                }
            }
            // Index of the fallthrough token in the argmax buffer.
            let idx = accepted
            fallbackTokenArr = argmax[idx ... idx]
        }

        // Commit accepted proposals to pendingTokens (after the bonus
        // already buffered).
        for i in 0 ..< accepted {
            processor?.didSample(token: proposalArrays[i])
            pendingTokens.append(proposals[i])
        }

        mtpAcceptedCount += accepted
        mtpProposedCount += k

        // Commit fallback token (always exactly one) — becomes the
        // next cycle's AR input.
        if accepted < k {
            processor?.didSample(token: fallbackTokenArr)
        }
        // Trim trunk cache: verify advanced by k+1, but only `accepted+1`
        // positions are valid (bonus + accepted proposals). Drop the
        // rest.
        let trunkTrim = k - accepted
        if trunkTrim > 0 {
            trimPromptCache(trunkCache, numTokens: trunkTrim)
        }

        // Trim per-head MTP caches: heads 0..<accepted advanced their
        // state on a "valid" prior, so their cache entries are
        // legitimate. Heads accepted..<k advanced on a now-rejected
        // proposal — their cache entries should be dropped. Each head
        // advanced its cache by exactly 1 per call.
        for headIdx in accepted ..< k {
            _ = headCaches[headIdx].trim(1)
        }

        // Set y for next cycle: the fallback (rejected-position model
        // argmax, or full-acceptance fallthrough).
        y = .init(tokens: fallbackTokenArr)
        // Buffer the fallback so the iterator emits it before draining.
        // The fallback advances the cache by 1 only on the next AR
        // forward — at that point the trunk consumes it.
        // NOTE: do NOT append fallback to pendingTokens — it'll be
        // consumed by the next cycle's AR forward and emitted there.
    }

    public var tokenCountValue: Int { tokenCount }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        cycle()
        if pendingTokens.isEmpty {
            return nil
        }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

/// Init / runtime errors thrown by the MTP iterator.
public struct MTPIteratorError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { "MTPIteratorError: \(message)" }
}
