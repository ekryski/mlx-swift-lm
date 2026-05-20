// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - DFlash speculative decoding iterator (phase 1 — full-attention only)
//
// State machine per cycle:
//   1. Ask the draft backend for K candidate tokens, given the last
//      committed token and the most recent target capture.
//   2. Build the verify-batch input  `[y, draft_1, ..., draft_K]`.
//   3. Run the target's ``DFlashTargetModel/dflashForwardWithCapture`` on
//      that batch — returns logits `[1, K+1, V]` + captured hidden states
//      `[layerID -> [1, K+1, H]]` at the requested layers.
//   4. Compute target argmax at each verify position — the K+1 "main
//      tokens".
//   5. Walk drafts in order; stop at the first mismatch with main tokens.
//      That's the accept count.
//   6. Emit the accepted prefix + main tokens[accept] (the bonus token —
//      either the correction after the first reject, or the K+1th token
//      after a full accept).
//   7. Trim the KV cache by (K - accept) for full-attention layers — those
//      drafted-K/V rows must be undone so the cache offset matches what
//      the iterator thinks it has.
//   8. Slice the captured hidden state at position `accept` and store it
//      for the next cycle's draft backend.
//
// Phase 1 supports **full-attention targets only** (`dflashIsHybridGDN ==
// false`). Hybrid GDN/Mamba targets need the tape-replay rollback path
// (spec 020); init throws for those targets so the caller can choose
// either to await Phase 3 or to fall back to a different iterator.

/// Generator of tokens with **DFlash block-diffusion speculative decoding**.
///
/// Pairs a small block-diffusion draft model (the ``DFlashDraftBackend``)
/// with a target language model that conforms to ``DFlashTargetModel``.
/// The draft conditions on captured target hidden states; the target
/// verifies the K-token candidate block in one forward pass and accepts
/// the longest matching prefix.
///
/// Drop-in replacement for ``TokenIterator`` when the caller has a
/// DFlash-conforming target and a registered draft backend. The cycle
/// produces byte-identical output to greedy ``TokenIterator`` decode at
/// `temperature: 0` (the verifier compares against the target's argmax).
///
/// Construction throws if the target reports `dflashIsHybridGDN == true`
/// — those need the tape-replay rollback path (spec 020) which lands in
/// Phase 3. Callers should fall back to ``TokenIterator`` or a different
/// speculative path on that error.
public struct DFlashSpeculativeTokenIterator: TokenIteratorProtocol {

    // MARK: - Inputs / state

    var y: LMInput.Text

    let target: any LanguageModel
    let dflashTarget: any DFlashTargetModel
    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    /// The draft backend — held as a typed value so its mutating-method
    /// state (KV cache, scripted cursor, etc.) survives across cycles.
    /// We use `any DFlashDraftBackend` and accept the existential
    /// dispatch cost: at K=16 the per-cycle backend call is one of the
    /// cheap pieces.
    var draftBackend: any DFlashDraftBackend

    /// Captured target hidden states from the most recent verify forward,
    /// sliced to the accepted-prefix's last position. Empty on the first
    /// cycle (the draft backend handles cold-start). Keyed by layerID.
    var lastCapturedHidden: [Int: MLXArray] = [:]

    let sampler: LogitSampler
    var tokenCount = 0
    let maxTokens: Int?

    // Per-round emission buffer
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    /// Prompt prefill time (ms), measured once at init.
    public private(set) var promptPrefillTime: TimeInterval = 0.0

    /// Tokens accepted from the draft backend (for acceptance-rate tracking).
    public private(set) var dflashAcceptedCount = 0

    /// Total draft tokens proposed across all cycles.
    public private(set) var dflashProposedCount = 0

    /// Cycle count — number of completed `speculateRound()` invocations.
    /// Used by the bench harness to compute mean accept-per-cycle, which
    /// cross-checks against the dflash-mlx published numbers.
    public private(set) var dflashCycleCount = 0

    public var dflashAcceptanceRate: Double {
        guard dflashProposedCount > 0 else { return 0 }
        return Double(dflashAcceptedCount) / Double(dflashProposedCount)
    }

    public var specDecodeProposed: Int { dflashProposedCount }
    public var specDecodeAccepted: Int { dflashAcceptedCount }

    public var kvCacheMemoryBytes: Int? {
        mainCache.isEmpty ? nil : mainCache.reduce(0) { $0 + $1.memoryBytes }
    }

    /// Verbose tracing (`MLX_DFLASH_DEBUG=1`). When enabled, every cycle
    /// logs draft/accept counts, cache offset, and capture-layer summary.
    private static var debugTracing: Bool {
        ProcessInfo.processInfo.environment["MLX_DFLASH_DEBUG"] == "1"
    }

    // MARK: - Init

    /// Initialize a `DFlashSpeculativeTokenIterator` over a target that
    /// conforms to both ``LanguageModel`` (for prefill / sampler glue)
    /// and ``DFlashTargetModel`` (for capture forward).
    ///
    /// The `draftBackend` parameter is consumed by value but stored as
    /// existential because Swift's `any P` is the only way to hold a
    /// mutating-method protocol value across cycles without losing
    /// type erasure benefits we need for testing.
    ///
    /// - Throws: ``KVCacheError`` if the target's cache is non-trimmable
    ///   (i.e. contains hybrid GDN/Mamba layers and `dflashIsHybridGDN`
    ///   is true). Phase 3 will lift this restriction via tape-replay.
    public init(
        input: LMInput,
        target: any LanguageModel & DFlashTargetModel,
        mainCache: [KVCache]? = nil,
        draftBackend: any DFlashDraftBackend,
        parameters: GenerateParameters
    ) throws {
        self.y = input.text
        self.target = target
        self.dflashTarget = target

        if target.dflashIsHybridGDN {
            throw KVCacheError(
                message: "DFlashSpeculativeTokenIterator phase 1 supports "
                    + "full-attention targets only. Hybrid GDN/Mamba "
                    + "targets need the tape-replay rollback path "
                    + "(spec 020 / phase 3). Got "
                    + "\(type(of: target)) with dflashIsHybridGDN == true.")
        }

        self.mainCache = mainCache ?? target.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache) else {
            throw KVCacheError(
                message: "DFlash speculative decoding (phase 1) requires "
                    + "trimmable KV caches. The target reported a hybrid "
                    + "or otherwise non-trimmable cache.")
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

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }

        if Self.debugTracing {
            print("[DFLASH] iterator engaged: blockSize=\(draftBackend.blockSize) "
                + "captureLayers=\(draftBackend.captureLayerIDs.sorted()) "
                + "promptTokens=\(input.text.tokens.size)")
        }
    }

    /// Prefill the target with the prompt + prime the pump with one decode
    /// step. Mirrors ``NGramSpeculativeTokenIterator/prepare`` — the first
    /// emitted token must be the target's argmax at the last prompt
    /// position. Without this step, the first `speculateRound` would
    /// conflate "prefill's last token" with "the first generated token"
    /// and produce a stream offset by one vs. ``TokenIterator``.
    ///
    /// As with the n-gram iterator, the trailing `eval(token)` is also a
    /// **sync barrier** required by Gemma 4 — without it, async-prefill
    /// KV writes may not have committed by the time the first verify
    /// forward in `speculateRound()` reads the cache.
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
        case .logits(let result):
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            eval(token)
            let tokenInt = token.item(Int.self)
            y = .init(tokens: token)
            mainState = result.state
            pendingTokens.append(tokenInt)
        }
    }

    // MARK: - Cycle

    /// One DFlash cycle: draft → verify → accept → trim.
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? draftBackend.blockSize
        guard remaining > 0 else { return }

        let draftInts = draftBackend.draftBlock(
            targetHidden: lastCapturedHidden,
            lastCommittedToken: y.tokens.asArray(Int.self).last ?? 0)

        // Empty draft → fall through to single-step AR decode.
        // Same shape as the n-gram iterator's miss path, but we keep it
        // single-step rather than batched (the AR-batch size choice is
        // workload-dependent and DFlash phase 1 is correctness-first).
        if draftInts.isEmpty {
            let result = target(y[text: .newAxis], cache: mainCache, state: mainState)
            quantizeKVCache(&mainCache)
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            eval(token)
            let tokenInt = token.item(Int.self)
            mainState = result.state
            pendingTokens.append(tokenInt)
            y = .init(tokens: token)
            dflashCycleCount += 1
            if Self.debugTracing {
                print("[DFLASH] cycle=\(dflashCycleCount) draft=0 (empty) "
                    + "ar_emit=\(tokenInt)")
            }
            return
        }

        // Clamp the draft to remaining-token budget. We need to keep at
        // least one slot for the bonus token, so cap drafts at
        // (remaining - 1) — when remaining == 1, we degenerate to no-draft
        // AR (handled above by the empty-draft branch logically, but here
        // the draft might still be non-empty; we just truncate).
        let budget = Swift.max(0, remaining - 1)
        let clampedDraft = budget < draftInts.count
            ? Array(draftInts.prefix(budget))
            : draftInts
        let numDraft = clampedDraft.count

        if numDraft == 0 {
            // Same path as empty draft — degenerate.
            let result = target(y[text: .newAxis], cache: mainCache, state: mainState)
            quantizeKVCache(&mainCache)
            let logits = result.logits[0..., -1, 0...]
            let token = sampler.sample(logits: logits)
            eval(token)
            let tokenInt = token.item(Int.self)
            mainState = result.state
            pendingTokens.append(tokenInt)
            y = .init(tokens: token)
            dflashCycleCount += 1
            return
        }

        let draftArray = MLXArray(clampedDraft.map { Int32($0) })

        // Verify-batch input: [y, draft_1, ..., draft_K]
        let verifyTokens = concatenated([y.tokens, draftArray])
        let verifyInputIDs = verifyTokens[.newAxis, 0...]  // [1, K+1]
        let verifyStart = verifyTokens.dim(0) - (numDraft + 1)

        // Capture hidden states at the configured layers via the DFlash
        // protocol method. Empty `captureLayerIDs` is fine — the target
        // implementation should return an empty `captured` dict in that
        // case.
        let (verifyLogitsFull, captured) = dflashTarget.dflashForwardWithCapture(
            inputIDs: verifyInputIDs,
            cache: mainCache,
            captureLayerIDs: draftBackend.captureLayerIDs)

        // Argmax per verify position. Greedy = `sampler.sample` at temp=0.
        let verifyLogits = verifyLogitsFull[0..., verifyStart..., 0...].squeezed(axis: 0)
        let mainTokens = sampler.sample(logits: verifyLogits)
        eval(mainTokens)
        let mainTokensList = mainTokens.asArray(Int.self)

        // Accept-prefix walk — pure-Swift helper so this slice is unit-tested.
        let accepted = dflashAcceptedPrefixLength(
            draft: clampedDraft,
            targetArgmax: mainTokensList)

        // Emit accepted prefix + bonus token.
        for i in 0 ..< accepted {
            pendingTokens.append(mainTokensList[i])
        }
        let bonusToken = mainTokensList[accepted]
        pendingTokens.append(bonusToken)

        // Trim the KV cache for rejected tokens — same as the linear-K
        // speculative path. The verify forward wrote K+1 positions; we
        // commit `accepted + 1` of them (accepted prefix + bonus). The
        // remaining `numDraft - accepted` rows must be undone.
        let rejected = numDraft - accepted
        if rejected > 0 {
            trimPromptCache(mainCache, numTokens: rejected)
        }
        quantizeKVCache(&mainCache)

        // Slice the captured hidden state at position `accepted` so the
        // next cycle's draft conditions on the hidden state at the actual
        // committed-prefix tail (not somewhere mid-rejected-draft). Phase
        // 2's real backend will use this; the stub backends ignore it.
        if !captured.isEmpty {
            var sliced: [Int: MLXArray] = [:]
            sliced.reserveCapacity(captured.count)
            for (layerID, h) in captured {
                // h shape: [1, K+1, H]. Take position `accepted` -> [1, 1, H].
                sliced[layerID] = h[0..., accepted ..< (accepted + 1), 0...]
            }
            lastCapturedHidden = sliced
        } else {
            lastCapturedHidden = [:]
        }

        // Bookkeeping.
        dflashAcceptedCount += accepted
        dflashProposedCount += numDraft
        dflashCycleCount += 1

        // Next round seeds from the bonus token.
        y = .init(tokens: mainTokens[accepted ... accepted])

        if Self.debugTracing {
            print("[DFLASH] cycle=\(dflashCycleCount) draft=\(numDraft) "
                + "accepted=\(accepted) rejected=\(rejected) bonus=\(bonusToken)")
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
