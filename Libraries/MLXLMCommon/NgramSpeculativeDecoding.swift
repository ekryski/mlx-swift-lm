// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - N-gram speculative decoding
//
// Prompt-lookup speculative decoding ("ngram speculative", "PLD"). Sources
// draft tokens from the prompt + already-generated tokens themselves rather
// than from a separate draft model — best for repetitive output (code,
// templates, factual re-quoting). See `Libraries/MLXLMCommon/Evaluate.swift`
// for the draft-model variant (`SpeculativeTokenIterator`).
//
// Public surface lives entirely in this file:
//   - `NGramSpeculativeTokenIterator` — drop-in replacement for
//     `TokenIterator` when `GenerateParameters.ngramSize > 0`.
//   - `NGramLookup` — internal multi-size hash table over the rolling token
//     suffix, supporting multi-size fallback and min-hits filtering.

/// Prompt-lookup speculative draft source.
///
/// Maintains one hash table per size in `[minNgramSize ... maxNgramSize]`.
/// On each speculation round, the lookup tries the longest size first and
/// falls back to shorter sizes on miss — longer matches are stricter priors,
/// so they're preferred when available.
///
/// Stays entirely on CPU (Swift dictionaries + arrays). Rolling: generated
/// tokens get added to the history so self-repetition during generation
/// produces hits too. With sizes 2–5 and a ~10k-token history the per-table
/// memory footprint is small (a few hundred KB total).
final class NGramLookup {
    /// Token history — prompt followed by accepted generated tokens.
    private var tokens: [Int]
    /// One table per `ngramSize`. Maps `size` → (FNV-1a hash → end positions).
    /// Collisions are handled on the verify side (a bad draft is just rejected
    /// — correctness preserved by the main-model argmax check).
    private var tables: [Int: [UInt64: [Int]]]
    let maxNgramSize: Int
    let minNgramSize: Int
    let minHits: Int

    init(promptTokens: [Int], maxNgramSize: Int, minNgramSize: Int, minHits: Int) {
        precondition(minNgramSize >= 1, "minNgramSize must be >= 1")
        precondition(
            maxNgramSize >= minNgramSize,
            "maxNgramSize (\(maxNgramSize)) must be >= minNgramSize (\(minNgramSize))")
        precondition(minHits >= 1, "minHits must be >= 1")
        self.tokens = promptTokens
        self.maxNgramSize = maxNgramSize
        self.minNgramSize = minNgramSize
        self.minHits = minHits
        self.tables = [:]
        rebuildAllTables()
    }

    /// FNV-1a 64-bit hash over the last `size` tokens ending at `endIdx`
    /// (inclusive). Rolling update is not worth it at these sizes.
    private func hashNgramEndingAt(_ endIdx: Int, size: Int) -> UInt64? {
        let start = endIdx - size + 1
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

    private func rebuildAllTables() {
        for k in minNgramSize ... maxNgramSize {
            var table: [UInt64: [Int]] = [:]
            if tokens.count >= k {
                for i in (k - 1) ..< tokens.count {
                    if let h = hashNgramEndingAt(i, size: k) {
                        table[h, default: []].append(i)
                    }
                }
            }
            tables[k] = table
        }
    }

    /// Append newly-accepted tokens to the history and extend each table so
    /// future lookups see the updated prefix.
    func extend(with newTokens: [Int]) {
        let startIdx = tokens.count
        tokens.append(contentsOf: newTokens)
        // New n-grams of size k end at indices `[max(startIdx, k-1),
        // tokens.count)`. Amortises O(k) work per appended token per table
        // rather than re-scanning the whole history.
        for k in minNgramSize ... maxNgramSize {
            let firstNewEnd = max(startIdx, k - 1)
            guard firstNewEnd < tokens.count else { continue }
            for i in firstNewEnd ..< tokens.count {
                if let h = hashNgramEndingAt(i, size: k) {
                    tables[k]![h, default: []].append(i)
                }
            }
        }
    }

    /// Propose up to `maxDraft` continuation tokens.
    ///
    /// Walks the size ladder from `maxNgramSize` down to `minNgramSize` and
    /// returns the continuation of the **most recent** prior occurrence at
    /// the longest size that has at least `minHits` prior occurrences.
    /// Returns an empty array on miss across all sizes.
    func proposeDraft(maxDraft: Int) -> [Int] {
        guard tokens.count >= minNgramSize, maxDraft > 0 else { return [] }
        let lastEnd = tokens.count - 1
        for k in stride(from: maxNgramSize, through: minNgramSize, by: -1) {
            guard tokens.count >= k,
                  let h = hashNgramEndingAt(lastEnd, size: k),
                  let positions = tables[k]?[h]
            else { continue }
            // Prior occurrences exclude the current suffix itself
            // (the entry at `lastEnd`). Apply the min-hits gate.
            let priorOccurrences = positions.filter { $0 < lastEnd }
            guard priorOccurrences.count >= minHits else { continue }
            guard let mostRecentBeforeEnd = priorOccurrences.last else { continue }
            let continuationStart = mostRecentBeforeEnd + 1
            let continuationEnd = min(continuationStart + maxDraft, tokens.count)
            guard continuationStart < continuationEnd else { continue }
            return Array(tokens[continuationStart ..< continuationEnd])
        }
        return []
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
    let ngramDraftMin: Int
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
        precondition(
            parameters.ngramDraftMin >= 1,
            "ngramDraftMin must be >= 1 (got \(parameters.ngramDraftMin)). "
                + "A floor of zero would let empty drafts trigger a verify "
                + "batch.")
        // Clamp the fallback floor so we never try to look up a size larger
        // than the configured ngramSize; the public floor `minNgramSize`
        // applies only when it's at most the primary size.
        let effectiveMinSize = Swift.min(parameters.minNgramSize, parameters.ngramSize)

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
        self.ngramDraftMin = parameters.ngramDraftMin

        let promptTokens = input.text.tokens.asArray(Int.self)
        self.lookup = NGramLookup(
            promptTokens: promptTokens,
            maxNgramSize: parameters.ngramSize,
            minNgramSize: effectiveMinSize,
            minHits: parameters.ngramMinHits)

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

        // `ngramDraftMin` gates short drafts — drafting fewer than N tokens
        // rarely amortises the verify-batch overhead, so fall through to the
        // pure autoregressive path. With the default `ngramDraftMin = 1`,
        // any non-empty draft is allowed (matches the prior single-size
        // behavior).
        if draftInts.count < ngramDraftMin {
            // Miss / too-short — pure autoregressive step.
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
