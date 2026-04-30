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
//   - `ngramRouteDecision(parameters:)` — eligibility + env-var defaults
//     used by `MLXLMCommon.generate(...)` to auto-route.

// MARK: - Route decision (auto-routing eligibility + env-var opt-in)

/// Outcome of the n-gram-route decision: whether to engage the
/// speculative iterator and (if env-var-driven) the patched parameters
/// that should be passed to it.
public struct NGramRouteDecision: Sendable {
    /// Engage `NGramSpeculativeTokenIterator` (cache-trimmability willing).
    public let shouldEngage: Bool

    /// Parameters the iterator should be constructed with. Equal to the
    /// caller-supplied parameters when the route was opted into via
    /// Swift code; differs when the env-var path injected default
    /// `ngramSize` / `maxNgramDraftTokens` values.
    public let parameters: GenerateParameters
}

/// N-gram speculative-decoding sensible defaults applied when the
/// caller opts in via env var (`MLX_NGRAM_ENABLED=1`) without setting
/// `ngramSize` / `maxNgramDraftTokens` in code. Picked from the
/// `ngram-spot` benchmark sweep on the supported-model set:
///   - `ngramSize = 3` — strikes the best recall/precision balance on
///     mixed input-grounded + paraphrastic prompts.
///   - `maxNgramDraftTokens = 4` — paired with `MLX_NGRAM_ADAPTIVE=1`
///     (default ON) the iterator scales this up to ~12 on regurgitative
///     workloads and down to 1-2 on paraphrastic ones.
public let ngramEnvDefaultSize: Int = 3
public let ngramEnvDefaultMaxDraft: Int = 4

/// Decide whether `MLXLMCommon.generate(...)` should auto-route to the
/// n-gram speculative iterator, and with which parameters.
///
/// **Opt-in modes** (any one is enough; explicit Swift parameters always
/// win over the env-var path):
///   1. **Swift parameters.** `parameters.ngramSize >= 1 &&
///      parameters.maxNgramDraftTokens >= 1`. Use this for production
///      code paths where you want speculative decoding for a known set
///      of requests.
///   2. **Env var.** `MLX_NGRAM_ENABLED=1`. Convenient for benchmark
///      runs and for one-off experimentation without recompiling.
///      Applies sensible defaults (`ngramSize = 3`,
///      `maxNgramDraftTokens = 4`); explicit Swift values still win
///      when both are set.
///
/// **Disqualifiers** (any one declines the route, falls back to
/// `TokenIterator`):
///   - `temperature != 0` — the verifier compares draft tokens against
///     the model's argmax, which is greedy. Non-zero temperature would
///     diverge from a sampling baseline (Leviathan-style accept/reject
///     sampling is the proper extension and is tracked as a follow-up).
///
/// The iterator now plumbs the full logit-processor chain
/// (`repetitionPenalty` / `presencePenalty` / `frequencyPenalty` /
/// `additionalProcessors`) through both the verify forward and the AR
/// fallback — see ``NGramSpeculativeTokenIterator``'s init / `prepare`
/// / `speculateRound`. Penalties no longer disqualify the route.
///
/// The cache-trimmability check is *not* in this predicate (it's done
/// later, after the cache is probed); a hybrid GDN/Mamba target falls
/// back to `TokenIterator` cleanly inside `generate`.
public func ngramRouteDecision(parameters: GenerateParameters) -> NGramRouteDecision {
    // Disqualifier: non-greedy sampling. The verifier's argmax compare
    // makes the n-gram path fundamentally greedy; sampling baselines
    // need accept/reject sampling, which is a separate iterator.
    if parameters.temperature != 0 {
        return NGramRouteDecision(shouldEngage: false, parameters: parameters)
    }

    // Swift-parameter opt-in — wins outright when set.
    let optedInBySwift =
        parameters.ngramSize >= 1 && parameters.maxNgramDraftTokens >= 1
    if optedInBySwift {
        return NGramRouteDecision(shouldEngage: true, parameters: parameters)
    }

    // Env-var opt-in — apply sensible defaults when the caller didn't
    // set `ngramSize` / `maxNgramDraftTokens` themselves. We respect any
    // partial caller settings (e.g. they set `ngramSize` to 5 but left
    // the cap at 0): only fields still at the disabled default get
    // populated.
    let envEnabled =
        ProcessInfo.processInfo.environment["MLX_NGRAM_ENABLED"] == "1"
    if envEnabled {
        var patched = parameters
        if patched.ngramSize < 1 {
            patched.ngramSize = ngramEnvDefaultSize
        }
        if patched.maxNgramDraftTokens < 1 {
            patched.maxNgramDraftTokens = ngramEnvDefaultMaxDraft
        }
        return NGramRouteDecision(shouldEngage: true, parameters: patched)
    }

    return NGramRouteDecision(shouldEngage: false, parameters: parameters)
}

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
    /// returns the longest hit. Within a hit, when multiple prior occurrences
    /// exist (`useMultiCandidate == true`), groups continuations by their
    /// first token, picks the **most frequent** group (tiebreaking by most
    /// recent), and optionally enforces a dominance gate (the winning group
    /// must outnumber all others combined). This mirrors llama.cpp's
    /// `ngram-map` complex-mode selection (`COMMON_NGRAM_MAX_VALUES = 4`,
    /// `max_occur > 2 * sum_others`). When only one candidate exists or
    /// `useMultiCandidate == false`, falls back to the single most-recent
    /// match. Returns an empty array on miss across all sizes.
    ///
    /// - Parameter maxDraft: per-round draft cap (after adaptive scaling).
    /// - Parameter useMultiCandidate: enable frequency-based selection.
    /// - Parameter requireDominance: require winning group to dominate
    ///   (`max > 2 * sum_others`); if false, just return most-frequent.
    func proposeDraft(
        maxDraft: Int,
        useMultiCandidate: Bool = false,
        requireDominance: Bool = false
    ) -> [Int] {
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

            let chosenPos: Int
            if useMultiCandidate, priorOccurrences.count >= 2 {
                // Group prior occurrences by the first continuation token.
                // Track count + most-recent position per group.
                var groups: [Int: (count: Int, lastPos: Int)] = [:]
                for pos in priorOccurrences {
                    let next = pos + 1
                    guard next < tokens.count else { continue }
                    let firstTok = tokens[next]
                    if let prior = groups[firstTok] {
                        groups[firstTok] = (prior.count + 1, max(prior.lastPos, pos))
                    } else {
                        groups[firstTok] = (1, pos)
                    }
                }
                guard !groups.isEmpty else { continue }
                // Pick max-count, tiebreak by most-recent position.
                let best = groups.max { lhs, rhs in
                    lhs.value.count != rhs.value.count
                        ? lhs.value.count < rhs.value.count
                        : lhs.value.lastPos < rhs.value.lastPos
                }!
                if requireDominance {
                    let bestCount = best.value.count
                    let sumOthers = groups.values.reduce(0) { $0 + $1.count } - bestCount
                    // llama.cpp's gate: max_occur > 2 * sum_others (strict).
                    // We use the same form so this is a near-port.
                    if sumOthers > 0 && bestCount <= 2 * sumOthers {
                        continue  // not dominant, fall down to smaller n
                    }
                }
                chosenPos = best.value.lastPos
            } else {
                guard let mostRecentBeforeEnd = priorOccurrences.last else { continue }
                chosenPos = mostRecentBeforeEnd
            }

            let continuationStart = chosenPos + 1
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
/// Enable via ``GenerateParameters/ngramSize`` >= 1 and
/// ``GenerateParameters/maxNgramDraftTokens`` >= 1. With both zero (default),
/// construction traps via `precondition` — callers should switch to
/// ``TokenIterator`` for non-speculative decode (or use the auto-routing
/// in ``MLXLMCommon/generate(input:cache:parameters:context:wiredMemoryTicket:)``,
/// which handles the fall-through automatically and additionally supports
/// the `MLX_NGRAM_ENABLED=1` env-var opt-in path with sensible defaults).
///
/// The accept walk has two paths. By default it accepts a drafted token
/// only when it matches the main model's argmax at the same verify
/// position; the strict-greedy guard (``GenerateParameters`` /
/// `MLX_NGRAM_STRICT_GREEDY`, default ON) additionally stops the chain at
/// any position where the top-1 vs top-2 logit margin is tight, since a
/// matching argmax there could be a numerical-drift coincidence between
/// the batched verify forward and a sequential greedy reference. The
/// matching token is still emitted (as the bonus) — the guard's effect is
/// to prevent *further* drafts from extending past a drift-risky position
/// and compounding the divergence. Output is byte-identical to
/// ``TokenIterator`` at `temperature: 0` for any draft-source-bound
/// workload — the spec-decode contract.
///
/// Non-greedy spec decode would require a per-position resample loop and
/// is tracked as a follow-up.
public struct NGramSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let mainModel: any LanguageModel
    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    let sampler: LogitSampler

    /// Optional logit processor — applies repetition / presence / frequency
    /// penalties and any caller-supplied `additionalProcessors`. Mirrors
    /// ``SpeculativeTokenIterator/processor`` semantics: only the *accepted*
    /// prefix + bonus advances the original processor's state, while a
    /// throwaway value-copy advances through the full verify-batch so the
    /// per-position logits each see the prior position's sampled token.
    var processor: (any LogitProcessor)?

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

    /// Adaptive-draft state — current draft cap, scaled per-round when
    /// `MLX_NGRAM_ADAPTIVE=1`. Starts at the configured max and floats in
    /// `[1, maxNgramDraftTokens]` based on recent acceptance rate.
    private var currentMaxDraft: Int = 0
    /// Rolling window of (proposed, accepted) pairs for the most recent
    /// verify rounds. Used to compute the trailing acceptance rate.
    private var recentRounds: [(Int, Int)] = []

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

    public var specDecodeProposed: Int { ngramProposedCount }
    public var specDecodeAccepted: Int { ngramAcceptedCount }

    /// Bytes held by the runtime KV cache after generation. Mirrors
    /// ``TokenIterator/kvCacheMemoryBytes`` so the bench harness can report
    /// the same footprint regardless of which iterator drove decode.
    public var kvCacheMemoryBytes: Int? {
        mainCache.isEmpty ? nil : mainCache.reduce(0) { $0 + $1.memoryBytes }
    }

    /// Verbose tracing toggle (`MLX_NGRAM_DEBUG=1`). When enabled, every
    /// speculation round logs a `[NGRAM]` line showing draft length,
    /// acceptance count, KV cache offset, and the AR/verify branch taken.
    /// Off by default — unset means zero overhead.
    private static var debugTracing: Bool {
        ProcessInfo.processInfo.environment["MLX_NGRAM_DEBUG"] == "1"
    }

    /// Force the AR fallback path on every round (`MLX_NGRAM_FORCE_AR=1`),
    /// bypassing draft proposal entirely. Diagnostic for isolating verify-path
    /// bugs from AR-path bugs.
    private static var forceAR: Bool {
        ProcessInfo.processInfo.environment["MLX_NGRAM_FORCE_AR"] == "1"
    }

    /// AR-fallback batch size (`MLX_NGRAM_AR_BATCH=N`). When the lookup misses,
    /// run N decode steps async-pipelined and sync once at the end so the
    /// per-token GPU→CPU `.item()` sync isn't on the critical path. Larger
    /// values pipeline more aggressively but waste up to (N-1) forward passes
    /// if EOS lands mid-batch. Default 4 — trades a few wasted forwards near
    /// EOS for ~3-5× faster AR throughput on memory-bound decode.
    private static var arBatchSize: Int {
        guard let raw = ProcessInfo.processInfo.environment["MLX_NGRAM_AR_BATCH"],
              let n = Int(raw), n >= 1 else { return 4 }
        return n
    }

    /// Adaptive-draft toggle (`MLX_NGRAM_ADAPTIVE`, **default ON**). When on,
    /// the iterator scales the per-round draft cap between 1 and
    /// ``maxNgramDraftTokens`` based on a rolling acceptance rate over the
    /// last ``adaptiveWindow`` verify rounds. Inspired by EAGLE-3's
    /// instance-adaptive depth and llama.cpp's dynamic `--draft-max` —
    /// expand when the workload is regurgitative (high accept), shrink
    /// when it's paraphrastic (low accept) so verify-batch overhead never
    /// dominates. Mathematically (cap clamped to `maxNgramDraftTokens`,
    /// floor 1):
    ///   - rate ≥ ``adaptiveExpandThreshold`` (default 0.7):
    ///     `current ← current + 1 + current/2` (≈ 1.5× growth, with a
    ///     +1 nudge so the formula doesn't stall at small `current`).
    ///   - rate ≤ ``adaptiveShrinkThreshold`` (default 0.3): halve.
    ///   - otherwise: hold steady.
    ///
    /// Default flipped to ON (2026-04-28) after the recipe-bulk benchmark on
    /// Gemma 4 26B A4B showed adaptive+strict beating static D=8 by ~9% and
    /// recovering parity-or-better against baseline on mixed template/content
    /// workloads where static D over-drafted. Set `MLX_NGRAM_ADAPTIVE=0` to
    /// pin the iterator to a fixed `maxNgramDraftTokens`.
    private static var adaptiveDraftEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_NGRAM_ADAPTIVE"] != "0"
    }
    private static var adaptiveWindow: Int {
        guard let raw = ProcessInfo.processInfo.environment["MLX_NGRAM_ADAPTIVE_WINDOW"],
              let n = Int(raw), n >= 1 else { return 4 }
        return n
    }
    private static var adaptiveExpandThreshold: Double {
        guard let raw = ProcessInfo.processInfo.environment["MLX_NGRAM_ADAPTIVE_HI"],
              let v = Double(raw), v > 0 && v <= 1 else { return 0.7 }
        return v
    }
    private static var adaptiveShrinkThreshold: Double {
        guard let raw = ProcessInfo.processInfo.environment["MLX_NGRAM_ADAPTIVE_LO"],
              let v = Double(raw), v >= 0 && v < 1 else { return 0.3 }
        return v
    }

    /// Multi-candidate selection (`MLX_NGRAM_MULTI_CANDIDATE=1`, default ON).
    /// Mirrors llama.cpp's `ngram-map` complex-mode: when a key n-gram has
    /// multiple prior occurrences, group continuations by their first token,
    /// pick the most-frequent group (tiebreak: most recent). Improves accept
    /// rate on long-context workloads where the same prefix has multiple
    /// observed continuations (RAG, multi-turn chat, repeated templates).
    /// Set to `0` for the legacy "most-recent" behaviour.
    private static var multiCandidateEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_NGRAM_MULTI_CANDIDATE"] != "0"
    }

    /// Dominance gate (`MLX_NGRAM_DOMINANCE=1`, default off). When enabled,
    /// the winning candidate group must dominate all others combined
    /// (`max_count > 2 * sum_others`) — otherwise the iterator falls back to
    /// the next-shorter n-gram size or AR. llama.cpp uses this to avoid
    /// drafting from ambiguous patterns; in our setting it trades recall for
    /// precision.
    private static var dominanceGateEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_NGRAM_DOMINANCE"] == "1"
    }

    /// Strict-greedy guard (`MLX_NGRAM_STRICT_GREEDY`, **default ON**). When
    /// enabled, the verify path checks the top-1 vs top-2 logit margin at
    /// each position and refuses to accept a draft whose match is suspicious
    /// (i.e. could be a batched-vs-sequential argmax flip caused by
    /// numerical drift). This eliminates the "regurgitation cascade" failure
    /// mode at high D values on summarization-style tasks at the cost of
    /// some throughput.
    ///
    /// Default flipped to ON (2026-04-28) after the recipe-bulk and code-refactor
    /// sweeps confirmed strict-greedy preserves byte-identical output to baseline
    /// while costing only a few hundred microseconds of GPU sort per verify
    /// round (the sort folds into the same `eval()` as the main argmax sample
    /// — zero extra GPU sync). Set `MLX_NGRAM_STRICT_GREEDY=0` to disable.
    private static var strictGreedyEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_NGRAM_STRICT_GREEDY"] != "0"
    }
    private static var strictGreedyEpsilon: Float {
        guard let raw = ProcessInfo.processInfo.environment["MLX_NGRAM_STRICT_EPSILON"],
              let v = Float(raw), v > 0 else { return 0.5 }
        return v
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
        self.processor = parameters.processor()
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

        self.currentMaxDraft = parameters.maxNgramDraftTokens

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }

        if Self.debugTracing {
            print("[NGRAM] iterator engaged: ngramSize=\(self.ngramSize) "
                + "maxDraft=\(self.maxNgramDraftTokens) "
                + "draftMin=\(self.ngramDraftMin) "
                + "minHits=\(parameters.ngramMinHits) "
                + "minSize=\(effectiveMinSize) "
                + "promptTokens=\(promptTokens.count)")
        }
    }

    /// Prefill the main model, then advance one decode step exactly like
    /// ``TokenIterator/prepare(input:windowSize:)`` does. The "primes the
    /// pump" step matters for two reasons:
    ///   1. Off-by-one parity. The first emitted token must be the model's
    ///      argmax at the last prompt position. Without this step, the first
    ///      `speculateRound` would conflate "prefill's last token" with "the
    ///      first generated token" — producing a stream offset by one vs.
    ///      ``TokenIterator``.
    ///   2. **Cache-write commit on Gemma 4.** Gemma 4's `prepare(...)` ends
    ///      its chunked prefill with `asyncEval(cache)` and `clearCache()`,
    ///      not a synchronous `eval(cache)`. The pending KV writes only
    ///      commit when something forces the GPU pipeline to drain. The
    ///      `eval(y.tokens)` at the bottom of this method is exactly that
    ///      barrier — without it, the next forward pass in
    ///      ``speculateRound()`` reads a cache whose prefill writes haven't
    ///      committed yet and produces garbage logits (manifests as a few
    ///      tokens of nonsense before the model recovers, or a hard
    ///      derail). This mirrors the "pad-token bug" comment on
    ///      ``TokenIterator/prepare(input:windowSize:)``.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        // Seed the processor with the prompt — the rep/presence/freq
        // penalty contexts use this as the initial token-ring window
        // for didSample tracking. No-op when processor is nil.
        processor?.prompt(input.text.tokens)

        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            // "Primes the pump": run one forward pass on the residual prompt
            // tokens, sample the first generated token, and commit cache
            // writes before any decode-time forward pass.
            let result = mainModel(tokens[text: .newAxis], cache: mainCache, state: nil)
            quantizeKVCache(&mainCache)
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            mainState = result.state
            // Sync-eval is required on Gemma 4 — see doc comment above.
            eval(token)
            let tokenInt = token.item(Int.self)
            y = .init(tokens: token)
            pendingTokens.append(tokenInt)
            lookup.extend(with: [tokenInt])
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            // Same sync barrier as the `.tokens` branch — VLMs that return
            // logits here can have the same async-prefill commit issue.
            eval(token)
            let tokenInt = token.item(Int.self)
            y = .init(tokens: token)
            mainState = result.state
            pendingTokens.append(tokenInt)
            lookup.extend(with: [tokenInt])
        }
    }

    /// One speculation round: look up draft tokens, verify with main model,
    /// emit accepted tokens to `pendingTokens`.
    mutating func speculateRound() {
        // Adaptive draft length: scale `currentMaxDraft` against the rolling
        // accept rate before drafting. The cap is the static configured
        // `maxNgramDraftTokens`; the floor is 1 (always allow at least one
        // draft when the lookup hits — going to zero would force the AR
        // fallback path even when speculation might pay off). This is the
        // EAGLE-3 / llama.cpp instance-adaptive idea, written in Swift.
        if Self.adaptiveDraftEnabled, recentRounds.count >= Self.adaptiveWindow {
            let totals = recentRounds.reduce(into: (proposed: 0, accepted: 0)) {
                $0.proposed += $1.0
                $0.accepted += $1.1
            }
            let rate = totals.proposed > 0
                ? Double(totals.accepted) / Double(totals.proposed) : 0
            let priorMax = currentMaxDraft
            if rate >= Self.adaptiveExpandThreshold {
                currentMaxDraft = Swift.min(maxNgramDraftTokens, currentMaxDraft + 1 + currentMaxDraft / 2)
            } else if rate <= Self.adaptiveShrinkThreshold {
                currentMaxDraft = Swift.max(1, currentMaxDraft / 2)
            }
            if Self.debugTracing && currentMaxDraft != priorMax {
                print("[NGRAM] adaptive: rate=\(String(format: "%.2f", rate)) "
                    + "draft \(priorMax)→\(currentMaxDraft)")
            }
        }
        let perRoundCap = Self.adaptiveDraftEnabled ? currentMaxDraft : maxNgramDraftTokens
        let remaining = maxTokens.map { $0 - tokenCount } ?? perRoundCap
        let budget = Swift.min(remaining, perRoundCap)
        guard budget > 0 else { return }

        let draftInts: [Int]
        if Self.forceAR {
            draftInts = []
        } else {
            draftInts = lookup.proposeDraft(
                maxDraft: budget,
                useMultiCandidate: Self.multiCandidateEnabled,
                requireDominance: Self.dominanceGateEnabled)
        }

        // `ngramDraftMin` gates short drafts — drafting fewer than N tokens
        // rarely amortises the verify-batch overhead, so fall through to the
        // pure autoregressive path. With the default `ngramDraftMin = 1`,
        // any non-empty draft is allowed (matches the prior single-size
        // behavior).
        if draftInts.count < ngramDraftMin {
            // Miss / too-short — fall back to autoregressive decode.
            //
            // Default path: runs ``Self.arBatchSize`` forward passes in a
            // row without syncing between them, then syncs once at the end.
            // The CPU-side `.item()` sync that ``TokenIterator`` defers via
            // its previousY trick is the dominant overhead at small model
            // size (Gemma 4 E2B 4bit decode ≈ 10 ms/token; an eager sync
            // adds ~5 ms on M1 Max). Batched async-eval reclaims that.
            //
            // EOS over-decode is bounded: if EOS lands at index `i` in the
            // batch, we still do (N-i) wasted forward passes whose tokens
            // the loop drains but never emits. With N=4 that's at worst
            // 3 wasted passes per generation.
            //
            // **Processor-active path:** when a logit processor is set
            // (`repetitionPenalty` etc. or `additionalProcessors`), each
            // step's logits must see the previous step's `didSample`
            // update — that's a per-token CPU↔GPU dependency. We collapse
            // the batch to size 1 here so the chain `forward → process →
            // sample → didSample → forward` runs in proper order. The
            // throughput cost is exactly the AR-batch optimisation we'd
            // otherwise get; in practice that's <5 ms/token, which is
            // small relative to the work the processor itself does.
            let arBatch: Int
            if processor != nil {
                arBatch = 1
            } else {
                arBatch = Swift.min(Self.arBatchSize, Swift.max(1, remaining))
            }
            var collected: [MLXArray] = []
            collected.reserveCapacity(arBatch)
            var currentY = y
            for _ in 0 ..< arBatch {
                let result = mainModel(
                    currentY[text: .newAxis], cache: mainCache, state: mainState)
                quantizeKVCache(&mainCache)
                var logits = result.logits[0..., -1, 0...]
                logits = processor?.process(logits: logits) ?? logits
                let token = sampler.sample(logits: logits)
                processor?.didSample(token: token)
                asyncEval(token)
                mainState = result.state
                collected.append(token)
                currentY = .init(tokens: token)
            }
            // Single sync for all batched AR steps.
            let combined = concatenated(collected)
            eval(combined)
            let ints = combined.asArray(Int.self)
            pendingTokens.append(contentsOf: ints)
            lookup.extend(with: ints)
            y = currentY
            if Self.debugTracing {
                print("[NGRAM] AR-batch draft=\(draftInts.count) "
                    + "size=\(arBatch) emit=\(ints)")
            }
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
        //
        // Two paths share the strict-greedy guard:
        //
        //  - **Processor-active**: per-position sampling with a value-copy
        //    of the processor (mirrors ``SpeculativeTokenIterator``'s
        //    `verifyProcessor` pattern). Each verify position sees the
        //    prior position's didSample update, matching what a sequential
        //    baseline would do. The original `processor` is advanced only
        //    on accepted prefix + bonus, below.
        //  - **No processor**: batch-sample all positions in one operation
        //    — the original fast path.
        //
        // Both paths produce a `[numDraft+1]` `mainTokens` array and (when
        // `strictGreedyEnabled`) a parallel `[numDraft+1]` `margins`
        // array. The accept walk below is identical for both.
        let strictGuard = Self.strictGreedyEnabled
        let mainTokens: MLXArray
        let margins: [Float]

        if var verifyProcessor = processor {
            var sampled = [MLXArray]()
            sampled.reserveCapacity(numDraft + 1)
            var marginSlices = [MLXArray]()
            if strictGuard { marginSlices.reserveCapacity(numDraft + 1) }
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                sampled.append(token)
                if strictGuard {
                    // Margin computed on the *processed* logits — the
                    // sampler's actual decision surface.
                    let top2 = top(logits, k: 2, axis: -1)  // [1, 2]
                    let top2Sorted = MLX.sorted(top2, axis: -1)
                    marginSlices.append(top2Sorted[0..., 1] - top2Sorted[0..., 0])
                }
            }
            mainTokens = concatenated(sampled)
            if strictGuard {
                let m = concatenated(marginSlices)
                eval(mainTokens, m)
                margins = m.asArray(Float.self)
            } else {
                eval(mainTokens)
                margins = []
            }
        } else {
            // No processor — batch-sample all verify positions in one go.
            // Original fast path with the strict-greedy guard folded into
            // the same eval as the argmax sample (zero extra GPU syncs vs.
            // the unguarded path).
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
            if strictGuard {
                // Top-2 via partial sort (`top` doesn't guarantee internal order).
                let top2 = top(verifyLogits, k: 2, axis: -1)
                let top2Sorted = MLX.sorted(top2, axis: -1)
                let m = top2Sorted[0..., 1] - top2Sorted[0..., 0]
                eval(mainTokens, m)
                margins = m.asArray(Float.self)
            } else {
                eval(mainTokens)
                margins = []
            }
        }
        let mainList = mainTokens.asArray(Int.self)

        // Acceptance check: walk drafts in order and stop at the first
        // mismatch. Non-consecutive matches are NOT acceptable — `mainList[i]`
        // for i past the first mismatch came from a verify pass against the
        // *drafted* prefix, not the actual accepted prefix, so its values are
        // computed against a wrong context. The previous implementation used
        // `for ... where ...` which silently skipped mismatches and kept
        // accepting — that produced wrong-context tokens and broke greedy-
        // equivalence with `TokenIterator` (some prompts diverged to early
        // EOS or short outputs).
        // Walk drafts in order. Stop at the first mismatch — the tail of
        // `mainTokens` after a mismatch was computed against a *drafted*
        // prefix that isn't the actual accepted prefix, so its values are
        // not real predictions. Under strict-greedy, also break on a
        // tight-margin coincidental match (see guard above).
        let epsilon = Self.strictGreedyEpsilon
        var accepted = 0
        for i in 0 ..< numDraft {
            if mainList[i] != draftInts[i] {
                break
            }
            if strictGuard && margins[i] < epsilon {
                if Self.debugTracing {
                    print("[NGRAM] strict-greedy break: margin[\(i)]="
                        + "\(margins[i]) < \(epsilon) (token=\(mainList[i]))")
                }
                break
            }
            pendingTokens.append(mainList[i])
            accepted += 1
        }

        ngramAcceptedCount += accepted
        ngramProposedCount += numDraft

        // Maintain the rolling window for adaptive draft scaling.
        if Self.adaptiveDraftEnabled {
            recentRounds.append((numDraft, accepted))
            if recentRounds.count > Self.adaptiveWindow {
                recentRounds.removeFirst(recentRounds.count - Self.adaptiveWindow)
            }
        }

        // Advance the *original* processor's state through the accepted
        // prefix. The verify-time copy advanced through all positions but
        // was discarded; the original lives across cycles, so it must
        // observe exactly the tokens that would have been sampled in a
        // sequential baseline — the accepted prefix here.
        // Mirrors ``SpeculativeTokenIterator``'s accept-loop didSample.
        if processor != nil {
            for i in 0 ..< accepted {
                processor?.didSample(token: mainTokens[i ... i])
            }
        }

        // The main model's token at position `accepted` is always emitted —
        // either the correction after the first rejected draft, or the
        // bonus token after a full-accept. Counts as a non-draft emission.
        let finalTokenInt = mainList[accepted]
        pendingTokens.append(finalTokenInt)
        processor?.didSample(token: mainTokens[accepted ... accepted])

        // Trim the KV cache by the number of rejected draft tokens — their
        // K/V rows must be undone so the cache offset matches what the
        // outer world thinks it has. Skip when nothing rejected — `trim(0)`
        // is a Swift-level no-op but still walks every cache layer; on
        // long-attention layer counts (e.g. Gemma 4 26B) this adds up.
        let rejected = numDraft - accepted
        if rejected > 0 {
            trimPromptCache(mainCache, numTokens: rejected)
        }
        quantizeKVCache(&mainCache)

        // Extend lookup with all the real tokens we just committed.
        let emitted = pendingTokens.suffix(accepted + 1)
        lookup.extend(with: Array(emitted))

        // Next round starts from the final emitted token.
        y = .init(tokens: mainTokens[accepted ... accepted])

        if Self.debugTracing {
            print("[NGRAM] verify draft=\(numDraft) accepted=\(accepted) "
                + "rejected=\(rejected) emit=\(accepted + 1) "
                + "next_y=\(finalTokenInt)")
        }
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
