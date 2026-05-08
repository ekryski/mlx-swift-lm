// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Bridge that wires the V3 engine's eviction callback into Tier 2's
// HTTP client. Mirrors the AMD-side `install_eviction_to_longctx` +
// `_evict_to_longctx_callback` in
// vllm-turboquant/vllm/v1/attention/triattention/backend_helpers.py.
//
// On the AMD path, the callback decodes evicted token positions back to
// text via a separately-loaded transformers tokenizer, groups contiguous
// runs into spans (with ±32 token bleed), and POSTs to /evict/write. On
// Swift, the tokenizer is already in scope via the standard mlx-swift-lm
// `Tokenizer` protocol — caller passes it directly. No subprocess
// tokenizer load needed.
//
// One-shot install: build the engine, register an array of prompt-
// token-ids per session id, install the bridge once, and every
// subsequent eviction round POSTs the decoded spans to longctx-svc.
import Foundation

/// Small protocol surface for the tokenizer the bridge needs. Lets the
/// caller plug in any conforming tokenizer (mlx-swift-lm's `Tokenizer`,
/// a wrapped Hugging Face one, or a test fake). Just needs
/// `decode(tokens:)`.
///
/// Caller-side adapter for swift-transformers' `Tokenizers.Tokenizer`
/// (the one mlx-swift-lm exposes via `loadModel(...)`). MLXLMCommon
/// doesn't link Tokenizers transitively (yyjson dependency lives in
/// the higher-level model crate), so the adapter is documented here
/// for callers to drop into their own code:
///
///     struct AppTokenizer: TriAttentionTokenizerLike {
///         let inner: Tokenizers.Tokenizer
///         func decode(tokens: [Int]) -> String {
///             inner.decode(tokens: tokens, skipSpecialTokens: true)
///         }
///     }
///     TriAttentionRescue.shared.setTokenizer(
///         AppTokenizer(inner: modelContext.tokenizer)
///     )
public protocol TriAttentionTokenizerLike {
    func decode(tokens: [Int]) -> String
}

/// Holds the per-process state used by the eviction bridge: token IDs
/// keyed by seqId, the bound tokenizer, and the longctx client.
/// Implemented as a singleton with a private lock — eviction callbacks
/// fire from the engine's lock; callers register state from the model
/// runner / bridge.
public final class TriAttentionRescue: @unchecked Sendable {

    public static let shared = TriAttentionRescue()

    private let lock = NSLock()
    private var promptTokenIds: [Int: [Int]] = [:]
    private var tokenizer: TriAttentionTokenizerLike?

    /// ±N token bleed on each grouped span before decoding to text.
    /// Default 32 mirrors the AMD path. Sub21 (2026-05-07) explicitly
    /// tested smaller values (BLEED=4) and found they HURT retrieval —
    /// most evicted spans are individual fact tokens; tightening the
    /// bleed strips the fact entirely. The actual retrieval bug we
    /// chased was on the QUERY side (passing the full user message
    /// instead of just the question to the embedder), not the chunk
    /// side. See the query-extract logic in rehydratePrompt below.
    public var spanBleed: Int = 32

    private init() {}

    /// Stash the tokenized prompt for a session so the eviction
    /// callback can decode evicted positions back to text. Call from
    /// the bridge / model runner per request.
    public func setPromptTokenIds(_ tokens: [Int], seqId: Int = 0) {
        lock.lock(); defer { lock.unlock() }
        promptTokenIds[seqId] = tokens
    }

    /// Bind the tokenizer used to decode evicted token IDs. Call once
    /// at engine init.
    public func setTokenizer(_ tok: TriAttentionTokenizerLike) {
        lock.lock(); defer { lock.unlock() }
        tokenizer = tok
    }

    /// Bind the longctx session id for the active request. Mirrors
    /// AMD's per-request `set_longctx_session_id` semantics — empty/nil
    /// resets to the singleton default so a header-bearing request can't
    /// leak its id into a later request that omits it. Call from
    /// ChatSession or the top-level driver before each turn.
    @discardableResult
    public func setSessionID(_ id: String?) -> String {
        TriAttentionLongctxClient.shared.setSessionID(id)
    }

    /// Wire the bridge to the V3 engine. Idempotent; replaces any prior
    /// callback. After this returns true, every successful eviction
    /// round POSTs decoded spans to `LONGCTX_ENDPOINT`/evict/write
    /// (no-op if the env var is unset).
    @discardableResult
    public func install(on engine: TriAttentionV3Engine) -> Bool {
        engine.setEvictionCallback { [weak self] seqId, evictedPositions, n in
            self?.handleEviction(
                seqId: seqId, evictedPositions: evictedPositions, count: n
            )
        }
        return true
    }

    private func handleEviction(
        seqId: Int, evictedPositions: [Int], count: Int
    ) {
        let snapshot: (tokens: [Int]?, tok: TriAttentionTokenizerLike?) = {
            lock.lock(); defer { lock.unlock() }
            return (promptTokenIds[seqId], tokenizer)
        }()
        guard let tokens = snapshot.tokens, let tok = snapshot.tok,
              !tokens.isEmpty, !evictedPositions.isEmpty
        else { return }

        // Group contiguous evicted positions into runs, then expand
        // each run by ±spanBleed for embedder context — same shape as
        // AMD's `_evict_to_longctx_callback`.
        let sortedPos = evictedPositions
            .filter { 0 <= $0 && $0 < tokens.count }
            .sorted()
        guard !sortedPos.isEmpty else { return }
        var runs: [(Int, Int)] = []
        var runStart = sortedPos[0]
        var runEnd = sortedPos[0]
        for p in sortedPos.dropFirst() {
            if p == runEnd + 1 {
                runEnd = p
            } else {
                runs.append((runStart, runEnd))
                runStart = p
                runEnd = p
            }
        }
        runs.append((runStart, runEnd))

        var spans: [EvictionSpan] = []
        spans.reserveCapacity(runs.count)
        for (s, e) in runs {
            let ws = max(0, s - spanBleed)
            let we = min(tokens.count, e + 1 + spanBleed)
            let slice = Array(tokens[ws..<we])
            let text = tok.decode(tokens: slice)
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            spans.append(EvictionSpan(
                text: text,
                tokenStart: ws, tokenEnd: we,
                layer: -1,    // multi-layer eviction; layer not surfaced
                score: 0.0    // not surfaced through callback yet
            ))
        }
        if spans.isEmpty { return }

        // Synchronous POST — eviction callback is already off the hot
        // decode path (only fires when V3's policy triggers).
        TriAttentionLongctxClient.shared.writeEvicted(spans)
    }

    /// Test / debug accessor — number of token-id stash entries.
    /// Exposed for unit tests; not part of the production surface.
    public func stashCount() -> Int {
        lock.lock(); defer { lock.unlock() }
        return promptTokenIds.count
    }

    /// Drop a session's stashed token ids. Called when a request
    /// completes; matches the AMD-side per-request cleanup.
    public func clearSession(seqId: Int) {
        lock.lock(); defer { lock.unlock() }
        promptTokenIds.removeValue(forKey: seqId)
    }

    /// Extract the discriminating retrieval-query signal from a
    /// potentially-long user message. Mirrors the AMD-side
    /// _extract_query_signal in
    /// vllm/v1/attention/triattention/prefill_rehydrate.py.
    ///
    /// Strategy:
    ///   1. Find the LAST "QUESTION:" marker (case-insensitive). If
    ///      present, return the trailing piece. NIAH-style harnesses
    ///      and structured-prompt callers often inject this marker.
    ///   2. Fall back to the last `tailChars` of the message. Generic
    ///      default for typical chat where the question is at the end.
    ///
    /// Why: passing a long haystack+question as the retrieval query
    /// drowns the question signal in MiniLM embedding. Cosine sim
    /// then picks chunks by haystack-similarity (filler beats fact).
    /// Sub21's diagnostic measured the swing at 50× — fact wins 13×
    /// over filler with question only; filler wins 4× over fact with
    /// full message.
    public static func extractQuerySignal(
        from text: String, tailChars: Int = 512
    ) -> String {
        // Try "QUESTION:" marker first.
        let lower = text.lowercased()
        if let markerRange = lower.range(of: "question:", options: .backwards) {
            let after = text.index(markerRange.upperBound,
                                   offsetBy: 0,
                                   limitedBy: text.endIndex)
                ?? text.endIndex
            let tail = text[after...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty {
                return tail
            }
        }
        // Tail fallback.
        if text.count <= tailChars {
            return text
        }
        let startIdx = text.index(text.endIndex, offsetBy: -tailChars)
        return String(text[startIdx...])
    }

    /// Convenience for TokenIterator-style call sites: if any cache in
    /// `caches` is a TriAttentionKVCache, extract `tokens` (the input
    /// MLXArray of prompt token ids) and stash via `setPromptTokenIds`.
    /// No-op when V3 isn't engaged. Lets a single hook point in
    /// TokenIterator's init feed every model's V3-eviction callback.
    ///
    /// Called once per request — the per-request reset semantics in
    /// V3's engine (`reset_seq_state` on AMD's set_prompt_token_ids;
    /// equivalent stash-overwrite here) make this safe to re-call.
    public func maybeStashPrompt(
        tokens: [Int]?, in caches: [Any]?, seqId: Int = 0
    ) {
        guard let tokens, !tokens.isEmpty,
              let caches, caches.contains(where: { $0 is TriAttentionKVCache })
        else { return }
        setPromptTokenIds(tokens, seqId: seqId)
    }

    /// Tier 3 prefill rehydrate: query longctx-svc for spans relevant
    /// to the user's current message, format them as a system-message
    /// body, return the string for the caller to prepend. Mirrors AMD's
    /// `maybe_rehydrate_messages` in
    /// vllm-turboquant/vllm/v1/attention/triattention/prefill_rehydrate.py.
    ///
    /// Returns nil when:
    ///   - LONGCTX_ENDPOINT is unset (no rescue available)
    ///   - the query is empty
    ///   - no chunks come back (cold session, score floor too high)
    ///   - the formatted body would exceed `maxChars` (caller cap)
    ///
    /// Caller responsibility: prepend the returned string as the body
    /// of a system message before the user message in the prompt
    /// rendered to the model. Format mirrors AMD's wrapper:
    ///
    ///     [Recovered context from earlier in this session
    ///     (evicted from KV cache, restored via longctx)]:
    ///     --- evicted span (tokens 100..130, layer -1) ---
    ///     <span text>
    ///     --- evicted span (tokens 200..230, layer -1) ---
    ///     <span text>
    ///
    /// Layer / token-range comments are inline — useful for debugging,
    /// ignored by the model.
    public func rehydratePrompt(
        query: String,
        topK: Int = 8,
        scoreFloor: Float = 0.20,
        maxChars: Int = 8000,
        seqId: Int = 0,
        queryTailChars: Int = 512
    ) -> String? {
        guard !query.isEmpty else { return nil }
        // Extract the discriminating retrieval signal from a potentially-
        // long user message. Sub21 (AMD, 2026-05-07) found that passing
        // a long haystack+question as the query drowned the question
        // signal in MiniLM embedding — cosine similarity ranked chunks
        // by haystack-similarity, picking filler over fact (4× wrong
        // direction). Solution: extract just the question.
        //
        // Two strategies, tried in order:
        //   1. Last "QUESTION:" marker (NIAH-style harnesses inject it).
        //   2. Tail N chars (generic fallback for typical chat where
        //      the question is the last sentence).
        // Caller can pass the full message; this function does the
        // right thing.
        let signal = Self.extractQuerySignal(
            from: query, tailChars: queryTailChars
        )
        let chunks = TriAttentionLongctxClient.shared.retrieveEvicted(
            query: signal, topK: topK, scoreFloor: scoreFloor
        )
        guard !chunks.isEmpty else { return nil }

        let header =
            "[Recovered context from earlier in this session "
            + "(evicted from KV cache, restored via longctx)]:\n"
        var body = header
        for c in chunks {
            let trimmed = c.text.trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let tr = c.tokenRange.count >= 2
                ? (c.tokenRange[0], c.tokenRange[1]) : (0, 0)
            let block =
                "\n--- evicted span (tokens \(tr.0)..\(tr.1), "
                + "layer \(c.layer)) ---\n\(trimmed)\n"
            if body.count + block.count > maxChars { break }
            body.append(block)
        }
        return body == header ? nil : body
    }
}
