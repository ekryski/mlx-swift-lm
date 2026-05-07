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
/// caller plug in any conforming tokenizer (MLX `Tokenizer`, a wrapped
/// Hugging Face one, or a test fake). Just needs `decode(tokens:)`.
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

    /// ±N token bleed on each grouped span — gives the embedder coherent
    /// text instead of a span boundary mid-sentence. Mirrors AMD's
    /// `BLEED = 32` constant; tunable for the retrieval-quality sweep.
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
}
