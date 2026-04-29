import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

// MARK: - HuggingFace Integration

// Macro-generated bridges from MLXHuggingFace:
// - `#hubDownloader()` wraps `HubClient.default` (cache-first; auto-detects
//   `~/.cache/huggingface/hub` unless `HF_HUB_CACHE` / `HF_HOME` override it)
//   as an `MLXLMCommon.Downloader`.
// - `#huggingFaceTokenizerLoader()` wraps swift-transformers' `AutoTokenizer`
//   as an `MLXLMCommon.TokenizerLoader`.
//
// The pattern mirrors `IntegrationTesting/IntegrationTestingTests/ToolCallIntegrationTests.swift`.
private let benchmarkDownloader: any Downloader = #hubDownloader()
private let benchmarkTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

// MARK: - Thinking Budget Processor

/// Forces </think> after a token budget to prevent unbounded thinking phases.
/// This keeps benchmarks tractable: models think up to `maxThinkingTokens`,
/// then must emit </think> and generate the actual response.
/// Reference-type box for additive logit-suppression masks, populated lazily
/// on the first `process()` call (vocab size isn't known at processor init).
/// The containing processor is a struct with a non-mutating `process(logits:)`
/// per the `LogitProcessor` protocol, so cacheing through this shared class
/// lets us populate the masks without making `process` mutating.
fileprivate final class SuppressMaskCache {
    var thinking: MLXArray?
    var postExhaust: MLXArray?
}

private struct ThinkingBudgetProcessor: LogitProcessor {
    let thinkStartTokenId: Int32
    let thinkEndTokenId: Int32
    let maxThinkingTokens: Int
    let initialThinkingPhase: Bool
    /// EOS token IDs to suppress while in thinking phase, preventing the model from
    /// terminating before generating </think> and actual response text.
    let eosTokenIds: [Int32]

    var inThinkingPhase: Bool
    var thinkingTokenCount: Int = 0
    /// Set when the budget forces </think>. Prevents the model from re-entering
    /// thinking mode by suppressing <think> in all subsequent tokens. Without this,
    /// models (especially Gemma 4) immediately re-emit <|channel> after the forced
    /// <channel|>, creating a marker-only loop that consumes the generation budget
    /// with no actual content — resulting in Gen PPL = nil.
    var budgetExhausted: Bool = false

    /// Pre-built additive masks applied to logits via one `+` op, replacing the
    /// previous scatter-per-suppressed-token loop. Shape `[V]`, value 0 at all
    /// positions except suppressed IDs which hold -inf. Built lazily on the
    /// first `process()` call when the vocabulary size is first observed.
    ///
    /// Stored in a reference-type box because `process(logits:)` is non-mutating
    /// on the `LogitProcessor` protocol — we need to populate the cache without
    /// marking the method mutating.
    let masks = SuppressMaskCache()

    /// Lazy accumulator of sampled tokens awaiting a batched state-machine
    /// update. Batching eliminates the per-token GPU→CPU sync that would
    /// otherwise block the next forward-pass dispatch on every step — the
    /// primary source of `--think` decode overhead on large models. The
    /// iterator has already called `asyncEval(token)` on every pending entry,
    /// so by the time we flush, evaluation is in flight and a single `eval`
    /// call syncs the whole batch.
    var pendingTokens: [MLXArray] = []

    /// Upper bound on pending-token buffer size. The state machine may be up
    /// to `flushBatchSize - 1` steps stale between flushes, which can cause
    /// the thinking budget to overshoot by the same amount. Chosen small
    /// enough that the overshoot is ≤5% of a typical 200-token budget.
    private static let flushBatchSize = 10

    init(
        thinkStartTokenId: Int32, thinkEndTokenId: Int32,
        maxThinkingTokens: Int, prefilled: Bool = false,
        eosTokenIds: [Int32] = []
    ) {
        self.thinkStartTokenId = thinkStartTokenId
        self.thinkEndTokenId = thinkEndTokenId
        self.maxThinkingTokens = maxThinkingTokens
        self.initialThinkingPhase = prefilled
        self.inThinkingPhase = prefilled
        self.eosTokenIds = eosTokenIds
    }

    mutating func prompt(_ prompt: MLXArray) {
        inThinkingPhase = initialThinkingPhase
        thinkingTokenCount = 0
        budgetExhausted = false
        pendingTokens.removeAll(keepingCapacity: true)
    }

    func process(logits: MLXArray) -> MLXArray {
        // Lazy mask construction: vocab size only known once we see the logits.
        // Shape [V] broadcasts against both [V] and [1, V] input shapes.
        let vocab = logits.shape.last ?? 0
        if masks.thinking == nil && vocab > 0 {
            masks.thinking = buildSuppressMask(
                suppressedIds: eosTokenIds, vocabSize: vocab)
        }
        if masks.postExhaust == nil && vocab > 0 {
            masks.postExhaust = buildSuppressMask(
                suppressedIds: [thinkStartTokenId], vocabSize: vocab)
        }

        // After budget exhaustion, suppress think-start to prevent re-entry loops.
        if budgetExhausted && !inThinkingPhase {
            if let m = masks.postExhaust {
                return logits + m
            }
            return logits
        }

        guard inThinkingPhase else { return logits }

        // Single elementwise-add replaces the per-EOS scatter loop.
        var modified = logits
        if let m = masks.thinking {
            modified = modified + m
        }

        // Budget exceeded: force </think> by boosting its logit to dominate softmax.
        // Use a large FINITE value (not +inf) — softmax(+inf) = exp(+inf)/sum = NaN
        // which causes the sampler to return garbage (token 0), never triggering transition.
        // With logits typically in [-30, 30], 100.0 gives P(</think>) ≈ 1.0 with no NaN.
        //
        // NB: due to batched state flushing, thinkingTokenCount may lag by up to
        // `flushBatchSize` tokens. This means the forced </think> actually triggers
        // somewhere in the [budget, budget + flushBatchSize) range. For a typical
        // 200-token budget that's ≤5% overshoot — acceptable slack in exchange
        // for removing the per-step GPU sync.
        if thinkingTokenCount >= maxThinkingTokens {
            if logits.ndim == 1 {
                modified[Int(thinkEndTokenId)] = MLXArray(Float(100.0))
                modified[Int(thinkStartTokenId)] = MLXArray(-Float.infinity)
            } else {
                modified[0..., Int(thinkEndTokenId)] = MLXArray(Float(100.0))
                modified[0..., Int(thinkStartTokenId)] = MLXArray(-Float.infinity)
            }
        }

        return modified
    }

    mutating func didSample(token: MLXArray) {
        // Defer the item() sync: the iterator has already asyncEval'd the
        // token, so by the time we flush a batch the syncs are close to free.
        // Synchronously calling .item() here would block the next forward
        // pass from dispatching — a per-step stall that dominates on large
        // models.
        pendingTokens.append(token)

        // Dynamic flush threshold: shrink when approaching the budget so
        // forcing triggers within a few tokens of the configured limit.
        //   thinkingTokenCount=195, budget=200 → headroom=5 → flush every 2.
        //   thinkingTokenCount=100, budget=200 → headroom=100 → flush every 10.
        let headroom = inThinkingPhase
            ? max(0, maxThinkingTokens - thinkingTokenCount)
            : Int.max
        let threshold = max(1, min(Self.flushBatchSize, headroom / 2))
        if pendingTokens.count >= threshold {
            flushPending()
        }
    }

    mutating func flushPending() {
        guard !pendingTokens.isEmpty else { return }
        // Batch eval — tokens were already async-dispatched by the iterator,
        // so one eval() call completes the whole queue in one sync point.
        eval(pendingTokens)
        let ids = pendingTokens.map { $0.item(Int32.self) }
        pendingTokens.removeAll(keepingCapacity: true)
        for id in ids {
            applyStateUpdate(id: id)
        }
    }

    private mutating func applyStateUpdate(id: Int32) {
        if id == thinkStartTokenId {
            inThinkingPhase = true
        } else if id == thinkEndTokenId {
            inThinkingPhase = false
            if thinkingTokenCount >= maxThinkingTokens {
                budgetExhausted = true
            }
        } else if inThinkingPhase {
            thinkingTokenCount += 1
        }
    }
}

/// Build a `[V]` additive mask with `-inf` at `suppressedIds` positions and
/// `0` elsewhere. Constructed host-side so the upload to GPU is one kernel —
/// not `N` scatter ops, which is what a naïve
/// `modified[id] = MLXArray(-inf)` loop would emit per call.
fileprivate func buildSuppressMask(
    suppressedIds: [Int32], vocabSize: Int
) -> MLXArray {
    var mask = [Float](repeating: 0, count: vocabSize)
    for id in suppressedIds {
        let i = Int(id)
        if i >= 0 && i < vocabSize {
            mask[i] = -.infinity
        }
    }
    return MLXArray(mask)
}

/// Harmony-format thinking-budget enforcer.
///
/// Analogue to ``ThinkingBudgetProcessor`` for models whose reasoning is
/// delimited by channel transitions rather than a `<think>…</think>`
/// bracket pair. Counts tokens emitted inside the analysis channel; once
/// `maxThinkingTokens` are produced, pins the next `forcedTransitionSequence.count`
/// sampling steps to force-emit the multi-token sequence that closes the
/// analysis message and opens the final-channel visible answer.
///
/// For GPT-OSS / harmony the transition is
/// `<|end|> <|start|> assistant <|channel|> final <|message|>` (6 tokens).
///
/// After the forced transition completes, the model is free to generate
/// final-channel content normally — but the analysis channel is
/// suppressed to prevent bouncing back into reasoning.
///
/// Caveat: the 6 forced tokens land in ``GenerateCompletionInfo.perTokenLogProbs``
/// with logprob ≈ 0 (we pinned the sampler). They slightly bias per-phase
/// PPL toward 1.0 — bounded contribution (6 tokens out of hundreds) and
/// not material to the signal we care about (reasoning-phase PPL/KLD
/// comparisons across KV configs).
private struct HarmonyThinkingBudgetProcessor: LogitProcessor {
    let channelMarkerTokenId: Int32
    let thinkingChannelTokenIds: Set<Int32>
    let generationChannelTokenIds: Set<Int32>
    let forcedTransitionSequence: [Int32]
    let maxThinkingTokens: Int
    let eosTokenIds: [Int32]

    var inThinking: Bool = false
    var awaitingChannelName: Bool = false
    var thinkingTokenCount: Int = 0
    /// When ≥ 0, we're actively forcing the transition sequence;
    /// indexes the next token to force. When the sequence completes, reset to -1.
    var forceIndex: Int = -1
    var budgetExhausted: Bool = false

    /// Pre-built additive masks — one per cacheable set of suppressed IDs.
    /// Replaces the scatter-per-suppressed-token loops with one elementwise
    /// add. `thinking` suppresses EOS during reasoning so the model emits a
    /// channel transition; `postExhaust` suppresses analysis-channel re-entry
    /// after the budget has forced a transition. See ``SuppressMaskCache``.
    let masks = SuppressMaskCache()

    /// Lazy accumulator of sampled content tokens awaiting a batched
    /// state-machine update. See ``ThinkingBudgetProcessor`` for the
    /// rationale; the win on GPT-OSS is larger because per-step GPU→CPU
    /// syncs block dispatch of the next forward pass on a 20B MoE.
    ///
    /// Tokens sampled while `forceIndex >= 0` are NOT queued — during an
    /// active force-emit we know each sampled ID deterministically (we
    /// pinned the sampler), so state can advance without any sync.
    var pendingTokens: [MLXArray] = []

    private static let flushBatchSize = 10

    init(
        channelMarkerTokenId: Int32,
        thinkingChannelTokenIds: Set<Int32>,
        generationChannelTokenIds: Set<Int32>,
        forcedTransitionSequence: [Int32],
        maxThinkingTokens: Int,
        eosTokenIds: [Int32] = []
    ) {
        self.channelMarkerTokenId = channelMarkerTokenId
        self.thinkingChannelTokenIds = thinkingChannelTokenIds
        self.generationChannelTokenIds = generationChannelTokenIds
        self.forcedTransitionSequence = forcedTransitionSequence
        self.maxThinkingTokens = maxThinkingTokens
        self.eosTokenIds = eosTokenIds
    }

    mutating func prompt(_ prompt: MLXArray) {
        inThinking = false
        awaitingChannelName = false
        thinkingTokenCount = 0
        forceIndex = -1
        budgetExhausted = false
        pendingTokens.removeAll(keepingCapacity: true)
    }

    func process(logits: MLXArray) -> MLXArray {
        // Lazy mask construction on first use.
        let vocab = logits.shape.last ?? 0
        if masks.thinking == nil && vocab > 0 {
            masks.thinking = buildSuppressMask(
                suppressedIds: eosTokenIds, vocabSize: vocab)
        }
        if masks.postExhaust == nil && vocab > 0 {
            masks.postExhaust = buildSuppressMask(
                suppressedIds: Array(thinkingChannelTokenIds), vocabSize: vocab)
        }

        // Active force-emit: pin the next sequence token.
        if forceIndex >= 0 && forceIndex < forcedTransitionSequence.count {
            let modified = logits
            let targetId = Int(forcedTransitionSequence[forceIndex])
            // Use a large finite value (100) — softmax(+inf) → NaN which the
            // sampler would misroute. 100 gives P(target) ≈ 1.0 with no NaN.
            if logits.ndim == 1 {
                modified[targetId] = MLXArray(Float(100.0))
            } else {
                modified[0..., targetId] = MLXArray(Float(100.0))
            }
            return modified
        }

        // After completing the forced transition, suppress analysis-channel
        // entry so the model can't oscillate back into reasoning.
        if budgetExhausted && !inThinking {
            if let m = masks.postExhaust {
                return logits + m
            }
            return logits
        }

        // During reasoning, suppress EOS so the model is forced to emit a
        // channel transition before terminating.
        guard inThinking else { return logits }
        if let m = masks.thinking {
            return logits + m
        }
        return logits
    }

    mutating func didSample(token: MLXArray) {
        // During active force-emit we know each sampled token deterministically
        // (we pinned its logit to +100). Advance state without any sync — no
        // need to queue the lazy MLXArray.
        if forceIndex >= 0 {
            forceIndex += 1
            if forceIndex >= forcedTransitionSequence.count {
                forceIndex = -1
                inThinking = false
                awaitingChannelName = false
                budgetExhausted = true
            }
            return
        }

        // Otherwise defer the sync: queue lazily and flush in batches. See
        // `ThinkingBudgetProcessor.didSample` for the full rationale.
        pendingTokens.append(token)

        let headroom = inThinking
            ? max(0, maxThinkingTokens - thinkingTokenCount)
            : Int.max
        let threshold = max(1, min(Self.flushBatchSize, headroom / 2))
        if pendingTokens.count >= threshold {
            flushPending()
        }
    }

    mutating func flushPending() {
        guard !pendingTokens.isEmpty else { return }
        eval(pendingTokens)
        let ids = pendingTokens.map { $0.item(Int32.self) }
        pendingTokens.removeAll(keepingCapacity: true)
        for id in ids {
            applyStateUpdate(id: id)
            // If the state machine armed force-emit mid-batch, stop draining
            // here — the remaining pending tokens (if any) were sampled
            // BEFORE the armed point, and the next `process()` call needs to
            // see `forceIndex >= 0` to start pinning the transition.
            //
            // Note: pendingTokens is already empty by the time we get here
            // (removeAll above), so there's nothing to preserve — the break
            // is defensive in case the loop is refactored.
            if forceIndex >= 0 { break }
        }
    }

    private mutating func applyStateUpdate(id: Int32) {
        if id == channelMarkerTokenId {
            awaitingChannelName = true
            return
        }
        if awaitingChannelName {
            awaitingChannelName = false
            if thinkingChannelTokenIds.contains(id) {
                inThinking = true
                thinkingTokenCount = 0
            } else if generationChannelTokenIds.contains(id) {
                inThinking = false
            }
            return
        }
        if inThinking {
            thinkingTokenCount += 1
            if thinkingTokenCount >= maxThinkingTokens {
                forceIndex = 0
            }
        }
    }
}

// MARK: - KV Cache Configuration

/// KV cache quantization configuration for benchmarks.
enum KVCacheConfig: CustomStringConvertible {
    case none                                       // No KV quantization
    case affine(bits: Int)                          // MLX affine quantization (kvBits)
    case turbo(bits: Int) // TurboQuant symmetric (kvScheme="turbo4")
    case turboAsym(keyBits: Int, valueBits: Int) // Asymmetric ("turbo4v2")

    var description: String {
        switch self {
        case .none: return "no-quant"
        case .affine(let b): return "affine-\(b)"
        case .turbo(let b): return "turbo\(b)"
        case .turboAsym(let kb, let vb): return "turbo\(kb)v\(vb)"
        }
    }

    var kvBits: Int? {
        if case .affine(let b) = self { return b }
        return nil
    }

    var kvScheme: String? {
        switch self {
        case .turbo(let b):
            return "turbo\(b)"
        case .turboAsym(let kb, let vb):
            return "turbo\(kb)v\(vb)"
        default: return nil
        }
    }

    var quantizedKVStart: Int {
        switch self {
        case .none: return 0
        case .affine: return 512
        case .turbo, .turboAsym: return 0
        }
    }

    /// Compute KV cache size in bytes from token count and model config.
    /// Deterministic and comparable across runs — independent of MLX memory pool.
    func cacheBytes(tokens: Int, kvHeads: Int, headDim: Int, layers: Int) -> Int {
        let perTokenPerHead: Int  // bytes for K+V per token per head
        switch self {
        case .none:
            // FP16: 2 bytes per element, K+V
            perTokenPerHead = headDim * 2 * 2  // K + V, FP16

        case .affine(let bits):
            // wq: headDim * bits / 8 bytes, scales: (headDim/64) * 4, biases: (headDim/64) * 4
            let groupSize = 64
            let groups = headDim / groupSize
            let wqBytes = headDim * bits / 8
            let metaBytes = groups * 4 * 2  // scale + bias per group, FP32
            perTokenPerHead = (wqBytes + metaBytes) * 2  // K + V

        case .turbo(let bits):
            // packed: packedWidth * 4 bytes, norm: 4 bytes
            let pw = (headDim * bits + 31) / 32
            perTokenPerHead = (pw * 4 + 4) * 2  // K + V

        case .turboAsym(let keyBits, let valueBits):
            let kpw = (headDim * keyBits + 31) / 32
            let vpw = (headDim * valueBits + 31) / 32
            perTokenPerHead = (kpw * 4 + 4) + (vpw * 4 + 4)  // K + V
        }

        return tokens * kvHeads * perTokenPerHead * layers
    }
}

// MARK: - KV Dimension Inference

/// Best-effort `(kvHeads, headDim)` inference for a loaded model, used to size
/// the wired-memory ticket precisely.
///
/// `kvHeads` comes from `KVCacheDimensionProvider` when available. `headDim` is
/// derived from the second axis of any `*.k_proj.weight` parameter
/// (`Linear(hidden, kvHeads * headDim)`). Both fall back to `nil` if probing
/// fails — `WiredMemoryUtils.estimateBudget` then uses its built-in heuristic.
func inferKVDimensions(model: any LanguageModel) -> (kvHeads: [Int]?, headDim: Int?) {
    let kvHeads = (model as? KVCacheDimensionProvider)?.kvHeads

    var headDim: Int? = nil
    for (path, array) in model.parameters().flattened() {
        guard path.hasSuffix(".k_proj.weight") || path.hasSuffix(".kProj.weight") else { continue }
        let outDim = array.shape.first ?? 0
        if outDim > 0, let firstHeadCount = kvHeads?.first, firstHeadCount > 0,
            outDim % firstHeadCount == 0
        {
            headDim = outDim / firstHeadCount
            break
        }
    }

    return (kvHeads, headDim)
}

// MARK: - Mock Tools

/// Minimal mock tool spec for tool-call benchmarking (no external dependencies).
enum MockTools {
    static func shellToolSpec() -> MLXLMCommon.ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "execute_shell",
                "description": "Run a command. Use for system tasks, file operations, data analysis, and all available CLI tools.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "command": [
                            "type": "string",
                            "description": "The command to execute"
                        ] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["command"]
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as MLXLMCommon.ToolSpec
    }
}

// MARK: - Environment

/// Centralized environment variable access for benchmarks.
/// All configuration comes from env vars set by benchmark.sh (or manually).
private enum BenchEnv {
    /// Model to benchmark — registry short name (e.g., "qwen35-0.8b"), alias (e.g., "nemotron-cascade-2"), or HF repo ID.
    static var model: String? {
        ProcessInfo.processInfo.environment["MLX_BENCH_MODEL"]
    }
    /// Benchmark method: simple, summarization, wikitext2, niah, multi-turn, tool-calling
    static var method: String {
        ProcessInfo.processInfo.environment["MLX_BENCH_METHOD"] ?? "simple"
    }
    /// Weight quantization: bf16, 8bit, 4bit (default: 4bit)
    static var quantization: String {
        ProcessInfo.processInfo.environment["MLX_BENCH_QUANT"] ?? "4bit"
    }
    /// KV cache configuration.
    static var kvConfig: KVCacheConfig {
        switch ProcessInfo.processInfo.environment["MLX_BENCH_KV"] {
        case "affine8": return .affine(bits: 8)
        case "affine4": return .affine(bits: 4)
        // Symmetric (K=V same bits)
        case "turbo8": return .turbo(bits: 8)
        case "turbo4": return .turbo(bits: 4)
        case "turbo3": return .turbo(bits: 3)
        case "turbo2": return .turbo(bits: 2)
        // Asymmetric K=8
        case "turbo8v4": return .turboAsym(keyBits: 8, valueBits: 4)
        case "turbo8v3": return .turboAsym(keyBits: 8, valueBits: 3)
        case "turbo8v2": return .turboAsym(keyBits: 8, valueBits: 2)
        // Asymmetric K=4
        case "turbo4v3": return .turboAsym(keyBits: 4, valueBits: 3)
        case "turbo4v2": return .turboAsym(keyBits: 4, valueBits: 2)
        // Asymmetric K=3
        case "turbo3v2": return .turboAsym(keyBits: 3, valueBits: 2)
        // Asymmetric K=0 (fp16 keys, compressed values only)
        case "turbo0v8": return .turboAsym(keyBits: 0, valueBits: 8)
        case "turbo0v4": return .turboAsym(keyBits: 0, valueBits: 4)
        case "turbo0v3": return .turboAsym(keyBits: 0, valueBits: 3)
        case "turbo0v2": return .turboAsym(keyBits: 0, valueBits: 2)
        default: return .none
        }
    }
    /// Context sizes to evaluate (comma-separated). Nil = use all default sizes.
    static var contexts: [Int]? {
        guard let filter = ProcessInfo.processInfo.environment["MLX_BENCH_CONTEXT"], !filter.isEmpty else {
            return nil
        }
        return filter.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
    /// Auto-select highest-fidelity variant that fits in memory.
    static var baselineMode: Bool {
        ProcessInfo.processInfo.environment["MLX_BENCH_BASELINE"] == "1"
    }
    /// Enable KL divergence computation vs bf16/8bit baseline.
    static var kldEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_BENCH_KLD"] == "1"
    }
    /// Number of concurrent generations to run (default: 1).
    static var batch: Int {
        Int(ProcessInfo.processInfo.environment["MLX_BENCH_BATCH"] ?? "1") ?? 1
    }
    /// Enable thinking mode for thinking-capable models (default: off for max speed).
    static var thinkEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_BENCH_THINK"] == "1"
    }
    /// Reasoning effort override for models that honour it (e.g. GPT-OSS).
    /// When nil, the family's default is used. Accepted values: "low",
    /// "medium", "high" — or any string the underlying chat template
    /// understands. Value validation is deliberately loose so future
    /// models' vocabularies pass through without a harness change.
    static var reasoningEffort: String? {
        guard let raw = ProcessInfo.processInfo.environment["MLX_BENCH_REASONING"] else {
            return nil
        }
        let v = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return v.isEmpty ? nil : v
    }
    /// N-gram speculative-decoding size for the benchmark run. Default is 0
    /// (disabled) so benchmarks measure pure autoregressive decode rather
    /// than a composite with variable accept-rate overhead. Set via
    /// `--ngram N` → `MLX_BENCH_NGRAM=N`.
    ///
    /// This differs from the library's `GenerateParameters.ngramSize`
    /// default (3) — the library picks a typical usable value; benchmarks
    /// prefer a clean baseline and let the user opt in to measure the
    /// speculative path explicitly.
    static var ngramSize: Int {
        guard let raw = ProcessInfo.processInfo.environment["MLX_BENCH_NGRAM"],
              let v = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              v >= 0
        else { return 0 }
        return v
    }

    /// Override the prefill chunk size from the bench harness. When unset,
    /// the iterator falls back to the model's `defaultPrefillStepSize`.
    /// `--prefill-chunk N` → `MLX_BENCH_PREFILL_CHUNK=N`. Used to sweep peak
    /// GPU vs prefill-throughput tradeoff.
    static var prefillChunkSize: Int? {
        guard let raw = ProcessInfo.processInfo.environment["MLX_BENCH_PREFILL_CHUNK"],
              let v = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              v > 0
        else { return nil }
        return v
    }
}

// MARK: - Baseline Token Data

/// Per-token data collected from baseline generation, used for in-memory KLD computation.
private struct BaselineTokenData {
    let tokenIds: [Int]
    let logProbs: [Double]
    let phases: [String]  // "think", "gen", or "marker"
}

// MARK: - Model Cache

/// Cache loaded models across test runs to avoid reloading per context size.
/// Safe because InferenceBenchmarks uses .serialized (sequential execution).
private final class ModelCache: @unchecked Sendable {
    static let shared = ModelCache()
    private var cache: [String: ModelContainer] = [:]
    private let lock = NSLock()

    func get(_ id: String) -> ModelContainer? {
        lock.lock()
        defer { lock.unlock() }
        return cache[id]
    }

    func set(_ id: String, _ container: ModelContainer) {
        lock.lock()
        defer { lock.unlock() }
        cache[id] = container
    }
}

// MARK: - Test Suite

/// Inference benchmarks for production models.
///
/// Single entry point driven by environment variables.
/// Usage: ./scripts/benchmark.sh --method <method> --model <model> [options]
/// See benchmarks/README.md for full documentation.
@Suite("Inference Benchmarks", .serialized)
struct InferenceBenchmarks {

    // MARK: - Constants

    static let contextSizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    static let toolQuery = "What is the current date and time?"
    static let multiTurnContext: [MLXLMCommon.Message] = [
        ["role": "user", "content": "Hello, what is your name?"],
        ["role": "assistant", "content": "Hello! I'm an AI assistant. How can I help you today?"],
        ["role": "user", "content": "My name is Bob and my partner's name is Alice."],
        ["role": "assistant", "content": "Nice to meet you! What can I help you with?"],
    ]
    static let multiTurnRecallTests: [(question: String, expected: String)] = [
        ("What is my name?", "Bob"),
        ("What is my partner's name?", "Alice"),
    ]
    static let minimalSystemPrompt = "You are a helpful assistant. Keep responses concise."
    static let simpleQuery = ProcessInfo.processInfo.environment["MLX_BENCH_PROMPT"] ?? "Hello! What is your name and what can you help me with?"
    /// Default context limit for non-scaling methods.
    /// Enforced via maxKVSize (RotatingKVCache) to simulate a realistic chat deployment.
    static let defaultContextLimit = 4096
    static let niahNeedle = "The special magic verification code is BLUE TIGER 42."
    static let niahAnswer = "BLUE TIGER 42"
    static let niahQuestion = "What is the special magic verification code mentioned in the text above? Reply with only the code, nothing else."
    static let niahDepths: [Double] = [0.1, 0.25, 0.5, 0.75, 0.9]

    // MARK: - Entry Point

    /// Single benchmark entry point. All configuration comes from env vars.
    /// Print build environment for debugging — confirms NAX and model template.
    /// If prefill < 10K tok/s on Gemma E2B, something is wrong.
    static func printBuildEnvironment() {
        // NAX check: if Cmlx was compiled with NAX, the .o files exist
        let naxObj = FileManager.default.fileExists(
            atPath: ".build/arm64-apple-macosx/release/Cmlx.build/mlx-generated/steel_attention_nax.cpp.o")
        print("[ENV] NAX: \(naxObj ? "ENABLED ✓" : "DISABLED ✗ — merge ekryski/mlx-swift#2, swift package clean, rebuild")")

        // Gemma 4 template check
        if let modelPath = ProcessInfo.processInfo.environment["MLX_BENCH_MODEL"],
           modelPath.lowercased().contains("gemma") {
            let tcPath = (modelPath as NSString).expandingTildeInPath + "/tokenizer_config.json"
            if let data = try? Data(contentsOf: URL(fileURLWithPath: tcPath)),
               let tc = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let tmpl = tc["chat_template"] as? String, tmpl.count > 100 {
                    print("[ENV] Gemma template: OK (\(tmpl.count) chars)")
                } else {
                    print("[ENV] Gemma template: MISSING ✗ — run: python3 scripts/fix-gemma4-template.py \(modelPath)")
                }
            }
        }
    }

    @Test @MainActor func benchmark() async throws {
        // Force line-buffered stdout so progress lines appear immediately when piped
        setlinebuf(stdout)

        Self.printBuildEnvironment()

        let family = try resolveFamily()
        let kv = BenchEnv.kvConfig
        let (variant, repoId) = try await resolveVariant(family: family)
        let method = BenchEnv.method

        switch method {
        case "simple":
            try await runGenerationBenchmark(
                family: family, variant: variant, repoId: repoId, kv: kv,
                label: "\(family.name) [\(variant.quantization)] — simple [\(kv)]",
                contextSize: Self.defaultContextLimit,
                messages: [["role": "user", "content": Self.simpleQuery]],
                systemPrompt: Self.minimalSystemPrompt, maxTokens: 200
            )

        case "summarization":
            let contexts = BenchEnv.contexts ?? Self.contextSizes
            let batch = BenchEnv.batch
            // Without thinking: match total decode cap of thinking runs (200 think + 200 answer).
            let summarizationMaxNewTokens = BenchEnv.thinkEnabled ? 200 : 400

            // ── Warmup pass: JIT Metal shaders and warm caches before timed runs ──
            // Without this, the first context size eats a cold-start penalty (shader
            // compilation, buffer allocation, Metal pipeline setup) that inflates TTFT
            // and deflates prefill tok/s. Run a short 64-token generation, sync GPU,
            // then discard. This matches llama.cpp's llama-bench warmup behavior.
            do {
                // Warmup with 512 tokens to warm Metal pipeline specializations.
                // Kernels are warm after the first ~10 tokens; 512 is enough to
                // trigger all dispatch paths without wasting time.
                // Skip warmup for turbo KV — turbo cache conversion doesn't survive
                // cross-run cleanly due to lazy eval graph interactions.
                if kv.kvScheme == nil {  // skip warmup for turbo KV
                    print("[WARMUP] Running warmup pass (512 tokens)...")
                    let warmupPrompt = try loadPrompt(tokenCount: 512)
                    try await runGenerationBenchmark(
                        family: family, variant: variant, repoId: repoId, kv: kv,
                        label: "warmup",
                        contextSize: 512,
                        messages: [["role": "user", "content": warmupPrompt]],
                        systemPrompt: nil, maxTokens: 16,
                        warmup: true
                    )
                    Stream.defaultStream(.gpu).synchronize()
                    MLX.Memory.clearCache()
                    TurboQuantKVCache.clearCodecCache()
                    print("[WARMUP] Done — Metal pipeline hot\n")
                } else {
                    print("[WARMUP] Skipped for TurboQuant KV\n")
                }
            }

            for (idx, ctx) in contexts.enumerated() {
                print("[PROGRESS] Context \(idx + 1)/\(contexts.count): \(ctx) tokens")
                let prompt = try loadPrompt(tokenCount: ctx)
                if batch > 1 {
                    try await runBatchedBenchmark(
                        batchSize: batch,
                        family: family, variant: variant, repoId: repoId, kv: kv,
                        label: "\(family.name) [\(variant.quantization)] — summarization \(ctx) [\(kv)] batch=\(batch)",
                        contextSize: ctx,
                        messages: [["role": "user", "content": prompt]],
                        systemPrompt: nil, maxTokens: summarizationMaxNewTokens
                    )
                } else {
                    try await runGenerationBenchmark(
                        family: family, variant: variant, repoId: repoId, kv: kv,
                        label: "\(family.name) [\(variant.quantization)] — summarization \(ctx) [\(kv)]",
                        contextSize: ctx,
                        messages: [["role": "user", "content": prompt]],
                        systemPrompt: nil, maxTokens: summarizationMaxNewTokens
                    )
                }
                // Release GPU buffers between contexts to prevent Invalid Resource
                // errors from Metal buffer reuse on large models (18+ GB).
                Stream.defaultStream(.gpu).synchronize()
                MLX.Memory.clearCache()
            }

        case "wikitext2":
            let contexts = BenchEnv.contexts ?? Self.contextSizes
            for (idx, ctx) in contexts.enumerated() {
                print("[PROGRESS] Context \(idx + 1)/\(contexts.count): \(ctx) tokens")
                try await runWikitext2Benchmark(family: family, kv: kv, contextSize: ctx)
            }

        case "niah":
            let contexts = BenchEnv.contexts ?? Self.contextSizes
            for (idx, ctx) in contexts.enumerated() {
                print("[PROGRESS] Context \(idx + 1)/\(contexts.count): \(ctx) tokens")
                try await runNIAHBenchmark(family: family, kv: kv, contextSize: ctx)
            }

        case "multi-turn":
            for (question, expected) in Self.multiTurnRecallTests {
                var messages = Self.multiTurnContext
                messages.append(["role": "user", "content": question])
                try await runGenerationBenchmark(
                    family: family, variant: variant, repoId: repoId, kv: kv,
                    label: "\(family.name) [\(variant.quantization)] — multi-turn(\(expected)) [\(kv)]",
                    contextSize: Self.defaultContextLimit,
                    messages: messages,
                    systemPrompt: Self.minimalSystemPrompt, maxTokens: 200,
                    validation: { output, _ in
                        output.lowercased().contains(expected.lowercased())
                            ? "PASS(\(expected)): " : "FAIL(missing \(expected)): "
                    }
                )
            }

        case "tool-calling":
            try await runGenerationBenchmark(
                family: family, variant: variant, repoId: repoId, kv: kv,
                label: "\(family.name) [\(variant.quantization)] — tool [\(kv)]",
                contextSize: Self.defaultContextLimit,
                messages: [["role": "user", "content": Self.toolQuery]],
                systemPrompt: Self.minimalSystemPrompt, maxTokens: 200,
                includeTools: true,
                validation: { _, toolCalls in
                    guard let tc = toolCalls.first else {
                        return "FAIL(no tool call): "
                    }
                    guard tc.function.name == "execute_shell" else {
                        return "FAIL(wrong tool: \(tc.function.name)): "
                    }
                    let cmdArg = tc.function.arguments["command"]
                    let cmdStr = cmdArg.flatMap { "\($0)" } ?? ""
                    guard cmdStr.lowercased().contains("date") else {
                        return "FAIL(wrong command: \(cmdStr)): "
                    }
                    return "PASS: "
                }
            )

        case "raw-prefill":
            let contexts = BenchEnv.contexts ?? Self.contextSizes
            for (idx, ctx) in contexts.enumerated() {
                print("[PROGRESS] Context \(idx + 1)/\(contexts.count): \(ctx) tokens")
                try await runRawPrefillBenchmark(family: family, kv: kv, contextSize: ctx)
            }

        default:
            print("[BENCH] Unknown method: \(method)")
        }
    }

    // MARK: - Family Resolution

    /// Resolve model family from MLX_BENCH_MODEL env var (short name, alias, or HF repo id).
    private func resolveFamily() throws -> ModelFamily {
        guard let name = BenchEnv.model else {
            throw BenchmarkError("MLX_BENCH_MODEL not set — use --model flag")
        }
        if let family = ModelRegistry.family(named: name) {
            return family
        }
        // If it doesn't contain '/' it's not a HF repo ID — the user likely mistyped a short name
        if !name.contains("/") {
            let available = ModelRegistry.allFamilies.map(\.shortName)
            let close = available.filter { $0.hasPrefix(name) || name.hasPrefix($0) }
            var msg = "Unknown model '\(name)'."
            if !close.isEmpty {
                msg += " Did you mean: \(close.joined(separator: ", "))?"
            }
            msg += "\nAvailable models: \(available.joined(separator: ", "))"
            throw BenchmarkError(msg)
        }
        // Treat as HuggingFace repo ID (custom model)
        return ModelRegistry.customFamily(repoId: name)
    }

    // MARK: - Variant Resolution

    /// Resolve which model variant to use based on env vars (baseline mode, quant selection).
    /// Pre-checks estimated model size against GPU memory and warns if it may not fit.
    private func resolveVariant(family: ModelFamily) async throws -> (ModelVariant, String) {
        if BenchEnv.baselineMode {
            let hw = SystemInfo.hardware()
            let variant = try await family.selectBaseline(hardware: hw)
            return (variant, variant.repoId)
        }

        let quant = BenchEnv.quantization
        guard let variant = family.variant(for: quant) else {
            let fallback = family.variants[0]
            print("[BENCH] No \(quant) variant for \(family.name), using \(fallback.quantization)")
            return (fallback, fallback.repoId)
        }

        // Pre-check: estimate if the model fits in GPU memory
        let hw = SystemInfo.hardware()
        do {
            let size = try await SystemInfo.estimateModelSize(repo: variant.repoId)
            if !SystemInfo.fitsInMemory(modelSizeBytes: size, hardware: hw) {
                throw BenchmarkError(
                    "\(family.name) \(quant) (~\(SystemInfo.formatGB(size))) exceeds GPU memory limit "
                    + "(\(String(format: "%.0f", hw.gpuMemoryLimitGB))GB). "
                    + "Use --baseline to auto-select a smaller variant, or specify --quant 8bit/4bit."
                )
            }
        } catch let error as BenchmarkError {
            throw error  // re-throw our own errors
        } catch {
            print("[BENCH] Could not estimate model size for \(variant.repoId): \(error)")
        }

        return (variant, variant.repoId)
    }

    // MARK: - Core Benchmark Runner

    /// Validation closure: receives (outputText, toolCalls) → returns status prefix for output column.
    /// Return nil for no validation, or e.g. "PASS: " / "FAIL: ".
    typealias ValidationCheck = @Sendable (_ output: String, _ toolCalls: [ToolCall]) -> String?

    private func runGenerationBenchmark(
        family: ModelFamily,
        variant: ModelVariant,
        repoId: String,
        kv: KVCacheConfig,
        label: String,
        contextSize: Int,
        messages: [MLXLMCommon.Message],
        systemPrompt: String?,
        maxTokens: Int = Int(ProcessInfo.processInfo.environment["MLX_BENCH_MAX_TOKENS"].flatMap { Int($0) } ?? 200),
        includeTools: Bool = false,
        validation: ValidationCheck? = nil,
        warmup: Bool = false
    ) async throws {
        // Lifecycle profile mode. Two levels:
        //
        // - `MLX_BENCH_PROFILE=1` — captures wall-clock timestamps at each
        //   phase boundary and prints an inline [PROFILE] breakdown at
        //   the end (model load, prompt prep, prefill, first-token
        //   warmup, steady-state decode). No external tools required.
        //
        // - `MLX_BENCH_PROFILE=2` — everything at level 1 PLUS
        //   `os_signpost` intervals at every phase boundary, opt-in
        //   captured by Instruments / xctrace. See
        //   `BenchmarkSignpost.swift` for the recording procedure.
        //   Overhead is ~40 ns per event when no tracer is attached.
        //   Includes TurboQuant compressedAttention sub-phases
        //   (`tq_encode`, `tq_score`, `tq_softmax`, `tq_value`,
        //   `tq_rotate`) — correlate with MLX's per-kernel signposts
        //   (subsystem `ai.mlx.metal`) in Metal System Trace to
        //   attribute GPU time per phase.
        //
        // Level 1 only adds a few timestamps + post-run arithmetic →
        // zero measurable impact on decode throughput. Level 2 adds
        // signpost begin/end pairs per decode step; overhead is well
        // under 50 µs for a 200-token run and is compiled out of the
        // hot path when Instruments isn't recording.
        let profileLevel: Int = {
            guard let raw = ProcessInfo.processInfo.environment["MLX_BENCH_PROFILE"],
                  let level = Int(raw) else { return 0 }
            return level
        }()
        let profileEnabled = profileLevel >= 1
        let benchEnterTime = CFAbsoluteTimeGetCurrent()

        let hr = String(repeating: "=", count: 80)
        print("\n\(hr)")
        print("[BENCH] \(label)")
        print("[BENCH] Model: \(repoId)")
        print("[BENCH] Quantization: \(variant.quantization)")
        print("[BENCH] KV: \(kv)")
        print(hr)

        // Thinking is only active when BOTH the model supports it AND --think flag is set
        let useThinking = family.supportsThinking && BenchEnv.thinkEnabled

        let thinkingBudget = 200  // max thinking tokens before forcing </think>

        // ── 1. Build user input (no container needed yet) ─────────────────────
        var allMessages: [MLXLMCommon.Message] = []
        if let sys = systemPrompt {
            allMessages.append(["role": "system", "content": sys])
        }
        allMessages.append(contentsOf: messages)
        // Force thinking mode via assistant prefill (Qwen-style: prefill with <think>\n)
        if useThinking && !family.thinkingConfig.assistantPrefill.isEmpty {
            allMessages.append(["role": "assistant", "content": family.thinkingConfig.assistantPrefill])
        }
        let tools: [MLXLMCommon.ToolSpec]? = includeTools ? [MockTools.shellToolSpec()] : nil
        // Pass enable_thinking to the chat template for models that support it (Qwen, Gemma 4)
        let additionalContext: [String: any Sendable]? = useThinking
            ? ["enable_thinking": true] : nil
        let userInput = UserInput(
            prompt: .messages(allMessages), tools: tools,
            additionalContext: additionalContext)

        // ── 2. Load target model (cached across context sizes) ──────────────────
        let loadStart = CFAbsoluteTimeGetCurrent()
        let wasCached = ModelCache.shared.get(repoId) != nil
        let loadHandle = BenchmarkSignpost.begin(
            BenchmarkSignpost.PhaseLabel.modelLoad,
            metadata: wasCached ? "cache_hit" : "cold:\(repoId)")
        let container = try await loadOrCacheModel(family: family, repoId: repoId)
        BenchmarkSignpost.end(loadHandle)
        let loadEnd = CFAbsoluteTimeGetCurrent()
        if profileEnabled {
            let loadMs = (loadEnd - loadStart) * 1000
            print(String(format: "[PROFILE] model_load: %.0fms%@",
                loadMs, wasCached ? " (cache hit)" : " (cold)"))
        }

        // ── 4. Discover thinking tokens with target container ─────────────────
        //
        // Two styles per ThinkingConfig:
        //   • bracket       — resolve a single start + end token pair (Qwen, Gemma 4).
        //   • harmonyChannel — resolve the channel marker plus each channel-name
        //                     token so phase labeling can run the harmony SM
        //                     (GPT-OSS analysis vs final/commentary).
        let thinkingConfig = family.thinkingConfig
        struct ThinkingTokens {
            var thinkStartId: Int32?
            var thinkEndId: Int32?
            var harmonyMarkerId: Int32?
            var harmonyThinkingIds: [Int32] = []
            var harmonyGenerationIds: [Int32] = []
            var harmonyTransitionIds: [Int32] = []
            var eosTokenIds: [Int32] = []
        }
        let thinkingTokenIds: ThinkingTokens = useThinking
            ? await container.perform { ctx in
                var result = ThinkingTokens()
                // Collect all EOS token IDs so we can suppress them during thinking phase
                var eosIds = ctx.configuration.eosTokenIds.map { Int32($0) }
                if let tokEos = ctx.tokenizer.eosTokenId { eosIds.append(Int32(tokEos)) }
                for token in ctx.configuration.extraEOSTokens {
                    if let id = ctx.tokenizer.convertTokenToId(token) { eosIds.append(Int32(id)) }
                }
                result.eosTokenIds = Array(Set(eosIds))

                switch thinkingConfig.style {
                case .bracket(let start, let end):
                    result.thinkStartId = ctx.tokenizer.convertTokenToId(start).map { Int32($0) }
                    result.thinkEndId = ctx.tokenizer.convertTokenToId(end).map { Int32($0) }
                case .harmonyChannel(let marker, let thinkingChans, let genChans, let transitionSeq):
                    result.harmonyMarkerId = ctx.tokenizer.convertTokenToId(marker).map { Int32($0) }
                    result.harmonyThinkingIds = thinkingChans.compactMap {
                        ctx.tokenizer.convertTokenToId($0).map { Int32($0) }
                    }
                    result.harmonyGenerationIds = genChans.compactMap {
                        ctx.tokenizer.convertTokenToId($0).map { Int32($0) }
                    }
                    // The forced-transition sequence is only useful if every
                    // token resolves; a missing ID would derail the force SM.
                    let resolved = transitionSeq.compactMap {
                        ctx.tokenizer.convertTokenToId($0).map { Int32($0) }
                    }
                    result.harmonyTransitionIds = resolved.count == transitionSeq.count ? resolved : []
                }
                return result
              }
            : ThinkingTokens()

        let thinkStartId = thinkingTokenIds.thinkStartId
        let thinkEndId = thinkingTokenIds.thinkEndId
        let harmonyMarkerId = thinkingTokenIds.harmonyMarkerId
        let harmonyThinkingIds = thinkingTokenIds.harmonyThinkingIds
        let harmonyGenerationIds = thinkingTokenIds.harmonyGenerationIds
        let harmonyTransitionIds = thinkingTokenIds.harmonyTransitionIds
        let eosTokenIds = thinkingTokenIds.eosTokenIds

        if let s = thinkStartId, let e = thinkEndId {
            if case .bracket(let startStr, let endStr) = thinkingConfig.style {
                print("[BENCH] Thinking tokens: \(startStr)=\(s), \(endStr)=\(e), budget=\(thinkingBudget), eos=\(eosTokenIds)")
            }
        } else if let m = harmonyMarkerId {
            print("[BENCH] Harmony channel marker=\(m), thinking=\(harmonyThinkingIds), generation=\(harmonyGenerationIds), transition=\(harmonyTransitionIds), eos=\(eosTokenIds)")
        }
        let thinkingPrefilled = !family.thinkingConfig.assistantPrefill.isEmpty
        let bracketBudgetProcessor: ThinkingBudgetProcessor? = thinkStartId.flatMap { startId in
            thinkEndId.map { endId in
                ThinkingBudgetProcessor(
                    thinkStartTokenId: startId,
                    thinkEndTokenId: endId,
                    maxThinkingTokens: thinkingBudget,
                    prefilled: thinkingPrefilled,
                    eosTokenIds: eosTokenIds
                )
            }
        }
        // Harmony equivalent: enforce the thinking budget by force-emitting
        // the channel-transition sequence after N analysis-message tokens.
        // Only constructed when every transition-sequence token resolved.
        let harmonyBudgetProcessor: HarmonyThinkingBudgetProcessor? = {
            guard let markerId = harmonyMarkerId,
                  !harmonyTransitionIds.isEmpty,
                  !harmonyGenerationIds.isEmpty
            else { return nil }
            return HarmonyThinkingBudgetProcessor(
                channelMarkerTokenId: markerId,
                thinkingChannelTokenIds: Set(harmonyThinkingIds),
                generationChannelTokenIds: Set(harmonyGenerationIds),
                forcedTransitionSequence: harmonyTransitionIds,
                maxThinkingTokens: thinkingBudget,
                eosTokenIds: eosTokenIds
            )
        }()

        // ── 5. Prepare input ──────────────────────────────────────────────────
        let prepareStart = Date()
        let promptPrepHandle = BenchmarkSignpost.begin(
            BenchmarkSignpost.PhaseLabel.promptPrep)
        var lmInput: LMInput
        do {
            lmInput = try await container.prepare(input: userInput)
        } catch {
            // Some models (e.g. Mistral) don't support system messages or tools.
            // Retry with just the user messages, no system prompt, no tools.
            let userOnly = allMessages.filter { ($0["role"] as? String) != "system" }
            print("[BENCH] Template error: \(error). Retrying without system/tools...")
            let fallbackInput = UserInput(prompt: .messages(userOnly))
            lmInput = try await container.prepare(input: fallbackInput)
        }
        var promptTokens = lmInput.text.tokens.dim(lmInput.text.tokens.ndim - 1)

        // Trim prompt to fit context limit. Pre-built prompt files are sized with a
        // reference tokenizer; the target model's tokenizer + chat template overhead
        // can push the actual token count above contextSize, causing the rotating KV
        // cache to silently drop the first tokens. Fix: trim the raw user content
        // from the end (preserving the instruction prefix) and re-prepare.
        if contextSize > 0 && promptTokens > contextSize {
            let overshoot = promptTokens - contextSize
            let messagesToTrim = allMessages
            let trimmedMessages: [MLXLMCommon.Message] = await container.perform { ctx in
                var result = messagesToTrim
                if let lastUserIdx = result.lastIndex(where: { $0["role"] as? String == "user" }),
                   let content = result[lastUserIdx]["content"] as? String
                {
                    let tokens = ctx.tokenizer.encode(text: content)
                    if tokens.count > overshoot {
                        let trimmedTokens = Array(tokens.prefix(tokens.count - overshoot))
                        let trimmedContent = ctx.tokenizer.decode(tokenIds: trimmedTokens)
                        result[lastUserIdx] = ["role": "user", "content": trimmedContent]
                    }
                }
                return result
            }
            let trimmedInput = UserInput(
                prompt: .messages(trimmedMessages), tools: tools,
                additionalContext: additionalContext)
            lmInput = try await container.prepare(input: trimmedInput)
            promptTokens = lmInput.text.tokens.dim(lmInput.text.tokens.ndim - 1)
            print("[BENCH] Trimmed prompt to \(promptTokens) tokens (context limit: \(contextSize))")
        }

        let prepareEndDate = Date()
        BenchmarkSignpost.end(
            promptPrepHandle,
            metadata: "prompt_tokens=\(promptTokens)")
        print("[BENCH] Prepared \(promptTokens) tokens in \(String(format: "%.0f", prepareEndDate.timeIntervalSince(prepareStart) * 1000))ms")

        // ── 6. Generate ───────────────────────────────────────────────────────
        // effectiveMaxTokens = thinking budget + response budget for thinking models.
        // Bracket mode (Qwen, Gemma 4) keys off thinkStartId; harmony mode
        // (GPT-OSS) keys off harmonyMarkerId. Either path reserves the same
        // thinkingBudget tokens for reasoning before the response budget.
        let hasThinkingBudget = thinkStartId != nil || harmonyMarkerId != nil
        let effectiveMaxTokens = hasThinkingBudget ? thinkingBudget + maxTokens : maxTokens
        // Exactly one budget enforcer applies per run: bracket XOR harmony.
        // Both nil → model runs unconstrained.
        var additionalProcessors: [any LogitProcessor] = []
        if let p = bracketBudgetProcessor { additionalProcessors.append(p) }
        if let p = harmonyBudgetProcessor { additionalProcessors.append(p) }

        // When KLD is enabled, collect per-token data during generation so we can
        // compare against the bf16/8bit baseline (no KV quant) via forced decode.
        // Only skip for bf16 + no KV quant (that IS the baseline).
        let isKVQuantized: Bool
        if case .none = kv { isKVQuantized = false } else { isKVQuantized = true }
        let needsKLD = BenchEnv.kldEnabled && (variant.quantization != "bf16" || isKVQuantized)

        let params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
            maxKVSize: contextSize > 0 ? contextSize : nil,
            kvBits: kv.kvBits,
            kvGroupSize: 64,
            quantizedKVStart: kv.quantizedKVStart,
            temperature: family.temperature,
            topP: family.topP,
            topK: family.topK,
            minP: family.minP,
            repetitionPenalty: family.repetitionPenalty,
            presencePenalty: family.presencePenalty,
            prefillStepSize: BenchEnv.prefillChunkSize,

            kvScheme: warmup ? nil : kv.kvScheme,
            additionalProcessors: additionalProcessors,
            reasoningEffort: BenchEnv.reasoningEffort ?? family.reasoningEffort,
            ngramSize: BenchEnv.ngramSize,
            maxNgramDraftTokens: BenchEnv.ngramSize,
            thinkStartTokenId: thinkStartId,
            thinkEndTokenId: thinkEndId,
            thinkingPhasePrefilled: thinkStartId != nil && !family.thinkingConfig.assistantPrefill.isEmpty,
            harmonyChannelMarkerTokenId: harmonyMarkerId,
            harmonyThinkingChannelTokenIds: harmonyThinkingIds,
            harmonyGenerationChannelTokenIds: harmonyGenerationIds,
            collectPerTokenData: needsKLD,
            trackPerplexity: ProcessInfo.processInfo.environment["MLX_BENCH_PPL"] == "1"
        )

        let maxTokens = contextSize > 0 ? contextSize + effectiveMaxTokens : 4096
        let ticket = await container.perform { ctx in
            let dims = inferKVDimensions(model: ctx.model)
            return WiredMemoryUtils.resolveTicket(
                model: ctx.model,
                maxTokens: maxTokens,
                parameters: params,
                batchSize: 1,
                kvHeadsOverride: dims.kvHeads,
                headDimOverride: dims.headDim
            )
        }
        print("[BENCH] Wired ticket: \(ticket.size / 1_048_576)MB for batch=1 maxTokens=\(maxTokens)")

        // Sync GPU before timing to flush any pending lazy eval from setup
        Stream.defaultStream(.gpu).synchronize()

        // Memory breakdown before generation
        let preGenActive = MLX.Memory.activeMemory
        let preGenCache = MLX.Memory.cacheMemory
        print("[MEM] Pre-generation: active=\(preGenActive / 1_048_576)MB cache=\(preGenCache / 1_048_576)MB")

        MLX.GPU.resetPeakMemory()
        let baselineGPU = MLX.Memory.activeMemory

        let genStart = Date()
        var tokenCount = 0
        var firstTokenTime: TimeInterval? = nil
        var outputText = ""
        var completionInfo: GenerateCompletionInfo? = nil
        var toolCalls: [ToolCall] = []

        // Per-token arrival times for warmup-vs-steady-state profile.
        // Always populated — one Date() + append per token is negligible
        // overhead, and we feed it into the benchmark row's Steady tok/s
        // column so warmup-induced regressions don't hide in the
        // Generation tok/s average.
        var tokenArrivalOffsets: [TimeInterval] = []
        tokenArrivalOffsets.reserveCapacity(512)

        // Signpost interval for prefill + first decoded token. Metal's
        // async generate stream buries both behind a single "first
        // chunk arrives" boundary, so we wrap from genStart to the
        // first yielded chunk. Subsequent tokens get their own
        // `decode_step` intervals below, bracketed around each
        // iteration boundary.
        let prefillHandle = BenchmarkSignpost.begin(
            BenchmarkSignpost.PhaseLabel.prefill,
            metadata: "prompt_tokens=\(promptTokens)")
        var decodeStepHandle: BenchmarkSignpost.IntervalHandle? = nil

        let stream = try await container.generate(input: lmInput, parameters: params, wiredMemoryTicket: ticket)
        for try await generation in stream {
            if case .info(let info) = generation {
                completionInfo = info
                continue
            }
            if case .toolCall(let tc) = generation {
                toolCalls.append(tc)
                print("[BENCH] Tool call: \(tc.function.name)(\(tc.function.arguments))")
                continue
            }
            guard generation.chunk != nil else { continue }

            // Close the previous in-flight interval (prefill on the
            // first iteration, decode_step on subsequent ones). The
            // close boundary is "a new chunk has just surfaced on
            // the Swift side", which includes GPU dispatch +
            // completion + Swift bridge + tokenizer decode.
            if let h = decodeStepHandle {
                BenchmarkSignpost.end(
                    h, metadata: "tokens_so_far=\(tokenCount + 1)")
                decodeStepHandle = nil
            } else {
                // First chunk = first token produced. Close the
                // prefill interval and emit a single-point event to
                // mark TTFT for easy filtering in Instruments.
                BenchmarkSignpost.end(
                    prefillHandle, metadata: "first_token=true")
                BenchmarkSignpost.event(
                    "first_token",
                    metadata: "ttft_ms=\(Int(Date().timeIntervalSince(genStart) * 1000))")
            }

            tokenArrivalOffsets.append(Date().timeIntervalSince(genStart))
            tokenCount += 1
            outputText += generation.chunk ?? ""

            if firstTokenTime == nil {
                firstTokenTime = Date().timeIntervalSince(genStart)
            }

            // Open a fresh decode_step interval for the NEXT token
            // — spans from now (just received token N) to the
            // arrival of token N+1, i.e. the full generator round
            // trip for producing the next token.
            decodeStepHandle = BenchmarkSignpost.begin(
                BenchmarkSignpost.PhaseLabel.decodeStep,
                metadata: "token_idx=\(tokenCount)")
        }
        // Stream terminated — close the trailing decode_step handle
        // (if any) with a sentinel "eos" label so the last interval
        // doesn't dangle in the trace.
        if let h = decodeStepHandle {
            BenchmarkSignpost.end(h, metadata: "eos")
        }

        let totalTime = Date().timeIntervalSince(genStart)
        let ttft = firstTokenTime ?? totalTime

        // Warmup: we only needed to push through the Metal pipeline. Skip reporting.
        if warmup {
            MLX.Memory.clearCache()
            return
        }

        let generationTime = totalTime - ttft
        let genTokPerSec = generationTime > 0 ? Double(tokenCount - 1) / generationTime : 0

        let prefillTime = completionInfo?.promptTime ?? ttft
        let prefillTokens = completionInfo?.promptTokenCount ?? promptTokens
        let prefillTokPerSec = prefillTime > 0 ? Double(prefillTokens) / prefillTime : 0

        let peakGPU = MLX.Memory.peakMemory
        let activeGPU = MLX.Memory.activeMemory
        let kvDelta = activeGPU > baselineGPU ? activeGPU - baselineGPU : 0

        // Memory breakdown
        let postGenCache = MLX.Memory.cacheMemory
        print("[MEM] Post-generation: active=\(activeGPU / 1_048_576)MB cache=\(postGenCache / 1_048_576)MB peak=\(peakGPU / 1_048_576)MB")
        // Sync GPU before clearing buffer pool — prevents Invalid Resource errors
        // when buffers are freed while command buffers still reference them.
        Stream.defaultStream(.gpu).synchronize()
        MLX.Memory.clearCache()
        let postClearActive = MLX.Memory.activeMemory
        print("[MEM] After clearCache: active=\(postClearActive / 1_048_576)MB (KV+weights delta: \((Int(postClearActive) - Int(preGenActive)) / 1_048_576)MB)")

        // KV cache size comes from the runtime iterator via
        // `GenerateCompletionInfo.kvCacheBytes` — sum of every live per-layer
        // `KVCache.memoryBytes`, captured inside generate() just before the
        // `.info` event fires. No bench-side cache reference, no extra GPU
        // work, no perturbation of peak / prefill / decode measurements.
        // Correctly reflects whatever the runtime actually swapped to
        // (`KVCacheSimple → QuantizedKVCache`, TurboQuant compression,
        // `RotatingKVCache` for sliding-window layers, KV sharing, etc.).
        //
        // Previously this was an analytical estimate from token count ×
        // hardcoded (kvHeads=16, headDim=128, layers=28). Those constants
        // only matched Qwen-style models and silently over/understated for
        // anything else (e.g. Gemma 4 26B A4B with sliding+full layer mix
        // and head_dim=256).
        let kvCacheBytes = completionInfo?.kvCacheBytes ?? 0

        let thinkingPerplexity = completionInfo?.thinkingPerplexity.map { Double($0) }
        let generationPerplexity = completionInfo?.generationPerplexity.map { Double($0) }

        // ── 7. [KLD] Forced decode through bf16/8bit baseline to get baseline logprobs ──
        // The normal generation (step 6) ran with the target model (weight quant + KV quant)
        // and collected per-token logprobs. Now we feed the SAME tokens through the highest-
        // fidelity baseline model (bf16 → 8bit fallback) WITHOUT KV quantization. KLD measures
        // the total quality cost of the deployment config vs the gold standard.
        var thinkKLD: Double? = nil
        var genKLD: Double? = nil
        if needsKLD,
           let quantizedIds = completionInfo?.perTokenIds,
           let quantizedLogProbs = completionInfo?.perTokenLogProbs,
           let quantizedPhases = completionInfo?.perTokenPhases,
           !quantizedIds.isEmpty
        {
            let quantizedData = BaselineTokenData(
                tokenIds: quantizedIds,
                logProbs: quantizedLogProbs.map { Double($0) },
                phases: quantizedPhases
            )

            // Select and load the highest-fidelity baseline variant (bf16 → 8bit → 4bit)
            let hw = SystemInfo.hardware()
            if let baselineVariant = try? await family.selectBaseline(hardware: hw),
               // Skip if the baseline would be the same model+config as the target
               // (e.g., 8bit fallback baseline vs 8bit target with no KV quant)
               !(baselineVariant.quantization == variant.quantization && !isKVQuantized)
            {
                print("[KLD] Loading baseline \(baselineVariant.quantization): \(baselineVariant.repoId)")
                let baselineConfig = family.extraEOSTokens.isEmpty
                    ? ModelConfiguration(id: baselineVariant.repoId)
                    : ModelConfiguration(id: baselineVariant.repoId, extraEOSTokens: Set(family.extraEOSTokens))

                let baselineContainer = try await LLMModelFactory.shared.loadContainer(
                    from: benchmarkDownloader,
                    using: benchmarkTokenizerLoader,
                    configuration: baselineConfig,
                    progressHandler: { p in
                        if p.fractionCompleted < 0.01 || p.fractionCompleted > 0.99 {
                            print("[KLD] Loading baseline: \(String(format: "%.0f", p.fractionCompleted * 100))%")
                        }
                    }
                )

                // Prepare input with the baseline model's tokenizer
                let kldInput = try await baselineContainer.prepare(
                    input: UserInput(prompt: .messages(allMessages), tools: tools))

                (thinkKLD, genKLD) = try await forcedDecodeKLD(
                    container: baselineContainer,
                    input: kldInput,
                    family: family,
                    quantizedData: quantizedData,
                    thinkStartId: thinkStartId,
                    thinkEndId: thinkEndId,
                    thinkingBudget: thinkingBudget,
                    maxTokens: maxTokens
                )

                // baselineContainer goes out of scope → freed
                MLX.Memory.clearCache()
            } else {
                print("[KLD] Could not select baseline variant, skipping KLD")
            }
        }

        // ── 8. Validation ──────────────────────────────────────────────────────
        let validationPrefix = validation?(outputText, toolCalls)
        let reportOutput = (validationPrefix ?? "") + outputText
        if let prefix = validationPrefix {
            print("[BENCH] Validation: \(prefix.trimmingCharacters(in: .whitespaces))")
        }

        // ── 9. Report ─────────────────────────────────────────────────────────
        let scenario = BenchEnv.method

        print("\n[BENCH] === RESULTS: \(label) ===")
        print("[BENCH] Method: \(scenario)")
        print("[BENCH] Context: \(contextSize) tokens, Prompt Tokens: \(prefillTokens) (after template)")
        print("[BENCH] Prefill: \(String(format: "%.1f", prefillTokPerSec)) tok/s")
        print("[BENCH] Generation: \(String(format: "%.1f", genTokPerSec)) tok/s (\(tokenCount) tokens)")
        print("[BENCH] TTFT: \(String(format: "%.0f", ttft * 1000))ms")
        print("[BENCH] Total: \(String(format: "%.1f", totalTime))s")
        if let ppl = thinkingPerplexity {
            print("[BENCH] Think PPL: \(String(format: "%.4f", ppl))")
        }
        if let ppl = generationPerplexity {
            print("[BENCH] Gen PPL: \(String(format: "%.4f", ppl))")
        }
        if let k = thinkKLD { print("[KLD] Think KLD: \(String(format: "%.6f", k))") }
        if let k = genKLD { print("[KLD] Gen KLD: \(String(format: "%.6f", k))") }
        print("[BENCH] GPU Baseline: \(formatBytes(baselineGPU))")
        print("[BENCH] GPU Peak: \(formatBytes(peakGPU))")
        print("[BENCH] KV Delta: \(formatBytes(kvDelta))")
        if kvCacheBytes > 0 {
            print("[BENCH] KV Cache: \(formatBytes(kvCacheBytes))")
        }
        print("[BENCH] Output: \(String(outputText.prefix(150)))")

        // Per-token timing split: warmup (tokens 2..4) vs steady (tokens 11..end).
        // Always computed — tokenArrivalOffsets is collected unconditionally and
        // the arithmetic is trivial. Feeds both the `[PROFILE]` inline block
        // (when MLX_BENCH_PROFILE=1) and the Steady tok/s column in the
        // benchmark markdown.
        var warmupAvgMs: Double? = nil
        var steadyAvgMs: Double? = nil
        if tokenArrivalOffsets.count >= 2 {
            // Consecutive diffs give per-token intervals.
            var deltas: [Double] = []
            deltas.reserveCapacity(tokenArrivalOffsets.count)
            deltas.append(tokenArrivalOffsets[0])  // first arrival from genStart = TTFT
            for i in 1..<tokenArrivalOffsets.count {
                deltas.append(tokenArrivalOffsets[i] - tokenArrivalOffsets[i - 1])
            }
            // Warmup: tokens 2..4 (index 1..3). Skip token 1 (= TTFT,
            // includes prefill).
            let warmupRange = 1..<min(4, deltas.count)
            if !warmupRange.isEmpty {
                let warmupSum = deltas[warmupRange].reduce(0, +)
                warmupAvgMs = warmupSum / Double(warmupRange.count) * 1000
            }
            // Steady state: tokens 11..end (index 10..).
            if deltas.count > 10 {
                let steadySum = deltas[10..<deltas.count].reduce(0, +)
                steadyAvgMs = steadySum / Double(deltas.count - 10) * 1000
            }
        }
        let steadyTokPerSec: Double? = steadyAvgMs.flatMap { $0 > 0 ? 1000.0 / $0 : nil }

        if profileEnabled {
            // Full-lifecycle breakdown: model load → prompt prep → prefill →
            // first-token (includes warmup: kernel JIT + pipeline creation)
            // → steady-state decode. Numbers come from timestamps captured
            // earlier in this function; no extra syncs. Prints at the very
            // end so the inline [PROFILE] breadcrumbs and this summary
            // co-locate.
            let benchTotalMs = (CFAbsoluteTimeGetCurrent() - benchEnterTime) * 1000
            let loadMs = (loadEnd - loadStart) * 1000
            let promptPrepMs = prepareEndDate.timeIntervalSince(prepareStart) * 1000
            let ttftMs = ttft * 1000
            let prefillMs = (completionInfo?.promptTime ?? ttft) * 1000
            let firstTokenOverheadMs = ttftMs - prefillMs
            let genMs = (totalTime - ttft) * 1000

            print("")
            print("[PROFILE] ── Lifecycle breakdown ───────────────────────────────")
            print(String(format: "[PROFILE] model_load                : %7.1f ms  %@",
                loadMs, wasCached ? "(cache hit)" : "(cold)"))
            print(String(format: "[PROFILE] prompt_prep               : %7.1f ms  (tokenize + template)",
                promptPrepMs))
            print(String(format: "[PROFILE] prefill                   : %7.1f ms  (%d tokens @ %.1f tok/s)",
                prefillMs, prefillTokens, prefillTokPerSec))
            print(String(format: "[PROFILE] first_token_overhead     : %7.1f ms  (TTFT − prefill: kernel JIT + first decode)",
                firstTokenOverheadMs))
            print(String(format: "[PROFILE] ttft                      : %7.1f ms",
                ttftMs))
            if let w = warmupAvgMs {
                print(String(format: "[PROFILE] decode_warmup_per_token  : %7.2f ms  (tokens 2..4 avg)", w))
            }
            if let s = steadyAvgMs {
                print(String(format: "[PROFILE] decode_steady_per_token  : %7.2f ms  (tokens 11..end avg) = %.1f tok/s",
                    s, s > 0 ? 1000.0 / s : 0))
            }
            print(String(format: "[PROFILE] generation_total         : %7.1f ms  (%d tokens @ %.1f tok/s)",
                genMs, max(0, tokenCount - 1), genTokPerSec))
            print(String(format: "[PROFILE] benchmark_total           : %7.1f ms",
                benchTotalMs))
            print("[PROFILE] ──────────────────────────────────────────────────────")

            // Per-phase wall-clock aggregator for `BenchmarkSignpost.PhaseLabel`
            // intervals (kv_update / sdpa / qsdpa / tq_*). Active under
            // `MLX_BENCH_PROFILE >= 2`. Measures CPU time between begin/end —
            // see `BenchmarkSignpost.dumpAggregator` for caveats about CPU vs
            // GPU time.
            BenchmarkSignpost.dumpAggregator()
        }

        print(hr + "\n")

        // Resolve the actual prefill chunk size used so the report shows
        // the real value, not "nil". When `params.prefillStepSize` is nil,
        // the iterator falls back to the model's `defaultPrefillStepSize`.
        let resolvedPrefillStepSize: Int = await container.perform { ctx in
            params.prefillStepSize ?? ctx.model.defaultPrefillStepSize
        }

        // ── 9. Write to markdown file ─────────────────────────────────────────
        BenchmarkWriter.append(
            model: family.name,
            repoId: variant.repoId,
            quantization: variant.quantization,
            kvConfig: kv.description,
            scenario: scenario,
            configKeyExtras: BenchEnv.ngramSize > 0 ? [("ngram", "\(BenchEnv.ngramSize)")] : [],
            contextSize: contextSize,
            promptTokens: prefillTokens,
            prefillTokPerSec: prefillTokPerSec,
            genTokPerSec: genTokPerSec,
            steadyTokPerSec: steadyTokPerSec,
            genTokens: tokenCount,
            ttftMs: ttft * 1000,
            thinkingPerplexity: thinkingPerplexity,
            generationPerplexity: generationPerplexity,
            thinkingKLD: thinkKLD,
            generationKLD: genKLD,
            baselineGPU: baselineGPU,
            peakGPU: Int(peakGPU),
            kvDelta: kvDelta,
            kvCacheBytes: kvCacheBytes,
            outputPreview: reportOutput,
            parameters: .init(
                generate: params,
                resolvedPrefillStepSize: resolvedPrefillStepSize,
                thinkingEnabled: useThinking,
                thinkingTokenBudget: hasThinkingBudget ? thinkingBudget : nil,
                kldSummary: kldParameterSummary(needsKLD: needsKLD, isWikitext2: false),
                maxOpsPerBuffer: BenchmarkWriter.resolvedMaxOpsPerBufferReport(),
                batchSize: BenchEnv.batch,
                speculativeDecoding: speculativeDecodingLabel(
                    ngramSize: params.ngramSize,
                    maxNgramDraftTokens: params.maxNgramDraftTokens,
                    draftModelId: draftModelIdForReport()
                ),
                systemPromptSummary: systemPromptSummary(for: systemPrompt, scenario: scenario)
            )
        )

        MLX.Memory.clearCache()
        #expect(tokenCount > 0, "[\(label)] Should generate at least 1 token")
    }

    // MARK: - Batched Benchmark

    /// Run N concurrent generations to measure multi-user throughput.
    /// Each generation runs independently through the ModelContainer's actor,
    /// which serializes access. This simulates N users sharing one model instance.
    private func runBatchedBenchmark(
        batchSize: Int,
        family: ModelFamily,
        variant: ModelVariant,
        repoId: String,
        kv: KVCacheConfig,
        label: String,
        contextSize: Int,
        messages: [MLXLMCommon.Message],
        systemPrompt: String?,
        maxTokens: Int = 200
    ) async throws {
        let hr = String(repeating: "=", count: 80)
        print("\n\(hr)")
        print("[BENCH] \(label)")
        print("[BENCH] Batch size: \(batchSize)")
        print(hr)

        let container = try await loadOrCacheModel(family: family, repoId: repoId)

        // Build input once (same prompt for all batch elements)
        var allMessages: [MLXLMCommon.Message] = []
        if let sys = systemPrompt {
            allMessages.append(["role": "system", "content": sys])
        }
        allMessages.append(contentsOf: messages)
        if family.supportsThinking && BenchEnv.thinkEnabled
            && !family.thinkingConfig.assistantPrefill.isEmpty {
            allMessages.append(["role": "assistant", "content": family.thinkingConfig.assistantPrefill])
        }
        let additionalContext: [String: any Sendable]? =
            (family.supportsThinking && BenchEnv.thinkEnabled) ? ["enable_thinking": true] : nil
        nonisolated(unsafe) let userInput = UserInput(prompt: .messages(allMessages), additionalContext: additionalContext)
        let lmInput = try await container.prepare(input: copy userInput)
        let promptTokens = lmInput.text.tokens.dim(lmInput.text.tokens.ndim - 1)
        print("[BENCH] Prepared \(promptTokens) tokens")

        let params = GenerateParameters(
            maxTokens: maxTokens,
            maxKVSize: contextSize > 0 ? contextSize : nil,
            kvBits: kv.kvBits,
            kvGroupSize: 64,
            quantizedKVStart: kv.quantizedKVStart,
            temperature: family.temperature,
            topP: family.topP,
            topK: family.topK,
            minP: family.minP,
            repetitionPenalty: family.repetitionPenalty,
            presencePenalty: family.presencePenalty,
            prefillStepSize: BenchEnv.prefillChunkSize,

            kvScheme: kv.kvScheme,
            ngramSize: BenchEnv.ngramSize,
            maxNgramDraftTokens: BenchEnv.ngramSize,
            trackPerplexity: false
        )

        let batchedMaxTokens = contextSize > 0 ? contextSize + maxTokens : 4096
        let batchedTicket = await container.perform { ctx in
            let dims = inferKVDimensions(model: ctx.model)
            return WiredMemoryUtils.resolveTicket(
                model: ctx.model,
                maxTokens: batchedMaxTokens,
                parameters: params,
                batchSize: batchSize,
                kvHeadsOverride: dims.kvHeads,
                headDimOverride: dims.headDim
            )
        }
        print("[BENCH] Wired ticket: \(batchedTicket.size / 1_048_576)MB for batch=\(batchSize) maxTokens=\(batchedMaxTokens)")

        MLX.GPU.resetPeakMemory()
        let baselineGPU = MLX.Memory.activeMemory

        struct BatchResult: Sendable {
            let tokenCount: Int
            let ttft: TimeInterval
            let totalTime: TimeInterval
        }

        let batchStart = Date()

        // Sendable capture for the task group: UserInput itself isn't Sendable, but
        // its constructor inputs (`[Message]` + optional `[String: any Sendable]`)
        // are. Rebuild a fresh UserInput inside each task from these primitives.
        let taskMessages = allMessages
        let taskAdditionalContext = additionalContext
        let perTaskTicket = batchedTicket

        let results: [BatchResult] = try await withThrowingTaskGroup(of: BatchResult.self) { group in
            for _ in 0..<batchSize {
                group.addTask {
                    // Each task prepares its own input (needs separate KV cache)
                    let taskUserInput = UserInput(
                        prompt: .messages(taskMessages),
                        additionalContext: taskAdditionalContext)
                    let input = try await container.prepare(input: taskUserInput)
                    let genStart = Date()
                    var tokenCount = 0
                    var firstTokenTime: TimeInterval? = nil

                    let stream = try await container.generate(
                        input: input,
                        parameters: params,
                        wiredMemoryTicket: perTaskTicket
                    )
                    for try await generation in stream {
                        guard generation.chunk != nil else { continue }
                        tokenCount += 1
                        if firstTokenTime == nil {
                            firstTokenTime = Date().timeIntervalSince(genStart)
                        }
                    }

                    let totalTime = Date().timeIntervalSince(genStart)
                    return BatchResult(
                        tokenCount: tokenCount,
                        ttft: firstTokenTime ?? totalTime,
                        totalTime: totalTime
                    )
                }
            }

            var collected: [BatchResult] = []
            for try await result in group {
                collected.append(result)
            }
            return collected
        }

        let batchWallTime = Date().timeIntervalSince(batchStart)
        let totalTokens = results.reduce(0) { $0 + $1.tokenCount }
        let avgTTFT = results.map(\.ttft).reduce(0, +) / Double(results.count)
        let avgPerSeqTokPerSec = results.map { r -> Double in
            let genTime = r.totalTime - r.ttft
            return genTime > 0 ? Double(r.tokenCount - 1) / genTime : 0
        }.reduce(0, +) / Double(results.count)
        let aggregateTokPerSec = batchWallTime > 0 ? Double(totalTokens) / batchWallTime : 0

        let peakGPU = MLX.Memory.peakMemory

        // ── Results ──────────────────────────────────────────────────────
        print("[BENCH] === RESULTS: \(label) ===")
        print("[BENCH] Method: summarization (batched)")
        print("[BENCH] Context: \(contextSize) tokens, Prompt Tokens: \(promptTokens) (after template)")
        print("[BENCH] Batch size: \(batchSize)")
        print("[BENCH] Aggregate throughput: \(String(format: "%.1f", aggregateTokPerSec)) tok/s (\(totalTokens) tokens in \(String(format: "%.1f", batchWallTime))s)")
        print("[BENCH] Avg per-sequence decode: \(String(format: "%.1f", avgPerSeqTokPerSec)) tok/s")
        print("[BENCH] Avg TTFT: \(String(format: "%.0f", avgTTFT * 1000))ms")
        print("[BENCH] GPU Baseline: \(String(format: "%.2f", Double(baselineGPU) / 1e9))GB")
        print("[BENCH] GPU Peak: \(String(format: "%.2f", Double(peakGPU) / 1e9))GB")
    }

    // MARK: - WikiText-2 Perplexity

    /// Standard LM perplexity evaluation via forced decode on WikiText-2 text.
    /// Tokenizes the text, feeds it through the model in chunks computing log-prob of
    /// each next token. No generation — pure evaluation of the model's predictive ability.
    /// PPL = exp(mean negative log-probability) over all predicted positions.
    private func runWikitext2Benchmark(
        family: ModelFamily,
        kv: KVCacheConfig,
        contextSize: Int
    ) async throws {
        let (variant, repoId) = try await resolveVariant(family: family)
        let label = "\(family.name) [\(variant.quantization)] — wikitext2 \(contextSize) [\(kv)]"

        let hr = String(repeating: "=", count: 80)
        print("\n\(hr)")
        print("[BENCH] \(label)")
        print(hr)

        // Load model
        let container = try await loadOrCacheModel(family: family, repoId: repoId)

        // Load and tokenize WikiText-2 text.
        // We tokenize the full text, take the first `contextSize` tokens, then decode
        // back to count whitespace-delimited words for word-level PPL normalization
        // (the standard metric per EleutherAI, comparable across tokenizers).
        let wikitext = try loadWikitext2()
        let (tokenIds, wordCount): ([Int32], Int) = await container.perform { ctx in
            let allTokens = ctx.tokenizer.encode(text: wikitext)
            let sliced = Array(allTokens.prefix(contextSize))
            // Decode the evaluated tokens back to text to count words
            let decodedText = ctx.tokenizer.decode(tokenIds: sliced)
            let words = decodedText.split(whereSeparator: { $0.isWhitespace }).count
            return (sliced.map { Int32($0) }, words)
        }

        guard tokenIds.count >= 2 else {
            print("[BENCH] Not enough tokens for wikitext2 evaluation at context \(contextSize)")
            return
        }

        print("[BENCH] WikiText-2: \(tokenIds.count) tokens, \(wordCount) words (target: \(contextSize))")

        let chunkSize = BenchEnv.prefillChunkSize ?? 2048  // Process in chunks to avoid OOM on the computation graph

        let params = GenerateParameters(
            maxKVSize: contextSize > 0 ? contextSize : nil,
            kvBits: kv.kvBits,
            kvGroupSize: 64,
            quantizedKVStart: kv.quantizedKVStart,
            prefillStepSize: chunkSize,
            kvScheme: kv.kvScheme
        )

        MLX.GPU.resetPeakMemory()
        let baselineGPU = MLX.Memory.activeMemory
        let startTime = Date()

        // Process the ENTIRE sequence in chunks, capturing logits from every position.
        // Unlike model.prepare() which discards logits from early chunks, we manually
        // feed each chunk and extract per-position log-probs before moving on.
        let ppl: (wordPPL: Double, tokenPPL: Double, totalNLL: Double) = await container.perform { ctx in
            var cache = ctx.model.newCache(parameters: params)
            var state: LMOutput.State? = nil
            var negLogProbSum: Double = 0
            var evalCount = 0

            // Process tokens in chunks, accumulating NLL
            let seqLen = tokenIds.count
            var offset = 0
            while offset < seqLen - 1 {
                let end = min(offset + chunkSize, seqLen - 1)
                let chunkTokens = MLXArray(Array(tokenIds[offset..<end]))
                let input = LMInput.Text(tokens: chunkTokens)

                let result = ctx.model(
                    input[text: .newAxis],
                    cache: cache.isEmpty ? nil : cache,
                    state: state
                )
                state = result.state

                // Affine quantization per chunk. Turbo deferred to after all chunks
                // (turbo conversion changes cache types which breaks mid-prefill).
                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: params.kvBits,
                    kvGroupSize: params.kvGroupSize,
                    quantizedKVStart: params.quantizedKVStart
                )

                // logits shape: [1, chunkLen, vocab]
                // Position i in chunk predicts token at global position (offset + i + 1)
                let logProbs = log(softmax(result.logits.asType(.float32), axis: -1))
                let chunkLen = end - offset

                for i in 0..<chunkLen {
                    let targetToken = Int(tokenIds[offset + i + 1])
                    let lp = logProbs[0, i, targetToken]
                    negLogProbSum -= Double(lp.item(Float.self))
                    evalCount += 1
                }

                // Force evaluation to free intermediate computation graph
                eval(cache)

                offset = end
            }

            // Word-level PPL (EleutherAI standard): normalize by word count, not token count.
            // This makes PPL comparable across models with different tokenizers/vocabularies.
            let wordPPL = exp(negLogProbSum / Double(wordCount))
            let tokenPPL = exp(negLogProbSum / Double(evalCount))
            return (wordPPL, tokenPPL, negLogProbSum)
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let tokPerSec = Double(tokenIds.count) / elapsed
        let peakGPU = MLX.Memory.peakMemory
        let activeGPU = MLX.Memory.activeMemory
        let kvDelta = activeGPU > baselineGPU ? activeGPU - baselineGPU : 0

        print("[BENCH] WikiText-2 Word PPL: \(String(format: "%.4f", ppl.wordPPL)) (token PPL: \(String(format: "%.4f", ppl.tokenPPL)))")
        print("[BENCH] Throughput: \(String(format: "%.1f", tokPerSec)) tok/s")
        print("[BENCH] Time: \(String(format: "%.1f", elapsed))s")
        print(hr)

        BenchmarkWriter.append(
            model: family.name,
            repoId: variant.repoId,
            quantization: variant.quantization,
            kvConfig: kv.description,
            scenario: "wikitext2",
            contextSize: contextSize,
            promptTokens: tokenIds.count,
            prefillTokPerSec: tokPerSec,
            genTokPerSec: 0,
            genTokens: 0,
            ttftMs: elapsed * 1000,
            thinkingPerplexity: nil,
            generationPerplexity: ppl.wordPPL,
            thinkingKLD: nil,
            generationKLD: nil,
            baselineGPU: baselineGPU,
            peakGPU: Int(peakGPU),
            kvDelta: kvDelta,
            outputPreview: "WikiText-2 forced-decode perplexity evaluation",
            parameters: .init(
                generate: params,
                resolvedPrefillStepSize: chunkSize,
                thinkingEnabled: false,
                thinkingTokenBudget: nil,
                kldSummary: kldParameterSummary(needsKLD: false, isWikitext2: true),
                maxOpsPerBuffer: BenchmarkWriter.resolvedMaxOpsPerBufferReport(),
                batchSize: BenchEnv.batch,
                speculativeDecoding: speculativeDecodingLabel(
                    ngramSize: params.ngramSize,
                    maxNgramDraftTokens: params.maxNgramDraftTokens,
                    draftModelId: draftModelIdForReport()
                ),
                systemPromptSummary: systemPromptSummary(for: nil, scenario: "wikitext2")
            )
        )

        MLX.Memory.clearCache()
    }

    // MARK: - Needle-in-a-Haystack

    /// Needle-in-a-haystack benchmark: insert a known fact at multiple depth positions
    /// in filler text and test whether the model can retrieve it. Runs one test per depth.
    private func runNIAHBenchmark(
        family: ModelFamily,
        kv: KVCacheConfig,
        contextSize: Int
    ) async throws {
        let (variant, repoId) = try await resolveVariant(family: family)
        let filler = try loadPrompt(tokenCount: contextSize)

        for depth in Self.niahDepths {
            let depthPct = Int(depth * 100)
            let insertAt = filler.index(
                filler.startIndex,
                offsetBy: max(0, min(Int(Double(filler.count) * depth), filler.count - 1))
            )
            let prompt = String(filler[..<insertAt])
                + "\n\n" + Self.niahNeedle + "\n\n"
                + String(filler[insertAt...])
                + "\n\n" + Self.niahQuestion

            let label = "\(family.name) [\(variant.quantization)] — niah \(contextSize) @\(depthPct)% [\(kv)]"
            try await runGenerationBenchmark(
                family: family, variant: variant, repoId: repoId,
                kv: kv, label: label, contextSize: 0,  // unbounded — NIAH needs full prompt visible
                messages: [["role": "user", "content": prompt]],
                systemPrompt: Self.minimalSystemPrompt, maxTokens: 100,
                validation: { output, _ in
                    let found = output.lowercased().contains(Self.niahAnswer.lowercased())
                    return found ? "PASS(@\(depthPct)%): " : "FAIL(@\(depthPct)%): "
                }
            )
        }
    }

    // MARK: - Raw Prefill Benchmark

    /// Minimal prefill timing test that bypasses the full generate() pipeline.
    /// Directly calls model.callAsFunction(tokens, cache) + eval(caches),
    /// matching what Python's `m(mx.array(tokens)[None], cache=c); mx.eval(...)` does.
    /// 3 warmup runs + 5 timed runs, reports median tok/s.
    private func runRawPrefillBenchmark(
        family: ModelFamily, kv: KVCacheConfig, contextSize: Int
    ) async throws {
        let (variant, repoId) = try await resolveVariant(family: family)
        let label = "\(family.name) [\(variant.quantization)] — raw-prefill \(contextSize) [\(kv)]"

        let hr = String(repeating: "=", count: 80)
        print("\n\(hr)")
        print("[BENCH] \(label)")
        print(hr)

        let container = try await loadOrCacheModel(family: family, repoId: repoId)

        // Tokenize a prompt of the target length
        let prompt = try loadPrompt(tokenCount: contextSize)
        let tokens: [Int] = await container.encode(prompt)
        let actualTokens = tokens.count
        print("[BENCH] Prompt tokens: \(actualTokens)")

        let warmupRuns = 3
        let timedRuns = 5

        // Run inside container.perform to get direct model access
        let timings: [Double] = await container.perform { ctx in
            let model = ctx.model

            // Build KV cache creation params matching the KV config
            var genParams = GenerateParameters()
            if let kvBits = kv.kvBits { genParams.kvBits = kvBits }
            if let kvScheme = kv.kvScheme { genParams.kvScheme = kvScheme }
            genParams.quantizedKVStart = kv.quantizedKVStart

            let tokenArray = MLXArray(tokens).reshaped(1, tokens.count)  // [1, seqLen]

            // ── Warmup runs ──
            for i in 0..<warmupRuns {
                let cache = model.newCache(parameters: genParams)
                // Build the lazy computation graph through the full model.
                // Only eval(cache) below — MLX's lazy eval skips lmHead since
                // logits aren't in the cache dependency graph. This gives us
                // pure prefill timing (embed + layers + norm), no wasted lmHead matmul.
                // Matches Python: m(mx.array(tokens)[None], cache=c); mx.eval(cache_state)
                let _ = model(tokenArray, cache: cache)
                // Only eval non-shared caches — matches Python's make_cache (15 caches)
                let nonSharedCaches = Array(cache.prefix(15))
                eval(nonSharedCaches)
                Stream.defaultStream(.gpu).synchronize()
                MLX.Memory.clearCache()
                print("[BENCH] Warmup \(i + 1)/\(warmupRuns) done")
            }

            // ── Timed runs ──
            var results: [Double] = []
            for i in 0..<timedRuns {
                let cache = model.newCache(parameters: genParams)

                let start = CFAbsoluteTimeGetCurrent()
                let _ = model(tokenArray, cache: cache)
                let nonSharedCaches = Array(cache.prefix(15))
                eval(nonSharedCaches)
                Stream.defaultStream(.gpu).synchronize()
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                MLX.Memory.clearCache()

                let tokPerSec = Double(actualTokens) / elapsed
                results.append(tokPerSec)
                print("[BENCH] Run \(i + 1)/\(timedRuns): \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.3f", elapsed * 1000)) ms)")
            }

            return results
        }

        // Report stats
        let sorted = timings.sorted()
        let median = sorted[sorted.count / 2]
        let mean = timings.reduce(0, +) / Double(timings.count)
        let min = sorted.first!
        let max = sorted.last!

        print("[BENCH] ─── Results (\(actualTokens) tokens) ───")
        print("[BENCH] Median: \(String(format: "%.1f", median)) tok/s")
        print("[BENCH] Mean:   \(String(format: "%.1f", mean)) tok/s")
        print("[BENCH] Min:    \(String(format: "%.1f", min)) tok/s")
        print("[BENCH] Max:    \(String(format: "%.1f", max)) tok/s")
        print("[RESULT] \(label) | \(actualTokens) tokens | \(String(format: "%.1f", median)) tok/s (median)")
    }

    // MARK: - Model Loading Helper

    /// Load a model container, using ModelCache to avoid reloading across context sizes.
    private func loadOrCacheModel(family: ModelFamily, repoId: String) async throws -> ModelContainer {
        if let cached = ModelCache.shared.get(repoId) {
            return cached
        }
        // Support local paths: if repoId starts with / or ~, treat as directory
        let isLocal = repoId.hasPrefix("/") || repoId.hasPrefix("~") || repoId.hasPrefix(".")
        let modelConfig: ModelConfiguration
        if isLocal {
            let expanded = NSString(string: repoId).expandingTildeInPath
            modelConfig = family.extraEOSTokens.isEmpty
                ? ModelConfiguration(directory: URL(fileURLWithPath: expanded))
                : ModelConfiguration(directory: URL(fileURLWithPath: expanded), extraEOSTokens: Set(family.extraEOSTokens))
        } else {
            modelConfig = family.extraEOSTokens.isEmpty
                ? ModelConfiguration(id: repoId)
                : ModelConfiguration(id: repoId, extraEOSTokens: Set(family.extraEOSTokens))
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: benchmarkDownloader,
            using: benchmarkTokenizerLoader,
            configuration: modelConfig,
            progressHandler: { p in
                if p.fractionCompleted < 0.01 || p.fractionCompleted > 0.99 {
                    print("[BENCH] Loading: \(String(format: "%.0f", p.fractionCompleted * 100))%")
                }
            }
        )
        ModelCache.shared.set(repoId, container)
        return container
    }

    // MARK: - KLD Forced Decode

    /// Forced decode: feed tokens from the target generation through the baseline model
    /// (bf16/8bit, no KV quantization), computing the baseline log prob of each token.
    /// KLD = mean(quantized_logprob - baseline_logprob) per phase (≥ 0).
    private func forcedDecodeKLD(
        container: ModelContainer,
        input: LMInput,
        family: ModelFamily,
        quantizedData: BaselineTokenData,
        thinkStartId: Int32?,
        thinkEndId: Int32?,
        thinkingBudget: Int,
        maxTokens: Int
    ) async throws -> (thinkKLD: Double?, genKLD: Double?) {
        // effectiveMaxTokens keys off whichever thinking style is active, matching
        // the hasThinkingBudget computation at the main call site.
        let hasThinkingBudget = thinkStartId != nil  // bracket-only here; harmony path
                                                      // populates quantizedData directly and we
                                                      // just process everything captured.
        let effectiveMaxTokens = hasThinkingBudget ? thinkingBudget + maxTokens : maxTokens

        // Params WITHOUT KV quantization — the baseline model runs unquantized
        let params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
            temperature: family.temperature,
            topP: family.topP,
            topK: family.topK,
            minP: family.minP,
            repetitionPenalty: family.repetitionPenalty,
            presencePenalty: family.presencePenalty,

            reasoningEffort: family.reasoningEffort,
            thinkStartTokenId: thinkStartId,
            thinkEndTokenId: thinkEndId,
            thinkingPhasePrefilled: thinkStartId != nil
        )

        print("[KLD] Running forced decode (no KV quant) for \(quantizedData.tokenIds.count) tokens...")

        // Manual prefill + forced decode (bypasses TokenIterator to avoid off-by-one:
        // TokenIterator.init does prefill AND samples the first token, consuming position-0
        // logits. We need those logits to compute KLD for token[0].)
        nonisolated(unsafe) let sendableInput = input
        return try await container.perform { ctx in
            // 1. Create cache and prefill — NO KV quantization
            let cache = ctx.model.newCache(parameters: params)
            var state: LMOutput.State? = nil
            var logits: MLXArray

            // Resolve to the model's audited default when params didn't set one.
            let resolvedWindowSize = params.prefillStepSize ?? ctx.model.defaultPrefillStepSize
            switch try ctx.model.prepare(sendableInput, cache: cache, windowSize: resolvedWindowSize) {
            case .tokens(let remaining):
                let result = ctx.model(
                    remaining[text: .newAxis],
                    cache: cache.isEmpty ? nil : cache,
                    state: nil
                )
                state = result.state
                logits = result.logits
            case .logits(let result):
                logits = result.logits
                state = result.state
            }

            // 2. Forced decode: compute baseline (unquantized) log prob for each token.
            // Process the full captured sequence — the target's generation loop already
            // bounded it via its own maxTokens, so capping again here would drop data.
            let tokenIds = quantizedData.tokenIds
            let quantizedLogProbs = quantizedData.logProbs
            let phases = quantizedData.phases
            let tokenCount = tokenIds.count
            _ = effectiveMaxTokens  // retained for the Parameters summary only

            var thinkKLDSum: Double = 0
            var thinkCount = 0
            var genKLDSum: Double = 0
            var genCount = 0
            var inThinkPhase = thinkStartId != nil  // prefilled

            for i in 0..<tokenCount {
                let tokenId = tokenIds[i]
                let quantizedLogProb = quantizedLogProbs[i]
                let phase = phases[i]

                // Compute baseline (unquantized) log prob
                let positionLogits = logits[0..., -1, 0...]
                let logprobs = log(softmax(positionLogits.asType(.float32)))
                let baselineLogProb = takeAlong(
                    logprobs.reshaped([1, -1]),
                    MLXArray(Int32(tokenId)).reshaped([1, 1]),
                    axis: 1
                ).reshaped([])
                let bLogProb = Double(baselineLogProb.item(Float.self))

                // Phase tracking
                if let startId = thinkStartId, tokenId == Int(startId) {
                    inThinkPhase = true
                } else if let endId = thinkEndId, tokenId == Int(endId) {
                    inThinkPhase = false
                }

                // KLD = quantized_logprob - baseline_logprob = KL(P_quant || P_base) ≥ 0
                // Positive values indicate the quantized model diverges from baseline.
                if phase == "think" || (phase != "marker" && inThinkPhase) {
                    thinkKLDSum += quantizedLogProb - bLogProb
                    thinkCount += 1
                } else if phase == "gen" || (phase != "marker" && !inThinkPhase) {
                    genKLDSum += quantizedLogProb - bLogProb
                    genCount += 1
                }

                // Feed token as next input → get logits for next position.
                // Must be [1]-shaped (not scalar) so that after [text: .newAxis] it becomes
                // [1,1] — matching what the sampler produces during normal generation.
                // A scalar + .newAxis = [1] (1D), which the embedding misinterprets as
                // (batch=1, seq=hiddenSize) instead of (batch=1, seq=1, hiddenSize).
                let forcedToken = MLXArray([Int32(tokenId)])
                let y = LMInput.Text(tokens: forcedToken)
                let nextResult = ctx.model(
                    y[text: .newAxis],
                    cache: cache.isEmpty ? nil : cache,
                    state: state
                )
                state = nextResult.state
                logits = nextResult.logits

                asyncEval(forcedToken)
            }

            let thinkKLD = thinkCount > 0 ? thinkKLDSum / Double(thinkCount) : nil
            let genKLD = genCount > 0 ? genKLDSum / Double(genCount) : nil

            print("[KLD] Forced decode: \(tokenCount) tokens, think=\(thinkCount) gen=\(genCount)")
            if let k = thinkKLD { print("[KLD] Think KLD: \(String(format: "%.6f", k))") }
            if let k = genKLD { print("[KLD] Gen KLD: \(String(format: "%.6f", k))") }

            return (thinkKLD, genKLD)
        }
    }

    // MARK: - Utilities

    private func formatBytes(_ bytes: Int) -> String {
        BenchmarkWriter.formatBytes(bytes)
    }

    /// Short text for benchmark `## System prompt` (no user prompt bodies).
    private func systemPromptSummary(for systemPrompt: String?, scenario: String) -> String {
        if scenario == "wikitext2" {
            return "Not applicable (WikiText-2 LM evaluation; no chat system role)."
        }
        guard let sp = systemPrompt else {
            return "No system role message; user-only messages per methodology (no full user prompt in this report)."
        }
        if sp == Self.minimalSystemPrompt {
            return "Standard assistant system prompt — verbatim text in [benchmarks README](../README.md#system-prompts)."
        }
        return "Custom system prompt (not repeated in this report)."
    }

    /// When set, reported as `draft (id)` in benchmark markdown (draft-model speculative runs).
    private func draftModelIdForReport() -> String? {
        let s = ProcessInfo.processInfo.environment["MLX_BENCH_DRAFT_MODEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return s.isEmpty ? nil : s
    }

    private func kldParameterSummary(needsKLD: Bool, isWikitext2: Bool) -> String {
        guard BenchEnv.kldEnabled else { return "No" }
        if isWikitext2 {
            return "Yes (not evaluated — wikitext2 method)"
        }
        if needsKLD {
            return "Yes"
        }
        return "Yes (not evaluated — baseline configuration)"
    }

    private func speculativeDecodingLabel(
        ngramSize: Int,
        maxNgramDraftTokens: Int,
        draftModelId: String? = nil
    ) -> String {
        if let id = draftModelId?.trimmingCharacters(in: .whitespacesAndNewlines), !id.isEmpty {
            return "draft (\(id))"
        }
        if ngramSize > 0 {
            return "ngram (size=\(ngramSize), maxDraft=\(maxNgramDraftTokens))"
        }
        return "none"
    }

    /// Resolve a resource file relative to the Tests/Benchmarks/Resources directory.
    /// Uses #filePath since Bundle.module is unavailable (no resource processing in this target).
    private static func resourceURL(name: String, ext: String, filePath: String = #filePath) -> URL? {
        let thisFile = URL(fileURLWithPath: filePath)
        let benchDir = thisFile.deletingLastPathComponent()
        let resourceURL = benchDir.appendingPathComponent("Resources")
            .appendingPathComponent("llm-test-prompts")
            .appendingPathComponent(name)
            .appendingPathExtension(ext)
        return FileManager.default.fileExists(atPath: resourceURL.path) ? resourceURL : nil
    }

    private func loadPrompt(tokenCount: Int) throws -> String {
        let filename = "prompt_\(tokenCount)"
        guard let url = Self.resourceURL(name: filename, ext: "txt") else {
            throw BenchmarkError("Missing test resource: \(filename).txt")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func loadWikitext2() throws -> String {
        guard let url = Self.resourceURL(name: "wikitext2_test", ext: "txt") else {
            throw BenchmarkError("Missing test resource: wikitext2_test.txt")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}
