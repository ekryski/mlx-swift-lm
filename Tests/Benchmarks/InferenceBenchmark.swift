import Foundation
import Hub
import Testing
import MLX
import MLXNN
@testable import MLXLMCommon
import MLXLLM
import Tokenizers

// MARK: - Thinking Budget Processor

/// Forces </think> after a token budget to prevent unbounded thinking phases.
/// This keeps benchmarks tractable: models think up to `maxThinkingTokens`,
/// then must emit </think> and generate the actual response.
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
    }

    func process(logits: MLXArray) -> MLXArray {
        guard inThinkingPhase else { return logits }

        var modified = logits

        // Suppress EOS tokens during thinking phase so the model is forced to generate </think>
        // before ending. Without this, small models (0.8B, 2B) terminate inside the thinking
        // phase and never produce generation tokens, causing Gen PPL to be nil.
        for eosId in eosTokenIds {
            if logits.ndim == 1 {
                modified[Int(eosId)] = MLXArray(-Float.infinity)
            } else {
                modified[0..., Int(eosId)] = MLXArray(-Float.infinity)
            }
        }

        // Budget exceeded: force </think> by boosting its logit to dominate softmax.
        // Use a large FINITE value (not +inf) — softmax(+inf) = exp(+inf)/sum = NaN
        // which causes the sampler to return garbage (token 0), never triggering transition.
        // With logits typically in [-30, 30], 100.0 gives P(</think>) ≈ 1.0 with no NaN.
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
        let id = token.item(Int32.self)
        if id == thinkStartTokenId {
            inThinkingPhase = true
        } else if id == thinkEndTokenId {
            inThinkingPhase = false
        } else if inThinkingPhase {
            thinkingTokenCount += 1
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

// MARK: - Mock Tools

/// Minimal mock tool spec for tool-call benchmarking (no external dependencies).
enum MockTools {
    static func shellToolSpec() -> ToolSpec {
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
        ] as ToolSpec
    }
}

// MARK: - Environment

/// Centralized environment variable access for benchmarks.
/// All configuration comes from env vars set by benchmark.sh (or manually).
private enum BenchEnv {
    /// Model to benchmark — registry short name (e.g., "qwen35-0.8b") or HF repo ID.
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
        case "affine4": return .affine(bits: 4)
        case "turbo4": return .turbo(bits: 4)
        case "turbo3": return .turbo(bits: 3)
        case "turbo4v2": return .turboAsym(keyBits: 4, valueBits: 2)
        case "turbo4v3": return .turboAsym(keyBits: 4, valueBits: 3)
        case "turbo3v2": return .turboAsym(keyBits: 3, valueBits: 2)
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
/// See Tests/Benchmarks/README.md for full documentation.
@Suite("Inference Benchmarks", .serialized)
struct InferenceBenchmarks {

    // MARK: - Constants

    static let contextSizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    static let toolQuery = "What is the current date and time?"
    static let multiTurnContext: [Message] = [
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
    static let simpleQuery = "Hello! What is your name and what can you help me with?"
    /// Default context limit for non-scaling methods.
    /// Enforced via maxKVSize (RotatingKVCache) to simulate a realistic chat deployment.
    static let defaultContextLimit = 4096
    static let niahNeedle = "The special magic verification code is BLUE TIGER 42."
    static let niahAnswer = "BLUE TIGER 42"
    static let niahQuestion = "What is the special magic verification code mentioned in the text above? Reply with only the code, nothing else."
    static let niahDepths: [Double] = [0.1, 0.25, 0.5, 0.75, 0.9]

    // MARK: - Entry Point

    /// Single benchmark entry point. All configuration comes from env vars.
    @Test func benchmark() async throws {
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
            for ctx in contexts {
                let prompt = try loadPrompt(tokenCount: ctx)
                try await runGenerationBenchmark(
                    family: family, variant: variant, repoId: repoId, kv: kv,
                    label: "\(family.name) [\(variant.quantization)] — summarization \(ctx) [\(kv)]",
                    contextSize: ctx,
                    messages: [["role": "user", "content": prompt]],
                    systemPrompt: nil, maxTokens: 200
                )
            }

        case "wikitext2":
            let contexts = BenchEnv.contexts ?? Self.contextSizes
            for ctx in contexts {
                try await runWikitext2Benchmark(family: family, kv: kv, contextSize: ctx)
            }

        case "niah":
            let contexts = BenchEnv.contexts ?? Self.contextSizes
            for ctx in contexts {
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

        default:
            print("[BENCH] Unknown method: \(method)")
        }
    }

    // MARK: - Family Resolution

    /// Resolve model family from MLX_BENCH_MODEL env var.
    private func resolveFamily() throws -> ModelFamily {
        guard let name = BenchEnv.model else {
            throw BenchmarkError("MLX_BENCH_MODEL not set — use --model flag")
        }
        if let family = ModelRegistry.family(named: name) {
            return family
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
    typealias ValidationCheck = (_ output: String, _ toolCalls: [ToolCall]) -> String?

    private func runGenerationBenchmark(
        family: ModelFamily,
        variant: ModelVariant,
        repoId: String,
        kv: KVCacheConfig,
        label: String,
        contextSize: Int,
        messages: [Message],
        systemPrompt: String?,
        maxTokens: Int = 200,
        includeTools: Bool = false,
        validation: ValidationCheck? = nil
    ) async throws {
        let hr = String(repeating: "=", count: 80)
        print("\n\(hr)")
        print("[BENCH] \(label)")
        print("[BENCH] Model: \(repoId)")
        print("[BENCH] Quantization: \(variant.quantization)")
        print("[BENCH] KV: \(kv)")
        print(hr)

        let thinkingBudget = 200  // max thinking tokens before forcing </think>

        // ── 1. Build user input (no container needed yet) ─────────────────────
        var allMessages: [Message] = []
        if let sys = systemPrompt {
            allMessages.append(["role": "system", "content": sys])
        }
        allMessages.append(contentsOf: messages)
        // Force thinking mode via assistant prefill for thinking-capable models
        if family.supportsThinking {
            allMessages.append(["role": "assistant", "content": "<think>\n"])
        }
        let tools: [ToolSpec]? = includeTools ? [MockTools.shellToolSpec()] : nil
        let userInput = UserInput(prompt: .messages(allMessages), tools: tools)

        // ── 2. Load target model (cached across context sizes) ──────────────────
        let container = try await loadOrCacheModel(family: family, repoId: repoId)

        // ── 4. Discover thinking tokens with target container ─────────────────
        let (thinkStartId, thinkEndId, eosTokenIds): (Int32?, Int32?, [Int32]) = family.supportsThinking
            ? await container.perform { ctx in
                let startId = ctx.tokenizer.convertTokenToId("<think>").map { Int32($0) }
                let endId = ctx.tokenizer.convertTokenToId("</think>").map { Int32($0) }
                // Collect all EOS token IDs so we can suppress them during thinking phase
                var eosIds = ctx.configuration.eosTokenIds.map { Int32($0) }
                if let tokEos = ctx.tokenizer.eosTokenId { eosIds.append(Int32(tokEos)) }
                for token in ctx.configuration.extraEOSTokens {
                    if let id = ctx.tokenizer.convertTokenToId(token) { eosIds.append(Int32(id)) }
                }
                return (startId, endId, Array(Set(eosIds)))
              }
            : (nil, nil, [])

        if let s = thinkStartId, let e = thinkEndId {
            print("[BENCH] Thinking tokens: <think>=\(s), </think>=\(e), budget=\(thinkingBudget), eos=\(eosTokenIds)")
        }
        let budgetProcessor: ThinkingBudgetProcessor? = thinkStartId.flatMap { startId in
            thinkEndId.map { endId in
                ThinkingBudgetProcessor(
                    thinkStartTokenId: startId,
                    thinkEndTokenId: endId,
                    maxThinkingTokens: thinkingBudget,
                    prefilled: true,
                    eosTokenIds: eosTokenIds
                )
            }
        }

        // ── 5. Prepare input ──────────────────────────────────────────────────
        let prepareStart = Date()
        let lmInput: LMInput
        do {
            lmInput = try await container.prepare(input: userInput)
        } catch {
            print("[BENCH] Template error: \(error). Retrying without tools...")
            let fallbackInput = UserInput(prompt: .messages(allMessages))
            lmInput = try await container.prepare(input: fallbackInput)
        }
        let promptTokens = lmInput.text.tokens.dim(lmInput.text.tokens.ndim - 1)
        print("[BENCH] Prepared \(promptTokens) tokens in \(String(format: "%.0f", Date().timeIntervalSince(prepareStart) * 1000))ms")

        // ── 6. Generate ───────────────────────────────────────────────────────
        // effectiveMaxTokens = thinking budget + response budget for thinking models
        let effectiveMaxTokens = thinkStartId != nil ? thinkingBudget + maxTokens : maxTokens
        let additionalProcessors: [any LogitProcessor] = budgetProcessor.map { [$0] } ?? []

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
            prefillStepSize: 2048,
            additionalProcessors: additionalProcessors,
            reasoningEffort: family.reasoningEffort,
            thinkStartTokenId: thinkStartId,
            thinkEndTokenId: thinkEndId,
            thinkingPhasePrefilled: thinkStartId != nil,
            collectPerTokenData: needsKLD
        )

        let ticket = WiredMemoryTicket(
            size: 20 * 1024 * 1024 * 1024,
            policy: MLX.WiredSumPolicy()
        )

        MLX.GPU.resetPeakMemory()
        let baselineGPU = MLX.Memory.activeMemory

        let genStart = Date()
        var tokenCount = 0
        var firstTokenTime: TimeInterval? = nil
        var outputText = ""
        var completionInfo: GenerateCompletionInfo? = nil
        var toolCalls: [ToolCall] = []

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
            guard let chunk = generation.chunk else { continue }

            tokenCount += 1
            outputText += chunk

            if firstTokenTime == nil {
                firstTokenTime = Date().timeIntervalSince(genStart)
            }
        }

        let totalTime = Date().timeIntervalSince(genStart)
        let ttft = firstTokenTime ?? totalTime
        let generationTime = totalTime - ttft
        let genTokPerSec = generationTime > 0 ? Double(tokenCount - 1) / generationTime : 0

        let prefillTime = completionInfo?.promptTime ?? ttft
        let prefillTokens = completionInfo?.promptTokenCount ?? promptTokens
        let prefillTokPerSec = prefillTime > 0 ? Double(prefillTokens) / prefillTime : 0

        let peakGPU = MLX.Memory.peakMemory
        let activeGPU = MLX.Memory.activeMemory
        let kvDelta = activeGPU > baselineGPU ? activeGPU - baselineGPU : 0

        // KV cache size computed from token count and quantization config.
        // Deterministic and comparable across runs (unlike MLX activeMemory delta).
        let totalTokens = prefillTokens + tokenCount
        let kvCacheBytes = kv.cacheBytes(
            tokens: totalTokens, kvHeads: 16, headDim: 128, layers: 28)

        let thinkingPerplexity = completionInfo?.thinkingPerplexity
        let generationPerplexity = completionInfo?.generationPerplexity

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
                    configuration: baselineConfig
                ) { p in
                    if p.fractionCompleted < 0.01 || p.fractionCompleted > 0.99 {
                        print("[KLD] Loading baseline: \(String(format: "%.0f", p.fractionCompleted * 100))%")
                    }
                }

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

        print(hr + "\n")

        // ── 9. Write to markdown file ─────────────────────────────────────────
        BenchmarkWriter.append(
            model: family.name,
            repoId: variant.repoId,
            quantization: variant.quantization,
            kvConfig: kv.description,
            scenario: scenario,
            contextSize: contextSize,
            promptTokens: prefillTokens,
            prefillTokPerSec: prefillTokPerSec,
            genTokPerSec: genTokPerSec,
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
                temperature: family.temperature,
                topP: family.topP,
                topK: family.topK,
                minP: family.minP,
                maxTokens: effectiveMaxTokens,
                thinkingBudget: thinkStartId != nil ? thinkingBudget : nil,
                repetitionPenalty: family.repetitionPenalty,
                presencePenalty: family.presencePenalty,
                reasoningEffort: family.reasoningEffort
            )
        )

        MLX.Memory.clearCache()
        #expect(tokenCount > 0, "[\(label)] Should generate at least 1 token")
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
            let decodedText = ctx.tokenizer.decode(tokens: sliced)
            let words = decodedText.split(whereSeparator: { $0.isWhitespace }).count
            return (sliced.map { Int32($0) }, words)
        }

        guard tokenIds.count >= 2 else {
            print("[BENCH] Not enough tokens for wikitext2 evaluation at context \(contextSize)")
            return
        }

        print("[BENCH] WikiText-2: \(tokenIds.count) tokens, \(wordCount) words (target: \(contextSize))")

        let chunkSize = 2048  // Process in chunks to avoid OOM on the computation graph

        let params = GenerateParameters(
            maxKVSize: contextSize > 0 ? contextSize : nil,
            kvBits: kv.kvBits,
            kvGroupSize: 64,
            quantizedKVStart: kv.quantizedKVStart,
            prefillStepSize: chunkSize
        )

        MLX.GPU.resetPeakMemory()
        let baselineGPU = MLX.Memory.activeMemory
        let startTime = Date()

        // Process the ENTIRE sequence in chunks, capturing logits from every position.
        // Unlike model.prepare() which discards logits from early chunks, we manually
        // feed each chunk and extract per-position log-probs before moving on.
        let ppl: (wordPPL: Double, tokenPPL: Double, totalNLL: Double) = try await container.perform { ctx in
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

                // Apply KV quantization after each chunk
                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: params.kvBits,
                    kvGroupSize: params.kvGroupSize,
                    quantizedKVStart: params.quantizedKVStart,
                    kvScheme: params.kvScheme
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
                temperature: family.temperature,
                topP: family.topP,
                topK: family.topK,
                minP: family.minP,
                maxTokens: nil,
                thinkingBudget: nil,
                repetitionPenalty: family.repetitionPenalty,
                presencePenalty: family.presencePenalty,
                reasoningEffort: family.reasoningEffort
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

    // MARK: - Model Loading Helper

    /// Load a model container, using ModelCache to avoid reloading across context sizes.
    private func loadOrCacheModel(family: ModelFamily, repoId: String) async throws -> ModelContainer {
        if let cached = ModelCache.shared.get(repoId) {
            return cached
        }
        let modelConfig = family.extraEOSTokens.isEmpty
            ? ModelConfiguration(id: repoId)
            : ModelConfiguration(id: repoId, extraEOSTokens: Set(family.extraEOSTokens))
        let container = try await LLMModelFactory.shared.loadContainer(configuration: modelConfig) { p in
            if p.fractionCompleted < 0.01 || p.fractionCompleted > 0.99 {
                print("[BENCH] Loading: \(String(format: "%.0f", p.fractionCompleted * 100))%")
            }
        }
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
        let effectiveMaxTokens = thinkStartId != nil ? thinkingBudget + maxTokens : maxTokens

        // Params WITHOUT KV quantization — the baseline model runs unquantized
        let params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
            temperature: family.temperature,
            topP: family.topP,
            topK: family.topK,
            minP: family.minP,
            repetitionPenalty: family.repetitionPenalty,
            presencePenalty: family.presencePenalty,
            prefillStepSize: 2048,
            reasoningEffort: family.reasoningEffort,
            thinkStartTokenId: thinkStartId,
            thinkEndTokenId: thinkEndId,
            thinkingPhasePrefilled: thinkStartId != nil
        )

        print("[KLD] Running forced decode (no KV quant) for \(quantizedData.tokenIds.count) tokens...")

        // Manual prefill + forced decode (bypasses TokenIterator to avoid off-by-one:
        // TokenIterator.init does prefill AND samples the first token, consuming position-0
        // logits. We need those logits to compute KLD for token[0].)
        return try await container.perform { ctx in
            // 1. Create cache and prefill — NO KV quantization
            var cache = ctx.model.newCache(parameters: params)
            var state: LMOutput.State? = nil
            var logits: MLXArray

            switch try ctx.model.prepare(input, cache: cache, windowSize: params.prefillStepSize) {
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

            // 2. Forced decode: compute baseline (unquantized) log prob for each token
            let tokenIds = quantizedData.tokenIds
            let quantizedLogProbs = quantizedData.logProbs
            let phases = quantizedData.phases
            let tokenCount = min(tokenIds.count, effectiveMaxTokens)

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

    private func loadPrompt(tokenCount: Int) throws -> String {
        let filename = "prompt_\(tokenCount)"
        guard let url = Bundle.module.url(forResource: filename, withExtension: "txt") else {
            throw BenchmarkError("Missing test resource: \(filename).txt")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func loadWikitext2() throws -> String {
        guard let url = Bundle.module.url(forResource: "wikitext2_test", withExtension: "txt") else {
            throw BenchmarkError("Missing test resource: wikitext2_test.txt")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}
