import Foundation
import Hub
import Testing
import MLX
import MLXNN
import MLXLMCommon
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

    var inThinkingPhase: Bool
    var thinkingTokenCount: Int = 0

    init(thinkStartTokenId: Int32, thinkEndTokenId: Int32, maxThinkingTokens: Int, prefilled: Bool = false) {
        self.thinkStartTokenId = thinkStartTokenId
        self.thinkEndTokenId = thinkEndTokenId
        self.maxThinkingTokens = maxThinkingTokens
        self.initialThinkingPhase = prefilled
        self.inThinkingPhase = prefilled
    }

    mutating func prompt(_ prompt: MLXArray) {
        inThinkingPhase = initialThinkingPhase
        thinkingTokenCount = 0
    }

    func process(logits: MLXArray) -> MLXArray {
        guard inThinkingPhase, thinkingTokenCount >= maxThinkingTokens else { return logits }
        // Budget exceeded: force </think> by boosting its logit to dominate softmax
        var modified = logits
        if logits.ndim == 1 {
            modified[Int(thinkEndTokenId)] = MLXArray(Float.infinity)
            modified[Int(thinkStartTokenId)] = MLXArray(-Float.infinity)
        } else {
            modified[0..., Int(thinkEndTokenId)] = MLXArray(Float.infinity)
            modified[0..., Int(thinkStartTokenId)] = MLXArray(-Float.infinity)
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
    case none                           // No KV quantization
    case affine(bits: Int)              // MLX affine quantization (kvBits)
    case turbo(bits: Int)               // TurboQuant MSE quantization (kvScheme)

    var description: String {
        switch self {
        case .none: return "no-quant"
        case .affine(let b): return "affine-\(b)"
        case .turbo(let b): return "turbo\(b)"
        }
    }

    var kvBits: Int? {
        if case .affine(let b) = self { return b }
        return nil
    }

    var kvScheme: String? {
        if case .turbo(let b) = self { return "turbo\(b)" }
        return nil
    }

    var quantizedKVStart: Int {
        switch self {
        case .none: return 0
        case .affine: return 512
        case .turbo: return 0
        }
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
private enum BenchEnv {
    static var contextBenchEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_CONTEXT_BENCH"] == "1"
    }
    static var speedBenchEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_SPEED_BENCH"] == "1"
    }
    static var baselineMode: Bool {
        ProcessInfo.processInfo.environment["MLX_BENCH_BASELINE"] == "1"
    }
    /// Target quantization: "bf16", "8bit", "4bit", "nvfp4", "mxfp4" (default: "4bit")
    static var quantization: String {
        ProcessInfo.processInfo.environment["MLX_BENCH_QUANT"] ?? "4bit"
    }
    /// Custom model repo ID (overrides registry)
    static var customModel: String? {
        ProcessInfo.processInfo.environment["MLX_BENCH_MODEL"]
    }
    /// Context size filter (comma-separated)
    static var contextFilter: Set<Int>? {
        guard let filter = ProcessInfo.processInfo.environment["MLX_BENCH_CONTEXT"], !filter.isEmpty else {
            return nil
        }
        return Set(filter.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
    }
}

// MARK: - Model Cache

/// Cache loaded models across test runs to avoid reloading per context size.
/// Safe because InferenceSpeedTests uses .serialized (sequential execution).
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

/// Inference speed benchmarks for production models.
///
/// Context tests: MLX_CONTEXT_BENCH=1 swift test --filter InferenceSpeedTests
/// Speed tests:   MLX_SPEED_BENCH=1 swift test --filter InferenceSpeedTests
/// Single model:  MLX_CONTEXT_BENCH=1 swift test --filter "qwen35_9B"
/// Quantization:  MLX_BENCH_QUANT=8bit MLX_CONTEXT_BENCH=1 swift test ...
/// Baseline:      MLX_BENCH_BASELINE=1 MLX_CONTEXT_BENCH=1 swift test ...
/// Custom model:  MLX_BENCH_MODEL=mlx-community/Qwen3.5-9B-4bit MLX_CONTEXT_BENCH=1 swift test --filter "custom"
@Suite("Inference Speed Benchmarks", .serialized)
struct InferenceSpeedTests {

    // MARK: - Constants

    static let contextSizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    static let toolQuery = "What is the current date and time?"
    static let multiTurnMessages: [Message] = [
        ["role": "user", "content": "What's the capital of France?"],
        ["role": "assistant", "content": "The capital of France is Paris."],
        ["role": "user", "content": "What about Germany?"],
        ["role": "assistant", "content": "The capital of Germany is Berlin."],
        ["role": "user", "content": "Which one has a larger population?"],
    ]
    static let minimalSystemPrompt = "You are a helpful assistant. Keep responses concise."

    // MARK: - Qwen3.5 0.8B

    @Test(arguments: contextSizes)
    func qwen35_08B(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_08B, kv: .none, contextSize: contextSize)
    }

    // MARK: - Qwen3.5 2B

    @Test(arguments: contextSizes)
    func qwen35_2B(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_2B, kv: .none, contextSize: contextSize)
    }

    // MARK: - Qwen3.5 4B

    @Test(arguments: contextSizes)
    func qwen35_4B(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_4B, kv: .none, contextSize: contextSize)
    }

    // MARK: - Qwen3.5 9B

    @Test(arguments: contextSizes)
    func qwen35_9B_noQuant(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_9B, kv: .none, contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_9B_affine4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_9B, kv: .affine(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_9B_turbo4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_9B, kv: .turbo(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_9B_turbo3(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_9B, kv: .turbo(bits: 3), contextSize: contextSize)
    }

    @Test func qwen35_9B_tool_noQuant() async throws {
        try skipUnlessEnabled()
        try await benchmarkTool(family: ModelRegistry.qwen35_9B, kv: .none)
    }

    @Test func qwen35_9B_multiTurn_noQuant() async throws {
        try skipUnlessEnabled()
        try await benchmarkMultiTurn(family: ModelRegistry.qwen35_9B, kv: .none)
    }

    // MARK: - Qwen3.5 27B

    @Test(arguments: contextSizes)
    func qwen35_27B_noQuant(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_27B, kv: .none, contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_27B_affine4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_27B, kv: .affine(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_27B_turbo4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_27B, kv: .turbo(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_27B_turbo3(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_27B, kv: .turbo(bits: 3), contextSize: contextSize)
    }

    @Test func qwen35_27B_tool_noQuant() async throws {
        try skipUnlessEnabled()
        try await benchmarkTool(family: ModelRegistry.qwen35_27B, kv: .none)
    }

    @Test func qwen35_27B_multiTurn_noQuant() async throws {
        try skipUnlessEnabled()
        try await benchmarkMultiTurn(family: ModelRegistry.qwen35_27B, kv: .none)
    }

    // MARK: - Qwen3.5 35B A3B (MoE)

    @Test(arguments: contextSizes)
    func qwen35_35B_A3B_noQuant(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_35B_A3B, kv: .none, contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_35B_A3B_affine4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_35B_A3B, kv: .affine(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_35B_A3B_turbo4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_35B_A3B, kv: .turbo(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func qwen35_35B_A3B_turbo3(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.qwen35_35B_A3B, kv: .turbo(bits: 3), contextSize: contextSize)
    }

    // MARK: - GPT-OSS 20B

    @Test(arguments: contextSizes)
    func gptOSS20B_noQuant(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.gptOSS20B, kv: .none, contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func gptOSS20B_turbo4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.gptOSS20B, kv: .turbo(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func gptOSS20B_turbo3(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.gptOSS20B, kv: .turbo(bits: 3), contextSize: contextSize)
    }

    @Test func gptOSS20B_tool_noQuant() async throws {
        try skipUnlessEnabled()
        try await benchmarkTool(family: ModelRegistry.gptOSS20B, kv: .none)
    }

    @Test func gptOSS20B_multiTurn_noQuant() async throws {
        try skipUnlessEnabled()
        try await benchmarkMultiTurn(family: ModelRegistry.gptOSS20B, kv: .none)
    }

    // MARK: - Nemotron 30B A3B

    @Test(arguments: contextSizes)
    func nemotron30B_noQuant(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.nemotron30B, kv: .none, contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func nemotron30B_affine4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.nemotron30B, kv: .affine(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func nemotron30B_turbo4(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.nemotron30B, kv: .turbo(bits: 4), contextSize: contextSize)
    }

    @Test(arguments: contextSizes)
    func nemotron30B_turbo3(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        try await benchmarkContext(family: ModelRegistry.nemotron30B, kv: .turbo(bits: 3), contextSize: contextSize)
    }

    // MARK: - Custom Model

    @Test(arguments: contextSizes)
    func custom(contextSize: Int) async throws {
        try skipUnlessContextBenchEnabled()
        guard let repoId = BenchEnv.customModel else { return }
        let family = ModelRegistry.customFamily(repoId: repoId)
        try await benchmarkContext(family: family, kv: .none, contextSize: contextSize)
    }

    // MARK: - Scenario Runners

    /// Context-length benchmark using pre-generated prompt files.
    private func benchmarkContext(
        family: ModelFamily,
        kv: KVCacheConfig,
        contextSize: Int
    ) async throws {
        if shouldSkipContext(contextSize) { return }
        let (variant, repoId) = try await resolveVariant(family: family)
        let prompt = try loadPrompt(tokenCount: contextSize)
        let label = "\(family.name) [\(variant.quantization)] — \(contextSize) tokens [\(kv)]"

        try await runBenchmark(
            family: family, variant: variant, repoId: repoId,
            kv: kv, label: label, contextSize: contextSize,
            messages: [["role": "user", "content": prompt]],
            systemPrompt: nil, includeTools: false, maxTokens: 600
        )
    }

    /// Tool call benchmark.
    private func benchmarkTool(family: ModelFamily, kv: KVCacheConfig) async throws {
        let (variant, repoId) = try await resolveVariant(family: family)
        let label = "\(family.name) [\(variant.quantization)] — tool [\(kv)]"
        try await runBenchmark(
            family: family, variant: variant, repoId: repoId,
            kv: kv, label: label, contextSize: 0,
            messages: [["role": "user", "content": Self.toolQuery]],
            systemPrompt: Self.minimalSystemPrompt, includeTools: true, maxTokens: 600
        )
    }

    /// 3-message multi-turn conversation benchmark.
    private func benchmarkMultiTurn(family: ModelFamily, kv: KVCacheConfig) async throws {
        let (variant, repoId) = try await resolveVariant(family: family)
        let label = "\(family.name) [\(variant.quantization)] — multi-turn [\(kv)]"
        try await runBenchmark(
            family: family, variant: variant, repoId: repoId,
            kv: kv, label: label, contextSize: 0,
            messages: Self.multiTurnMessages,
            systemPrompt: Self.minimalSystemPrompt, includeTools: false, maxTokens: 600
        )
    }

    // MARK: - Variant Resolution

    /// Resolve which model variant to use based on env vars (baseline mode, quant selection).
    private func resolveVariant(family: ModelFamily) async throws -> (ModelVariant, String) {
        if BenchEnv.baselineMode {
            let hw = SystemInfo.hardware()
            let variant = try await family.selectBaseline(hardware: hw)
            return (variant, variant.repoId)
        }

        let quant = BenchEnv.quantization
        guard let variant = family.variant(for: quant) else {
            // Fall back to first available variant
            let fallback = family.variants[0]
            print("[BENCH] No \(quant) variant for \(family.name), using \(fallback.quantization)")
            return (fallback, fallback.repoId)
        }
        return (variant, variant.repoId)
    }

    // MARK: - Core Benchmark Runner

    private func runBenchmark(
        family: ModelFamily,
        variant: ModelVariant,
        repoId: String,
        kv: KVCacheConfig,
        label: String,
        contextSize: Int,
        messages: [Message],
        systemPrompt: String?,
        includeTools: Bool,
        maxTokens: Int
    ) async throws {
        let hr = String(repeating: "=", count: 80)
        print("\n\(hr)")
        print("[BENCH] \(label)")
        print("[BENCH] Model: \(repoId)")
        print("[BENCH] Quantization: \(variant.quantization)")
        print("[BENCH] KV: \(kv)")
        print(hr)

        // 1. Load model (cached across test runs)
        let container: ModelContainer
        let loadStart = Date()
        if let cached = ModelCache.shared.get(repoId) {
            container = cached
            print("[BENCH] Model loaded from cache")
        } else {
            let modelConfig = family.extraEOSTokens.isEmpty
                ? ModelConfiguration(id: repoId)
                : ModelConfiguration(id: repoId, extraEOSTokens: Set(family.extraEOSTokens))

            container = try await LLMModelFactory.shared.loadContainer(configuration: modelConfig) { p in
                if p.fractionCompleted < 0.01 || p.fractionCompleted > 0.99 {
                    print("[BENCH] Loading: \(String(format: "%.0f", p.fractionCompleted * 100))%")
                }
            }
            ModelCache.shared.set(repoId, container)
            print("[BENCH] Model loaded in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")
        }

        // Look up thinking phase token IDs for models that support thinking
        let (thinkStartId, thinkEndId): (Int32?, Int32?) = family.supportsThinking
            ? await container.perform { ctx in
                let startId = ctx.tokenizer.convertTokenToId("<think>").map { Int32($0) }
                let endId = ctx.tokenizer.convertTokenToId("</think>").map { Int32($0) }
                return (startId, endId)
              }
            : (nil, nil)
        if let s = thinkStartId, let e = thinkEndId {
            print("[BENCH] Thinking tokens: <think>=\(s), </think>=\(e), budget=300")
        }

        // Build thinking budget processor if model supports thinking mode
        let thinkingBudget = 300  // max thinking tokens before forcing </think>
        let budgetProcessor: ThinkingBudgetProcessor? = thinkStartId.flatMap { startId in
            thinkEndId.map { endId in
                ThinkingBudgetProcessor(
                    thinkStartTokenId: startId,
                    thinkEndTokenId: endId,
                    maxThinkingTokens: thinkingBudget,
                    prefilled: true  // <think> is in the prompt, not generated
                )
            }
        }

        // 2. Build messages. Prompt files are pre-sized so that content + template overhead
        // lands just under contextSize tokens. No runtime trimming needed.
        var allMessages: [Message] = []
        if let sys = systemPrompt {
            allMessages.append(["role": "system", "content": sys])
        }
        allMessages.append(contentsOf: messages)

        // Force thinking mode via assistant prefill: append a partial assistant turn
        // starting with <think> so the model must continue in thinking mode.
        if thinkStartId != nil {
            allMessages.append(["role": "assistant", "content": "<think>\n"])
        }

        let tools: [ToolSpec]? = includeTools ? [MockTools.shellToolSpec()] : nil
        let userInput = UserInput(
            prompt: .messages(allMessages),
            tools: tools
        )

        // 3. Prepare input
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

        // 4. Generate
        // maxTokens = thinking budget + response budget (300 + 200 = 500 + headroom)
        let effectiveMaxTokens = thinkStartId != nil ? thinkingBudget + maxTokens : maxTokens
        let additionalProcessors: [any LogitProcessor] = budgetProcessor.map { [$0] } ?? []

        let params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
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
            thinkingPhasePrefilled: thinkStartId != nil
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

        let stream = try await container.generate(input: lmInput, parameters: params, wiredMemoryTicket: ticket)
        for try await generation in stream {
            if case .info(let info) = generation {
                completionInfo = info
                continue
            }
            if case .toolCall(_) = generation { continue }
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

        let thinkingPerplexity = completionInfo?.thinkingPerplexity
        let generationPerplexity = completionInfo?.generationPerplexity

        // 5. Report
        let scenario: String
        if label.contains("tool") { scenario = "tool-call" }
        else if label.contains("multi-turn") { scenario = "multi-turn" }
        else { scenario = "context" }

        print("\n[BENCH] === RESULTS: \(label) ===")
        print("[BENCH] Scenario: \(scenario)")
        print("[BENCH] Input Target: \(contextSize) tokens, Prompt Tokens: \(prefillTokens) (after template)")
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
        print("[BENCH] GPU Baseline: \(formatBytes(baselineGPU))")
        print("[BENCH] GPU Peak: \(formatBytes(peakGPU))")
        print("[BENCH] KV Delta: \(formatBytes(kvDelta))")
        print("[BENCH] Output: \(String(outputText.prefix(150)))")

        print(hr + "\n")

        // 6. Write to markdown file
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
            thinkingKLD: nil,
            generationKLD: nil,
            baselineGPU: baselineGPU,
            peakGPU: Int(peakGPU),
            kvDelta: kvDelta,
            outputPreview: outputText,
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

    // MARK: - Utilities

    private func skipUnlessEnabled() throws {
        try #require(BenchEnv.speedBenchEnabled, "Skipped — set MLX_SPEED_BENCH=1 to run")
    }

    private func skipUnlessContextBenchEnabled() throws {
        try #require(BenchEnv.contextBenchEnabled, "Skipped — set MLX_CONTEXT_BENCH=1 to run")
    }

    private func shouldSkipContext(_ contextSize: Int) -> Bool {
        guard let allowed = BenchEnv.contextFilter else { return false }
        return !allowed.contains(contextSize)
    }

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
}
