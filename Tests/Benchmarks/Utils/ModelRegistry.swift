import Foundation
import MLXLMCommon

/// A specific quantization variant of a model with its HuggingFace repo ID.
struct ModelVariant {
    let quantization: String   // "bf16", "8bit", "4bit", "nvfp4", "mxfp4"
    let repoId: String
}

/// A model family with multiple quantization variants and generation parameters.
/// Thinking token configuration for models that support reasoning traces.
///
/// Two styles are represented:
/// - `bracket` — a single start token pairs with a single end token
///   (Qwen `<think>…</think>`, Gemma 4 `<|channel>…<channel|>`).
/// - `harmonyChannel` — channel transitions via a marker token plus a
///   channel-name token (GPT-OSS `<|channel|>analysis<|message|>…`).
struct ThinkingConfig {
    enum Style {
        case bracket(start: String, end: String)
        case harmonyChannel(
            marker: String,
            thinkingChannels: [String],
            generationChannels: [String],
            /// Token-string sequence forced into the output stream when the
            /// harmony budget processor decides reasoning has run long enough.
            /// For GPT-OSS: `<|end|>`, `<|start|>`, `assistant`, `<|channel|>`,
            /// `final`, `<|message|>` — closing the analysis message and
            /// opening the final-channel visible answer. All six are resolved
            /// to token IDs at benchmark start-up and handed to the processor.
            transitionSequence: [String]
        )
    }

    let style: Style

    /// Assistant prefill to inject before generation to trigger thinking mode.
    /// Qwen-style models use "<think>\n" prefill; Gemma 4 and GPT-OSS use their
    /// chat template's enable_thinking / reasoning_effort controls instead.
    let assistantPrefill: String

    /// Qwen-style: <think>...</think>
    static let qwen = ThinkingConfig(
        style: .bracket(start: "<think>", end: "</think>"),
        assistantPrefill: "<think>\n"
    )

    /// Gemma 4 style: <|channel>thought\n...<channel|>
    /// Thinking is triggered via enable_thinking=true in the chat template.
    static let gemma4 = ThinkingConfig(
        style: .bracket(start: "<|channel>", end: "<channel|>"),
        assistantPrefill: ""
    )

    /// GPT-OSS / harmony format: <|channel|>analysis<|message|>… for reasoning,
    /// <|channel|>final<|message|>… for the visible answer. The model emits
    /// the transitions itself; no assistant prefill is needed. `commentary`
    /// channel is treated as a generation-phase channel (visible tool/meta
    /// commentary rather than chain-of-thought).
    static let harmony = ThinkingConfig(
        style: .harmonyChannel(
            marker: "<|channel|>",
            thinkingChannels: ["analysis"],
            generationChannels: ["final", "commentary"],
            transitionSequence: [
                "<|end|>", "<|start|>", "assistant",
                "<|channel|>", "final", "<|message|>"
            ]
        ),
        assistantPrefill: ""
    )
}

struct ModelFamily {
    let name: String           // Display name: "Qwen3.5 27B"
    let shortName: String      // CLI filter: "qwen35-27b"
    let variants: [ModelVariant]
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
    let presencePenalty: Float?
    let repetitionPenalty: Float?
    let extraEOSTokens: [String]
    /// Whether this model supports thinking mode.
    /// When true, benchmarks will force thinking via assistant prefill and track
    /// think/gen perplexity separately.
    let supportsThinking: Bool
    /// Thinking token configuration. Defaults to Qwen-style if not specified.
    var thinkingConfig: ThinkingConfig = .qwen
    /// Reasoning effort hint passed to GenerateParameters (e.g., "low", "medium", "high").
    /// Used by models like GPT-OSS that support configurable reasoning depth.
    let reasoningEffort: String?

    /// Get the variant for a specific quantization, or nil if not available.
    func variant(for quantization: String) -> ModelVariant? {
        variants.first { $0.quantization == quantization }
    }

    /// Ordered preference for baseline selection: bf16 → 8bit → 4bit
    static let baselinePreference = ["bf16", "8bit", "4bit"]

    /// Select the highest-fidelity variant that fits in memory.
    func selectBaseline(hardware: SystemInfo.Hardware) async throws -> ModelVariant {
        for quant in Self.baselinePreference {
            guard let v = variant(for: quant) else { continue }
            do {
                let size = try await SystemInfo.estimateModelSize(repo: v.repoId)
                if SystemInfo.fitsInMemory(modelSizeBytes: size, hardware: hardware) {
                    return v
                }
                print("[BENCH] Skipping \(name) \(quant) (\(SystemInfo.formatGB(size))) — exceeds \(String(format: "%.0f", hardware.gpuMemoryLimitGB))GB GPU memory")
            } catch {
                print("[BENCH] Could not check size for \(v.repoId): \(error)")
            }
        }

        // Try all remaining variants as last resort
        for v in variants where !Self.baselinePreference.contains(v.quantization) {
            do {
                let size = try await SystemInfo.estimateModelSize(repo: v.repoId)
                if SystemInfo.fitsInMemory(modelSizeBytes: size, hardware: hardware) {
                    return v
                }
            } catch {
                continue
            }
        }

        let smallest = variants.last!
        let size = (try? await SystemInfo.estimateModelSize(repo: smallest.repoId)) ?? 0
        throw BenchmarkError(
            "\(name) won't fit: smallest variant (\(smallest.quantization)) is \(SystemInfo.formatGB(size)) "
            + "but system has \(String(format: "%.0f", hardware.gpuMemoryLimitGB))GB GPU memory"
        )
    }
}

struct BenchmarkError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// Central registry of all benchmark model families and their quantization variants.
enum ModelRegistry {

    // MARK: - Qwen3.5 Family

    static let qwen35_08B = ModelFamily(
        name: "Qwen3.5 0.8B", shortName: "qwen35-0.8b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen3.5-0.8B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3.5-0.8B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.5-0.8B-4bit"),
            .init(quantization: "nvfp4", repoId: "mlx-community/Qwen3.5-0.8B-nvfp4"),
            .init(quantization: "mxfp4", repoId: "mlx-community/Qwen3.5-0.8B-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    static let qwen35_2B = ModelFamily(
        name: "Qwen3.5 2B", shortName: "qwen35-2b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen3.5-2B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3.5-2B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.5-2B-4bit"),
            .init(quantization: "nvfp4", repoId: "mlx-community/Qwen3.5-2B-nvfp4"),
            .init(quantization: "mxfp4", repoId: "mlx-community/Qwen3.5-2B-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    static let qwen35_4B = ModelFamily(
        name: "Qwen3.5 4B", shortName: "qwen35-4b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen3.5-4B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3.5-4B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.5-4B-4bit"),
            .init(quantization: "nvfp4", repoId: "mlx-community/Qwen3.5-4B-nvfp4"),
            .init(quantization: "mxfp4", repoId: "mlx-community/Qwen3.5-4B-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    static let qwen35_9B = ModelFamily(
        name: "Qwen3.5 9B", shortName: "qwen35-9b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen3.5-9B-MLX-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3.5-9B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.5-9B-4bit"),
            .init(quantization: "nvfp4", repoId: "mlx-community/Qwen3.5-9B-nvfp4"),
            .init(quantization: "mxfp4", repoId: "mlx-community/Qwen3.5-9B-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    static let qwen35_27B = ModelFamily(
        name: "Qwen3.5 27B", shortName: "qwen35-27b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen3.5-27B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3.5-27B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.5-27B-4bit"),
            .init(quantization: "nvfp4", repoId: "dumtjul/Qwen3.5-27B-mlx-nvfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    static let qwen36_27B = ModelFamily(
        name: "Qwen3.6 27B", shortName: "qwen36-27b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.6-27B-4bit"),
            .init(quantization: "4bit-ud", repoId: "unsloth/Qwen3.6-27B-UD-MLX-4bit"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    static let qwen35_35B_A3B = ModelFamily(
        name: "Qwen3.5 35B A3B", shortName: "qwen35-35b-a3b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen3.5-35B-A3B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3.5-35B-A3B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3.5-35B-A3B-4bit"),
            .init(quantization: "nvfp4", repoId: "RepublicOfKorokke/Qwen3.5-35B-A3B-mlx-vlm-nvfp4"),
            .init(quantization: "mxfp4", repoId: "RepublicOfKorokke/Qwen3.5-35B-A3B-mlx-vlm-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: 1.5, repetitionPenalty: 1.0,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: true, reasoningEffort: nil
    )

    // MARK: - GPT-OSS

    static let gptOSS20B = ModelFamily(
        name: "GPT-OSS 20B", shortName: "gpt-oss-20b",
        variants: [
            .init(quantization: "bf16", repoId: "sjgdr/gpt-oss-20b-mlx-fp16"),
            .init(quantization: "4bit", repoId: "loan-star/gpt-oss-20b-mlx-4Bit"),
            .init(quantization: "mxfp4", repoId: "mlx-community/gpt-oss-20b-MXFP4-Q8"),
        ],
        temperature: 0.8, topP: 0.8, topK: 0, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: true, thinkingConfig: .harmony, reasoningEffort: "medium"
    )

    // MARK: - Nemotron (Cascade 2)

    static let nemotron30B = ModelFamily(
        name: "Nemotron 30B A3B", shortName: "nemotron-30b-a3b",
        variants: [
            .init(quantization: "8bit", repoId: "mlx-community/Nemotron-Cascade-2-30B-A3B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Nemotron-Cascade-2-30B-A3B-4bit"),
            .init(quantization: "nvfp4", repoId: "RepublicOfKorokke/Nemotron-Cascade-2-30B-A3B-mlx-nvfp4"),
            .init(quantization: "mxfp4", repoId: "RepublicOfKorokke/Nemotron-Cascade-2-30B-A3B-mlx-mxfp4"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: true, reasoningEffort: nil
    )

    // MARK: - Gemma 4

    static let gemma4_E2B = ModelFamily(
        name: "Gemma 4 E2B", shortName: "gemma4-e2b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/gemma-4-e2b-it-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/gemma-4-e2b-it-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/gemma-4-e2b-it-4bit"),
            .init(quantization: "mxfp4", repoId: "mlx-community/gemma-4-e2b-it-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: true, thinkingConfig: .gemma4, reasoningEffort: nil
    )

    static let gemma4_E4B = ModelFamily(
        name: "Gemma 4 E4B", shortName: "gemma4-e4b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/gemma-4-e4b-it-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/gemma-4-e4b-it-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/gemma-4-e4b-it-4bit"),
            .init(quantization: "mxfp4", repoId: "mlx-community/gemma-4-e4b-it-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: true, thinkingConfig: .gemma4, reasoningEffort: nil
    )

    static let gemma4_26B_A4B = ModelFamily(
        name: "Gemma 4 26B A4B", shortName: "gemma4-26b-a4b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/gemma-4-26b-a4b-it-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/gemma-4-26b-a4b-it-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/gemma-4-26b-a4b-it-4bit"),
            .init(quantization: "mxfp4", repoId: "mlx-community/gemma-4-26b-a4b-it-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: true, thinkingConfig: .gemma4, reasoningEffort: nil
    )

    static let gemma4_31B = ModelFamily(
        name: "Gemma 4 31B", shortName: "gemma4-31b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/gemma-4-31b-it-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/gemma-4-31b-it-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/gemma-4-31b-it-4bit"),
            .init(quantization: "mxfp4", repoId: "mlx-community/gemma-4-31b-it-mxfp4"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: true, thinkingConfig: .gemma4, reasoningEffort: nil
    )

    // MARK: - LFM 2 Family

    static let lfm2_1_2B = ModelFamily(
        name: "LFM 2 1.2B", shortName: "lfm2-1.2b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/LFM2-1.2B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/LFM2-1.2B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/LFM2-1.2B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let lfm2_VL_1_6B = ModelFamily(
        name: "LFM 2 VL 1.6B", shortName: "lfm2vl-1.6b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/LFM2-VL-1.6B-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/LFM2-VL-1.6B-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/LFM2-VL-1.6B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - LFM 2.5 Family
    //
    // LFM 2.5 uses the same `LFM2.swift` / `LFM2VL.swift` Swift decoder
    // classes as LFM 2 (no separate architecture). Listed as distinct
    // ModelFamily entries because the model checkpoints diverge — LFM 2.5
    // ships fresh weights + a Pythonic-style chat / tool-call template
    // (see `ToolCallProcessor.infer(from:)` — both `lfm2*` and `lfm2.5*`
    // resolve to `.lfm2`). Worth covering both versions in the smoke
    // matrix for any quiet weight-key / template drift.

    static let lfm2_5_1_2B = ModelFamily(
        name: "LFM 2.5 1.2B Instruct", shortName: "lfm2.5-1.2b",
        variants: [
            .init(quantization: "bf16", repoId: "LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16"),
            .init(quantization: "4bit", repoId: "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let lfm2_5_VL_1_6B = ModelFamily(
        name: "LFM 2.5 VL 1.6B", shortName: "lfm2.5vl-1.6b",
        variants: [
            // Already wired into VLMModelFactory as `lfm2_5_vl_1_6B_4bit`.
            .init(quantization: "4bit", repoId: "mlx-community/LFM2.5-VL-1.6B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Llama (text-only)

    static let llama3_2_3B = ModelFamily(
        name: "Llama 3.2 3B Instruct", shortName: "llama3.2-3b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Llama-3.2-3B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|eot_id|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let llama3_1_8B = ModelFamily(
        name: "Llama 3.1 8B Instruct", shortName: "llama3.1-8b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|eot_id|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Phi 3 / Phi MoE

    static let phi3_5_mini = ModelFamily(
        name: "Phi 3.5 mini Instruct", shortName: "phi3.5-mini",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Phi-3.5-mini-instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|end|>", "<|endoftext|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let phi3_5_moe = ModelFamily(
        name: "Phi 3.5 MoE Instruct (16x3.8B)", shortName: "phi3.5-moe",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Phi-3.5-MoE-instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|end|>", "<|endoftext|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Olmo 2 / OlmoE

    static let olmo2_7B = ModelFamily(
        name: "OLMo 2 7B Instruct", shortName: "olmo2-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/OLMo-2-1124-7B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let olmoE_1B_7B = ModelFamily(
        name: "OLMoE 1B-7B Instruct (MoE)", shortName: "olmoe-1b-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Granite (3.x dense + 4.x hybrid)

    static let granite3_3_2B = ModelFamily(
        name: "Granite 3.3 2B Instruct", shortName: "granite3.3-2b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/granite-3.3-2b-instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    /// Granite 4 H Tiny — the GraniteMoeHybrid (Mamba 2 + attention) family.
    /// Spec 040 lights up state-replay rollback for n-gram speculative on
    /// this architecture (matches Nemotron-H / FalconH1 path).
    static let granite4_H_Tiny = ModelFamily(
        name: "Granite 4 H Tiny (hybrid)", shortName: "granite4-h-tiny",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/granite-4.0-h-tiny-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Falcon H1 (hybrid)

    /// FalconH1 — Mamba 2 + attention hybrid; spec 040 state-replay applies.
    static let falconH1_7B = ModelFamily(
        name: "Falcon H1 7B Instruct", shortName: "falcon-h1-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Falcon-H1-7B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Jamba (hybrid — Mamba + attention + MoE)
    //
    // Note: spec 040 has an open caveat for Jamba — `ssmStep` uses a 2D
    // `A_log` shape that doesn't match the Mamba state-replay kernel's
    // `[H]` ALog signature. Jamba falls back to vanilla TokenIterator
    // (no n-gram speculative) until the kernel-side shape generalisation
    // or Swift-side reformulation lands. Should still load + generate.

    static let jambaReasoning_3B = ModelFamily(
        name: "Jamba Reasoning 3B", shortName: "jamba-reasoning-3b",
        variants: [
            // Only bf16 ships on mlx-community; no 4bit variant as of 2026-05.
            .init(quantization: "bf16", repoId: "mlx-community/AI21-Jamba-Reasoning-3B-bf16"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Qwen 2.5 / Qwen 3 (text-only — distinct from Qwen 3.5 GDN)

    static let qwen25_7B = ModelFamily(
        name: "Qwen 2.5 7B Instruct", shortName: "qwen25-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Qwen2.5-7B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let qwen3_4B = ModelFamily(
        name: "Qwen 3 4B", shortName: "qwen3-4b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3-4B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let qwen3_30B_A3B = ModelFamily(
        name: "Qwen 3 30B A3B (MoE)", shortName: "qwen3-30b-a3b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3-30B-A3B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - SmolLM 3, MiMo, Exaone 4, Gemma 2, Gemma 3n text

    static let smollm3_3B = ModelFamily(
        name: "SmolLM 3 3B", shortName: "smollm3-3b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/SmolLM3-3B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>", "<|endoftext|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let mimo_7B = ModelFamily(
        name: "MiMo 7B SFT", shortName: "mimo-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/MiMo-7B-SFT-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    static let exaone4_1_2B = ModelFamily(
        name: "EXAONE 4.0 1.2B", shortName: "exaone4-1.2b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/exaone-4.0-1.2b-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    static let gemma2_2B = ModelFamily(
        name: "Gemma 2 2B Instruct", shortName: "gemma2-2b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/gemma-2-2b-it-4bit"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<end_of_turn>"],
        supportsThinking: false, reasoningEffort: nil
    )

    /// Gemma 3n text-only LM head variant. Multimodal Gemma 3n is separate;
    /// this row exercises the `Gemma3nText` text decoder by itself.
    static let gemma3n_E2B_text = ModelFamily(
        name: "Gemma 3n E2B (text-only)", shortName: "gemma3n-e2b-text",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/gemma-3n-E2B-it-lm-4bit"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<end_of_turn>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Cohere, Internlm 2, MiniCPM, Apertus, Starcoder 2, Baichuan M1

    static let cohereCommandR = ModelFamily(
        name: "Cohere Command R v01 (35B)", shortName: "cohere-command-r",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/c4ai-command-r-v01-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|END_OF_TURN_TOKEN|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let internlm2_5_7B = ModelFamily(
        name: "InternLM 2.5 7B Chat", shortName: "internlm2.5-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/internlm2_5-7b-chat-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let miniCPM3_4B = ModelFamily(
        name: "MiniCPM 3 4B", shortName: "minicpm3-4b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/MiniCPM3-4B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let apertus_8B = ModelFamily(
        name: "Apertus 8B Instruct", shortName: "apertus-8b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Apertus-8B-Instruct-2509-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    static let starcoder2_7B = ModelFamily(
        name: "Starcoder 2 7B", shortName: "starcoder2-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/starcoder2-7b-4bit"),
        ],
        temperature: 0.2, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    static let baichuanM1_14B = ModelFamily(
        name: "Baichuan M1 14B Instruct", shortName: "baichuan-m1-14b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Baichuan-M1-14B-Instruct-4bit-ft"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - LFM 2 MoE, GLM 4.5 MoE, ERNIE 4.5 MoE, OpenELM, MiMo V2

    /// LFM 2 8B A1B — canonical MoE checkpoint that exercises the
    /// `LFM2MoE.swift` Swift code path. The experimental LFM2-2.6B-Exp is
    /// also an LFM2MoE variant but the 8B/1B-A is the canonical Liquid
    /// release shape.
    static let lfm2_8B_A1B = ModelFamily(
        name: "LFM 2 8B A1B (MoE)", shortName: "lfm2-8b-a1b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/LFM2-8B-A1B-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    /// GLM 4.5 Air — 110B / 12B-A MoE; ~55GB at 4-bit. Marginal on 64GB
    /// M1 Max with KV cache + workspace; may OOM at long context. Listed
    /// for the GLM4MOE Swift code path; downsize to GLM 4 9B if it fails.
    static let glm45_Air = ModelFamily(
        name: "GLM 4.5 Air (110B/12B-A MoE)", shortName: "glm4.5-air",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/GLM-4.5-Air-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    /// ERNIE 4.5 21B A3B — Baidu's MoE with 3B activated; ~12GB at 4-bit.
    static let ernie4_5_21B_A3B = ModelFamily(
        name: "ERNIE 4.5 21B A3B (MoE)", shortName: "ernie4.5-21b-a3b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/ERNIE-4.5-21B-A3B-PT-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    /// Apple's OpenELM — small dense decoder.
    static let openELM_1_1B = ModelFamily(
        name: "OpenELM 1.1B Instruct", shortName: "openelm-1.1b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/OpenELM-1_1B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    /// MiMo V2 Flash — exercises the `MiMoV2Flash` Swift class (newer than
    /// the original MiMo).
    static let mimoV2_Flash = ModelFamily(
        name: "MiMo V2 Flash", shortName: "mimov2-flash",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/MiMo-V2-Flash-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Mistral 3 / Ministral 3 Family

    static let ministral3_3B = ModelFamily(
        name: "Ministral 3 3B", shortName: "ministral3-3b",
        variants: [
            .init(quantization: "bf16",
                repoId: "mlx-community/Ministral-3-3B-Instruct-2512-bf16"),
            .init(quantization: "8bit",
                repoId: "mlx-community/Ministral-3-3B-Instruct-2512-8bit"),
            .init(quantization: "4bit",
                repoId: "mlx-community/Ministral-3-3B-Instruct-2512-4bit"),
        ],
        temperature: 0.6, topP: 1.0, topK: 0, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    static let mistralSmall3_1_24B_VL = ModelFamily(
        name: "Mistral Small 3.1 24B (VL)", shortName: "mistral3vl-24b",
        variants: [
            .init(quantization: "bf16",
                repoId: "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-bf16"),
            .init(quantization: "8bit",
                repoId: "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit"),
            .init(quantization: "4bit",
                repoId: "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit"),
        ],
        temperature: 0.6, topP: 1.0, topK: 0, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Qwen 2 / 2.5 VL Family

    static let qwen2VL_2B = ModelFamily(
        name: "Qwen 2 VL 2B", shortName: "qwen2vl-2b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/Qwen2-VL-2B-Instruct-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen2-VL-2B-Instruct-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/Qwen2-VL-2B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let qwen25VL_3B = ModelFamily(
        name: "Qwen 2.5 VL 3B", shortName: "qwen25vl-3b",
        variants: [
            .init(quantization: "bf16",
                repoId: "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"),
            .init(quantization: "8bit",
                repoId: "mlx-community/Qwen2.5-VL-3B-Instruct-8bit"),
            .init(quantization: "4bit",
                repoId: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - FastVLM Family
    //
    // Apple's FastVLM uses a Qwen 2 backbone. Only bf16 ships on
    // mlx-community as of 2026-05; community 4-bit variants exist
    // (e.g. `InsightKeeper/FastVLM-0.5B-MLX-4bit`) but the config
    // schema diverges from mlx-community's bf16, so we list only
    // the canonical bf16 here.

    static let fastvlm_0_5B = ModelFamily(
        name: "FastVLM 0.5B", shortName: "fastvlm-0.5b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/FastVLM-0.5B-bf16"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Gemma 3 Family

    static let gemma3_1B = ModelFamily(
        name: "Gemma 3 1B", shortName: "gemma3-1b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/gemma-3-1b-it-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/gemma-3-1b-it-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/gemma-3-1b-it-qat-4bit"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<end_of_turn>"],
        supportsThinking: false, reasoningEffort: nil
    )

    /// Gemma 3 4B is text+vision (VLM). The QAT 4-bit variant is the
    /// canonical compressed checkpoint; bf16 / 8bit also available.
    static let gemma3_4B = ModelFamily(
        name: "Gemma 3 4B (VL)", shortName: "gemma3-4b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/gemma-3-4b-it-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/gemma-3-4b-it-8bit"),
            .init(quantization: "4bit", repoId: "mlx-community/gemma-3-4b-it-qat-4bit"),
        ],
        temperature: 1.0, topP: 0.95, topK: 64, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<end_of_turn>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - GLM 4

    static let glm4_9B = ModelFamily(
        name: "GLM 4 9B 0414", shortName: "glm4-9b",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/GLM-4-9B-0414-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/GLM-4-9B-0414-8bit"),
            .init(quantization: "6bit", repoId: "mlx-community/GLM-4-9B-0414-6bit"),
            .init(quantization: "4bit", repoId: "mlx-community/GLM-4-9B-0414-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    static let glm4_32B = ModelFamily(
        name: "GLM 4 32B 0414", shortName: "glm4-32b",
        variants: [
            .init(quantization: "8bit", repoId: "mlx-community/GLM-4-32B-0414-8bit"),
            .init(quantization: "6bit", repoId: "mlx-community/GLM-4-32B-Base-0414-6bit"),
            .init(quantization: "4bit", repoId: "mlx-community/GLM-4-32B-0414-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Pixtral (VLM)

    static let pixtral_12B = ModelFamily(
        name: "Pixtral 12B (VL)", shortName: "pixtral-12b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/pixtral-12b-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Idefics 3 (VLM)

    static let idefics3_8B = ModelFamily(
        name: "Idefics 3 8B (Llama 3) (VL)", shortName: "idefics3-8b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Idefics3-8B-Llama3-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|eot_id|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - SmolVLM (v1 + v2)

    /// SmolVLM v1 (`Idefics3`-derived backbone). Older but present in cache —
    /// useful for SmolVLM-specific Swift code paths.
    static let smolVLM_v1 = ModelFamily(
        name: "SmolVLM Instruct (v1)", shortName: "smolvlm-v1",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/SmolVLM-Instruct-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>", "<end_of_utterance>"],
        supportsThinking: false, reasoningEffort: nil
    )

    static let smolVLM2_2_2B = ModelFamily(
        name: "SmolVLM 2 2.2B Instruct", shortName: "smolvlm2-2.2b",
        variants: [
            // Single mlx-community variant — `-mlx` suffix denotes the 4-bit
            // canonical conversion; no separate `-4bit` tag.
            .init(quantization: "4bit", repoId: "mlx-community/SmolVLM2-2.2B-Instruct-mlx"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|im_end|>", "<end_of_utterance>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - MiMo VL (VLM)

    /// MiMo VL — vision-language MiMo variant. Distinct from `mimov2-flash`
    /// (text-only V2). Useful for the MiMo VLM dispatch path coverage.
    static let mimoVL_7B = ModelFamily(
        name: "MiMo VL 7B RL", shortName: "mimovl-7b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/MiMo-VL-7B-RL-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - Qwen 3 VL

    static let qwen3VL_4B = ModelFamily(
        name: "Qwen 3 VL 4B Instruct", shortName: "qwen3vl-4b",
        variants: [
            .init(quantization: "4bit", repoId: "mlx-community/Qwen3-VL-4B-Instruct-4bit"),
            .init(quantization: "8bit", repoId: "mlx-community/Qwen3-VL-4B-Instruct-8bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - GlmOcr (VLM — vision encoder + GLM 4 text decoder)

    static let glmOcr = ModelFamily(
        name: "GLM-OCR (VLM)", shortName: "glm-ocr",
        variants: [
            .init(quantization: "bf16", repoId: "mlx-community/GLM-OCR-bf16"),
            .init(quantization: "8bit", repoId: "mlx-community/GLM-OCR-8bit"),
            .init(quantization: "6bit", repoId: "mlx-community/GLM-OCR-6bit"),
            .init(quantization: "4bit", repoId: "mlx-community/GLM-OCR-4bit"),
        ],
        temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
        presencePenalty: nil, repetitionPenalty: nil,
        extraEOSTokens: [],
        supportsThinking: false, reasoningEffort: nil
    )

    // MARK: - All Families

    static let allFamilies: [ModelFamily] = [
        // ─── LLM-only ──────────────────────────────────────────────────────
        // Qwen 3.5 family (GatedDeltaNet hybrid)
        qwen35_08B, qwen35_2B, qwen35_4B, qwen35_9B, qwen35_27B,
        qwen36_27B,                 // Qwen3.6 / Qwen3Next arch
        qwen35_35B_A3B,             // Qwen35MoE
        // Qwen 2.5 / Qwen 3 (text-only, distinct from 3.5 GDN)
        qwen25_7B,                  // Qwen2 arch
        qwen3_4B,                   // Qwen3 arch
        qwen3_30B_A3B,              // Qwen3MoE arch
        // Other large MoE / hybrid
        gptOSS20B,
        nemotron30B,                // NemotronH (Mamba 2 hybrid)
        granite4_H_Tiny,            // GraniteMoeHybrid (Mamba 2 hybrid, small)
        falconH1_7B,                // FalconH1 (Mamba 2 hybrid)
        jambaReasoning_3B,          // Jamba (hybrid; spec 040 caveat)
        // Gemma 4 family
        gemma4_E2B, gemma4_E4B, gemma4_26B_A4B, gemma4_31B,
        // Other dense
        llama3_2_3B, llama3_1_8B,
        phi3_5_mini, phi3_5_moe,
        olmo2_7B, olmoE_1B_7B,
        granite3_3_2B,
        lfm2_1_2B, lfm2_5_1_2B,     // LFM 2 + 2.5 (same Swift class, different weights/template)
        ministral3_3B,
        gemma3_1B, gemma2_2B, gemma3n_E2B_text,
        glm4_9B, glm4_32B,
        smollm3_3B,
        mimo_7B,
        exaone4_1_2B,
        cohereCommandR,
        internlm2_5_7B,
        miniCPM3_4B,
        apertus_8B,
        starcoder2_7B,
        baichuanM1_14B,
        lfm2_8B_A1B,                // LFM2MoE
        glm45_Air,                  // GLM4MOE (large; may OOM)
        ernie4_5_21B_A3B,           // Ernie4_5
        openELM_1_1B,               // OpenELM
        mimoV2_Flash,               // MiMoV2Flash
        // ─── Vision-language (text + vision) ───────────────────────────────
        lfm2_VL_1_6B, lfm2_5_VL_1_6B,  // LFM 2 + 2.5 VL
        mistralSmall3_1_24B_VL,
        qwen2VL_2B, qwen25VL_3B, qwen3VL_4B,
        fastvlm_0_5B,
        gemma3_4B,
        glmOcr,
        pixtral_12B,
        idefics3_8B,
        smolVLM_v1, smolVLM2_2_2B,
        mimoVL_7B,                  // MiMo VL (VLM dispatch path)
    ]

    /// Alternate `--model` names → registry `shortName` (keys lowercased).
    private static let familyAliases: [String: String] = [
        // Nemotron Cascade 2 30B A3B (nemotron_h on MLX)
        "nemotron-cascade-2": "nemotron-30b-a3b",
        "nemotron-cascade2": "nemotron-30b-a3b",
        "nemotron-cascade-2-30b": "nemotron-30b-a3b",
        "nemotron-cascade-2-30b-a3b": "nemotron-30b-a3b",
        "nemotron-cascade2-30b-a3b": "nemotron-30b-a3b",
    ]

    /// Look up a model family by its short name (CLI filter) or a known alias.
    static func family(named shortName: String) -> ModelFamily? {
        let normalized = shortName.lowercased()
        let key = familyAliases[normalized] ?? normalized
        return allFamilies.first { $0.shortName == key }
    }

    /// Create an ad-hoc family from a custom HuggingFace repo ID.
    static func customFamily(repoId: String) -> ModelFamily {
        ModelFamily(
            name: repoId, shortName: repoId,
            variants: [.init(quantization: "custom", repoId: repoId)],
            temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
            presencePenalty: nil, repetitionPenalty: nil,
            extraEOSTokens: [],
            supportsThinking: false, reasoningEffort: nil
        )
    }
}
