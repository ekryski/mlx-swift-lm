import Foundation
import MLXLMCommon

/// A specific quantization variant of a model with its HuggingFace repo ID.
struct ModelVariant {
    let quantization: String   // "bf16", "8bit", "4bit", "nvfp4", "mxfp4"
    let repoId: String
}

/// A model family with multiple quantization variants and generation parameters.
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
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"]
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
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"]
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
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"]
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
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"]
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
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"]
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
        extraEOSTokens: ["<|endoftext|>", "<|im_end|>"]
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
        extraEOSTokens: []
    )

    // MARK: - Nemotron

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
        extraEOSTokens: []
    )

    // MARK: - All Families

    static let allFamilies: [ModelFamily] = [
        qwen35_08B, qwen35_2B, qwen35_4B, qwen35_9B, qwen35_27B, qwen35_35B_A3B,
        gptOSS20B, nemotron30B,
    ]

    /// Look up a model family by its short name (CLI filter).
    static func family(named shortName: String) -> ModelFamily? {
        allFamilies.first { $0.shortName == shortName }
    }

    /// Create an ad-hoc family from a custom HuggingFace repo ID.
    static func customFamily(repoId: String) -> ModelFamily {
        ModelFamily(
            name: repoId, shortName: repoId,
            variants: [.init(quantization: "custom", repoId: repoId)],
            temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0,
            presencePenalty: nil, repetitionPenalty: nil,
            extraEOSTokens: []
        )
    }
}
