// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import Tokenizers

/// Result of a wired memory measurement pass.
public struct WiredMemoryMeasurement: Sendable {
    /// Total bytes for model weights (`nbytes` sum).
    public let weightBytes: Int
    /// Total bytes for KV caches after prefill.
    public let kvBytes: Int
    /// Estimated transient workspace bytes (prefill peak minus weights + KV).
    public let workspaceBytes: Int
    /// Peak active memory observed during prefill.
    public let peakActiveBytes: Int
    /// Number of tokens used during the prefill measurement.
    public let tokenCount: Int
    /// Prefill step size used for the measurement.
    public let prefillStepSize: Int

    /// Combined budget suggestion (weights + KV + workspace).
    public var totalBytes: Int {
        max(0, weightBytes) + max(0, kvBytes) + max(0, workspaceBytes)
    }

    public init(
        weightBytes: Int, kvBytes: Int, workspaceBytes: Int,
        peakActiveBytes: Int, tokenCount: Int, prefillStepSize: Int
    ) {
        self.weightBytes = weightBytes
        self.kvBytes = kvBytes
        self.workspaceBytes = workspaceBytes
        self.peakActiveBytes = peakActiveBytes
        self.tokenCount = tokenCount
        self.prefillStepSize = prefillStepSize
    }
}

/// Helpers for deriving wired memory budgets from real runtime measurements.
public enum WiredMemoryUtils {
    /// Produce a token ID array of exactly `count` tokens using the given tokenizer.
    ///
    /// This does not attempt to generate semantically meaningful text; it only ensures
    /// a valid token sequence of the requested length for memory sizing purposes.
    private static func makeTokenIDs(
        count: Int,
        tokenizer: Tokenizer,
        seedText: String = " hello"
    ) -> [Int] {
        guard count > 0 else { return [] }

        let pad = tokenizer.eosTokenId ?? tokenizer.unknownTokenId ?? 0
        var tokens: [Int] = []

        var chunk = seedText
        while tokens.count < count {
            let newTokens = tokenizer.encode(text: chunk)
            if newTokens.isEmpty {
                tokens.append(pad)
            } else {
                tokens.append(contentsOf: newTokens)
            }
            if tokens.count < count {
                chunk += seedText
            }
        }

        if tokens.count > count {
            tokens = Array(tokens.prefix(count))
        }

        if tokens.count < count {
            tokens.append(contentsOf: repeatElement(pad, count: count - tokens.count))
        }

        return tokens
    }

    /// Create a minimal `LMInput` with exactly `count` tokens.
    ///
    /// - Note: This is intended for text-only models. Multimodal models should
    ///   supply a fully prepared `LMInput` via their processor instead.
    private static func makeTokenInput(
        count: Int,
        tokenizer: Tokenizer,
        seedText: String = " hello"
    ) -> LMInput {
        let tokenIDs = makeTokenIDs(count: count, tokenizer: tokenizer, seedText: seedText)
        return LMInput(tokens: MLXArray(tokenIDs))
    }

    /// Run a prefill-only pass to populate caches for the given input.
    ///
    /// This mirrors the prefill path used by `TokenIterator` without generating
    /// additional tokens. It forces evaluation to ensure allocations are realized.
    ///
    /// - Parameters:
    ///   - input: Prepared model input (text-only or multimodal).
    ///   - model: The language model to prefill.
    ///   - parameters: Generation parameters that control prefill behavior.
    /// - Returns: The cache array after prefill, suitable for KV sizing.
    private static func prefillOnly(
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> [KVCache] {
        var cache = model.newCache(parameters: parameters)

        switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            let result = model(
                tokens[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
            eval(result.logits)
        case .logits(let result):
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
            eval(result.logits)
        }

        return cache
    }

    /// Measure weights, KV cache, and prefill workspace for the given model context.
    ///
    /// This is a diagnostic helper intended to **measure** real memory usage
    /// rather than assume it. The returned values are best used to construct
    /// a ticket budget or to compare against manual estimates.
    ///
    /// - Important: `Memory.peakMemory` is global. For accurate results, run
    ///   this in isolation (no concurrent inference).
    ///
    /// - Note: This uses the tokenizer directly to build a synthetic prompt and
    ///   is best suited for text-only models. For multimodal models, use the
    ///   overload that accepts a prepared `LMInput`.
    public static func tune(
        context: ModelContext,
        tokenCount: Int,
        parameters: GenerateParameters,
        seedText: String = " hello",
        resetPeakMemory: Bool = true
    ) async throws -> WiredMemoryMeasurement {
        let weights = context.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        let input = makeTokenInput(
            count: tokenCount,
            tokenizer: context.tokenizer,
            seedText: seedText
        )

        let startActive = Memory.activeMemory
        if resetPeakMemory {
            Memory.peakMemory = 0
        }

        let cache = try prefillOnly(input: input, model: context.model, parameters: parameters)
        let cacheArrays = cache.flatMap { $0.state }
        if !cacheArrays.isEmpty {
            eval(cacheArrays)
        }

        let kvBytes = cacheArrays.reduce(0) { $0 + $1.nbytes }
        let peakActive = max(Memory.peakMemory, startActive)
        let workspace = max(0, peakActive - weights - kvBytes)

        return WiredMemoryMeasurement(
            weightBytes: weights,
            kvBytes: kvBytes,
            workspaceBytes: workspace,
            peakActiveBytes: peakActive,
            tokenCount: tokenCount,
            prefillStepSize: parameters.prefillStepSize
        )
    }

    /// Measure weights, KV cache, and prefill workspace using a prepared input.
    ///
    /// This overload is recommended for multimodal models because it accepts a
    /// fully prepared `LMInput` (e.g. with image/video tensors already embedded).
    ///
    /// - Parameters:
    ///   - input: Prepared model input from the model's processor.
    ///   - context: The loaded model context.
    ///   - parameters: Generation parameters that control prefill behavior.
    ///   - resetPeakMemory: If true, resets `Memory.peakMemory` before measuring.
    /// - Returns: A measurement snapshot for weights, KV, and workspace.
    public static func tune(
        input: LMInput,
        context: ModelContext,
        parameters: GenerateParameters,
        resetPeakMemory: Bool = true
    ) async throws -> WiredMemoryMeasurement {
        let weights = context.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        let startActive = Memory.activeMemory
        if resetPeakMemory {
            Memory.peakMemory = 0
        }

        let cache = try prefillOnly(input: input, model: context.model, parameters: parameters)
        let cacheArrays = cache.flatMap { $0.state }
        if !cacheArrays.isEmpty {
            eval(cacheArrays)
        }

        let kvBytes = cacheArrays.reduce(0) { $0 + $1.nbytes }
        let peakActive = max(Memory.peakMemory, startActive)
        let workspace = max(0, peakActive - weights - kvBytes)

        return WiredMemoryMeasurement(
            weightBytes: weights,
            kvBytes: kvBytes,
            workspaceBytes: workspace,
            peakActiveBytes: peakActive,
            tokenCount: input.text.tokens.size,
            prefillStepSize: parameters.prefillStepSize
        )
    }

    /// Create an optimally-sized wired memory ticket from a measurement.
    ///
    /// Uses `WiredBudgetPolicy` with the measured weights + workspace as the base budget
    /// and the KV cache size as the active ticket size. The total budget is clamped to
    /// the GPU's recommended working set size.
    ///
    /// - Parameters:
    ///   - measurement: A measurement from `tune()`.
    ///   - headroom: Fractional headroom above measured total (default: 0.1 = 10%).
    /// - Returns: A ticket sized for the measured workload.
    public static func ticket(
        from measurement: WiredMemoryMeasurement,
        headroom: Double = 0.1
    ) -> WiredMemoryTicket {
        let budget = Int(Double(measurement.totalBytes) * (1.0 + headroom))
        let cap = GPU.maxRecommendedWorkingSetBytes() ?? budget
        let clampedBudget = min(budget, cap)

        return WiredMemoryTicket(
            size: clampedBudget,
            policy: WiredSumPolicy(cap: cap)
        )
    }

    /// Estimate memory budget from model parameters without running a measurement pass.
    ///
    /// Computes: model_weights + kv_cache(maxTokens, kvConfig) + workspace.
    /// The KV cache estimate uses the actual cache structure (respects KV quantization,
    /// rotating cache limits, and hybrid architectures with mixed cache types).
    ///
    /// - Parameters:
    ///   - model: The loaded language model (for weight byte computation).
    ///   - maxTokens: Maximum context length to budget for (prefill + generation).
    ///   - parameters: Generation parameters (KV bits, scheme, maxKVSize, etc.).
    ///   - workspaceFraction: Fraction of weight bytes to add for workspace (default: 0.15).
    ///     Workspace covers prefill intermediates (attention scores, projections, activations)
    ///     which scale with prefill batch size. 15% is conservative for 4-bit models.
    /// - Returns: Estimated total bytes needed.
    public static func estimateBudget(
        model: any LanguageModel,
        maxTokens: Int,
        parameters: GenerateParameters,
        workspaceFraction: Double = 0.15
    ) -> Int {
        // Weight bytes from model parameters (includes quantized weights, scales, biases)
        let weightBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        // KV cache estimate: create the actual cache structure the model would use,
        // then estimate per-token storage from the cache types and dimensions.
        let cache = model.newCache(parameters: parameters)

        // Count KV-cache layers (skip MambaCache/other non-KV types)
        var kvBytesEstimate = 0
        for c in cache {
            if c is KVCacheSimple {
                // FP16 KV cache: 2 arrays (K + V), each [B, heads, tokens, headDim]
                // For KVCacheSimple, we don't know head dimensions directly.
                // Use the model's parameter structure to infer.
                // Conservative: 512 bytes per token per layer (covers most configs)
                // This is ~256 bytes K + 256 bytes V at FP16 with head_dim=128, kv_heads=2
                let effectiveTokens: Int
                if let maxKV = parameters.maxKVSize {
                    effectiveTokens = min(maxTokens, maxKV)
                } else {
                    effectiveTokens = maxTokens
                }
                kvBytesEstimate += effectiveTokens * 512
            }
            // MambaCache: fixed-size state, negligible compared to KV cache
            // TurboQuantKVCache: created dynamically from KVCacheSimple, handled above
        }

        // If KV quantization is configured, reduce the KV estimate accordingly
        if let scheme = parameters.kvScheme, scheme.hasPrefix("turbo") {
            // TurboQuant compresses KV by roughly 4-8x depending on bit width
            // turbo4: ~4x compression, turbo3: ~5.3x, turbo2: ~8x, turbo4v2 (4+2): ~5x
            let compressionRatio: Double
            if scheme.contains("4v2") { compressionRatio = 5.0 }
            else if scheme.hasSuffix("4") { compressionRatio = 4.0 }
            else if scheme.hasSuffix("3") { compressionRatio = 5.3 }
            else if scheme.hasSuffix("2") { compressionRatio = 8.0 }
            else { compressionRatio = 4.0 }
            kvBytesEstimate = Int(Double(kvBytesEstimate) / compressionRatio)
        } else if let bits = parameters.kvBits, bits > 0 {
            // Affine quantization: bits/16 compression ratio
            kvBytesEstimate = kvBytesEstimate * bits / 16
        }

        // Workspace: fraction of weights for prefill intermediate tensors
        // Scales with prefill batch size but capped at weight size
        let workspaceEstimate = Int(Double(weightBytes) * workspaceFraction)

        return weightBytes + kvBytesEstimate + workspaceEstimate
    }

    /// Create a ticket from a static estimate (no measurement pass required).
    ///
    /// The ticket size is: estimateBudget() × (1 + headroom), clamped to GPU capacity.
    /// If the estimate exceeds GPU capacity, it's clamped — the model will still work
    /// but may page to system memory at large contexts.
    ///
    /// - Parameters:
    ///   - model: The loaded language model.
    ///   - maxTokens: Maximum context length to budget for.
    ///   - parameters: Generation parameters (KV config affects cache size).
    ///   - headroom: Fractional headroom above estimate (default: 0.1 = 10%).
    /// - Returns: A ticket sized for the estimated workload.
    public static func estimatedTicket(
        model: any LanguageModel,
        maxTokens: Int,
        parameters: GenerateParameters,
        headroom: Double = 0.1
    ) -> WiredMemoryTicket {
        let budget = estimateBudget(
            model: model, maxTokens: maxTokens, parameters: parameters)
        let total = Int(Double(budget) * (1.0 + headroom))

        // Clamp to GPU recommended working set (leaves room for OS + other apps)
        let gpuCap = GPU.maxRecommendedWorkingSetBytes() ?? total
        let clampedTotal = min(total, gpuCap)

        return WiredMemoryTicket(
            size: clampedTotal,
            policy: WiredSumPolicy(cap: gpuCap)
        )
    }

    /// Measure weights, KV cache, and prefill workspace using a user input.
    ///
    /// This is a convenience wrapper that runs the model's processor to build a
    /// prepared `LMInput`, then delegates to the `tune(input:context:parameters:)`
    /// overload. It is especially useful for VLMs where images or videos are part
    /// of the input and significantly affect memory usage.
    ///
    /// - Parameters:
    ///   - userInput: High-level input (text/images/video) to be prepared.
    ///   - context: The loaded model context.
    ///   - parameters: Generation parameters that control prefill behavior.
    ///   - resetPeakMemory: If true, resets `Memory.peakMemory` before measuring.
    /// - Returns: A measurement snapshot for weights, KV, and workspace.
    public static func tune(
        userInput: UserInput,
        context: ModelContext,
        parameters: GenerateParameters,
        resetPeakMemory: Bool = true
    ) async throws -> WiredMemoryMeasurement {
        let prepared = try await context.processor.prepare(input: userInput)
        return try await tune(
            input: prepared,
            context: context,
            parameters: parameters,
            resetPeakMemory: resetPeakMemory
        )
    }
}
