// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN

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
    private static func makeTokenIds(
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
        let tokenIds = makeTokenIds(count: count, tokenizer: tokenizer, seedText: seedText)
        return LMInput(tokens: MLXArray(tokenIds))
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

        let resolvedWindowSize = parameters.prefillStepSize ?? model.defaultPrefillStepSize
        switch try model.prepare(input, cache: cache, windowSize: resolvedWindowSize) {
        case .tokens(let tokens):
            let result = model(
                tokens[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            // Gate turbo on model support — sinks-using models (GPT-OSS) opt out (#85).
            let scheme = model.supportsTurboQuantization ? parameters.kvScheme : nil
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                kvScheme: scheme,
                turboBoundarySkip: parameters.turboBoundarySkip
            )
            eval(result.logits)
        case .logits(let result):
            let scheme = model.supportsTurboQuantization ? parameters.kvScheme : nil
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                kvScheme: scheme,
                turboBoundarySkip: parameters.turboBoundarySkip
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
            prefillStepSize: parameters.prefillStepSize ?? context.model.defaultPrefillStepSize
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
            prefillStepSize: parameters.prefillStepSize ?? context.model.defaultPrefillStepSize
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
    /// - Returns: Estimated total bytes needed.
    public static func estimateBudget(
        model: any LanguageModel,
        maxTokens: Int,
        parameters: GenerateParameters,
        workspaceFraction: Double = 0.15
    ) -> Int {
        let weightBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        // KV cache estimate: create actual cache structure, estimate per-token storage
        let cache = model.newCache(parameters: parameters)

        var kvBytesEstimate = 0
        for c in cache {
            if c is KVCacheSimple {
                // Conservative: 512 bytes per token per layer (covers most configs)
                let effectiveTokens: Int
                if let maxKV = parameters.maxKVSize {
                    effectiveTokens = min(maxTokens, maxKV)
                } else {
                    effectiveTokens = maxTokens
                }
                kvBytesEstimate += effectiveTokens * 512
            }
            // MambaCache: fixed-size state, negligible compared to KV cache
        }

        // Adjust for KV compression schemes (gated on model support — sinks-using
        // models like GPT-OSS opt out of TurboQuant; #85). The A default decode
        // path keeps K/V at full FP16 (no compression at decode), so don't
        // apply the scheme's nominal compression ratio for budget estimation —
        // dividing here was undersizing the wired-memory ticket by ~5× and
        // forcing MLX allocations outside the wired pool at long contexts. B
        // (opt-in `useCompressedAttention=true`) could shrink the budget but
        // falls back to FP16 if any kernel path fails, so FP16 is the safe
        // upper bound either way.
        let effectiveScheme = model.supportsTurboQuantization ? parameters.kvScheme : nil
        if effectiveScheme?.hasPrefix("turbo") == true {
            // No-op: turbo A path is FP16 at decode; keep `kvBytesEstimate` unchanged.
        } else if let bits = parameters.kvBits, bits > 0 {
            kvBytesEstimate = kvBytesEstimate * bits / 16
        }

        let workspaceEstimate = Int(Double(weightBytes) * workspaceFraction)

        return weightBytes + kvBytesEstimate + workspaceEstimate
    }

    /// Create a ticket from a static estimate (no measurement pass required).
    ///
    /// The ticket size is: estimateBudget() * (1 + headroom), clamped to GPU capacity.
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

        let gpuCap = GPU.maxRecommendedWorkingSetBytes() ?? total
        let clampedTotal = min(total, gpuCap)

        return WiredMemoryTicket(
            size: clampedTotal,
            policy: WiredSumPolicy(cap: gpuCap)
        )
    }
}
