// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Returns `parameters.compressionAlgorithm` with the turbo case suppressed when
/// the model doesn't support turbo (e.g., sinks-using GPT-OSS — issue #85).
internal func effectiveAlgorithm(
    parameters: GenerateParameters, model: any LanguageModel
) -> KVCache.CompressionAlgorithm? {
    if case .turbo = parameters.compressionAlgorithm, !model.supportsTurboQuantization {
        return nil
    }
    return parameters.compressionAlgorithm
}

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
        let cache = model.newCache(parameters: parameters)

        let resolvedWindowSize = parameters.prefillStepSize ?? model.defaultPrefillStepSize
        switch try model.prepare(input, cache: cache, windowSize: resolvedWindowSize) {
        case .tokens(let tokens):
            let result = model(
                tokens[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            eval(result.logits)
        case .logits(let result):
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

    /// Per-token KV bytes for a single layer × kvHead, given a quantization scheme.
    ///
    /// Returned value covers K **and** V combined for one head at one token.
    /// Multiply by `kvHeads × layers × tokens × batchSize` to get a total.
    ///
    /// FP16 baseline: `headDim × 2 (bytes) × 2 (K+V)`.
    ///
    /// - Important: TurboQuant's A decode path keeps K/V at FP16 in the cache
    ///   (compression is applied lazily at attention time, not in the cache
    ///   itself), so passing `kvScheme = "turbo*"` returns the FP16 baseline.
    ///   The B compressed-attention path can shrink storage further but falls
    ///   back to FP16 on kernel error, so FP16 is the safe upper bound.
    public static func kvBytesPerTokenPerHead(
        headDim: Int,
        algorithm: KVCache.CompressionAlgorithm? = nil
    ) -> Int {
        precondition(headDim > 0, "headDim must be > 0")

        switch algorithm ?? .none {
        case .none, .turbo:
            // FP16 baseline. TurboQuant's A decode path keeps K/V at FP16 in
            // the cache; the B compressed-attention path can shrink storage
            // further but falls back to FP16 on kernel error, so FP16 is the
            // safe upper bound.
            return headDim * 2 * 2
        case let .affine(bits, groupSize):
            guard bits > 0, bits < 16 else { return headDim * 2 * 2 }
            let groups = max(1, headDim / groupSize)
            let perSidePacked = (headDim * bits + 7) / 8
            let perSideMeta = groups * 4 * 2
            return (perSidePacked + perSideMeta) * 2
        }
    }

    /// Total KV cache bytes for a request, given dimensions, scheme, and batch.
    ///
    /// `kvHeads` is one entry per layer (matches `KVCacheDimensionProvider.kvHeads`).
    public static func estimateKVBytes(
        tokens: Int,
        kvHeads: [Int],
        headDim: Int,
        batchSize: Int = 1,
        algorithm: KVCache.CompressionAlgorithm? = nil
    ) -> Int {
        guard tokens > 0, headDim > 0, batchSize > 0, !kvHeads.isEmpty else { return 0 }
        let perTokenPerHead = kvBytesPerTokenPerHead(headDim: headDim, algorithm: algorithm)
        let headsAcrossLayers = kvHeads.reduce(0, +)
        return tokens * headsAcrossLayers * perTokenPerHead * batchSize
    }

    /// Parse `MLX_MEMORY_LIMIT` style values. Accepts plain bytes and human-friendly
    /// suffixes (`g`/`gb`, `m`/`mb`, `k`/`kb`), case-insensitive. Returns nil for
    /// missing / blank / unparseable input.
    public static func parseMemoryLimit(_ raw: String?) -> Int? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return nil }

        let multiplier: Double
        let numberPart: String
        if trimmed.hasSuffix("gb") {
            multiplier = 1_073_741_824
            numberPart = String(trimmed.dropLast(2))
        } else if trimmed.hasSuffix("mb") {
            multiplier = 1_048_576
            numberPart = String(trimmed.dropLast(2))
        } else if trimmed.hasSuffix("kb") {
            multiplier = 1024
            numberPart = String(trimmed.dropLast(2))
        } else if trimmed.hasSuffix("g") {
            multiplier = 1_073_741_824
            numberPart = String(trimmed.dropLast())
        } else if trimmed.hasSuffix("m") {
            multiplier = 1_048_576
            numberPart = String(trimmed.dropLast())
        } else if trimmed.hasSuffix("k") {
            multiplier = 1024
            numberPart = String(trimmed.dropLast())
        } else {
            multiplier = 1
            numberPart = trimmed
        }

        guard let value = Double(numberPart), value > 0, value.isFinite else { return nil }
        let bytes = value * multiplier
        guard bytes >= 1, bytes <= Double(Int.max) else { return nil }
        return Int(bytes)
    }

    /// Read `MLX_MEMORY_LIMIT` from the process environment.
    public static func envMemoryLimit() -> Int? {
        parseMemoryLimit(ProcessInfo.processInfo.environment["MLX_MEMORY_LIMIT"])
    }

    /// Smart memory is on by default. Explicit `MLX_SMART_MEMORY=0` disables it.
    public static func envSmartMemoryEnabled() -> Bool {
        ProcessInfo.processInfo.environment["MLX_SMART_MEMORY"] != "0"
    }

    /// Estimate memory budget from model parameters without running a measurement pass.
    ///
    /// Computes: `weights + kv(maxTokens × batchSize, kvConfig) + workspace`.
    ///
    /// When `kvHeadsOverride` and `headDimOverride` are both supplied, the KV term
    /// is computed precisely from the model's actual dimensions. Otherwise a per-layer
    /// heuristic of `headsHint × 256 × 2` bytes/token (≈ FP16 with kvHeads=8, headDim=128)
    /// is used. The bench harness always supplies overrides; library callers without
    /// architecture knowledge fall back to the heuristic, which is intentionally
    /// generous to avoid undersizing the wired ticket.
    ///
    /// - Parameters:
    ///   - model: The loaded language model (for weight byte computation).
    ///   - maxTokens: Maximum context length per sequence (prefill + generation).
    ///   - parameters: Generation parameters (KV bits, scheme, maxKVSize, etc.).
    ///   - batchSize: Number of concurrent sequences sharing the same model weights
    ///     (each sequence has its own KV cache; weights and workspace amortize).
    ///   - kvHeadsOverride: Per-layer KV head counts (typically from
    ///     `KVCacheDimensionProvider.kvHeads`). When `nil`, the cache count is
    ///     used as a layer count and a default head count is assumed.
    ///   - headDimOverride: Per-head KV dimension. When `nil`, a default of 128 is
    ///     used (typical for 7B-30B class models).
    ///   - workspaceFraction: Fraction of weight bytes to add for workspace (default: 0.15).
    /// - Returns: Estimated total bytes needed.
    public static func estimateBudget(
        model: any LanguageModel,
        maxTokens: Int,
        parameters: GenerateParameters,
        batchSize: Int = 1,
        kvHeadsOverride: [Int]? = nil,
        headDimOverride: Int? = nil,
        workspaceFraction: Double = 0.15
    ) -> Int {
        let weightBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        let cache = model.newCache(parameters: parameters)
        let layerCount = cache.filter { $0 is StandardKVCache }.count

        let effectiveTokens: Int
        if let maxKV = parameters.maxKVSize {
            effectiveTokens = min(maxTokens, maxKV)
        } else {
            effectiveTokens = maxTokens
        }

        let resolvedAlgorithm = effectiveAlgorithm(parameters: parameters, model: model)

        let kvHeadsForBudget = kvHeadsOverride ?? Array(repeating: 8, count: max(layerCount, 1))
        let headDimForBudget = headDimOverride ?? 128

        let kvBytesEstimate = estimateKVBytes(
            tokens: effectiveTokens,
            kvHeads: kvHeadsForBudget,
            headDim: headDimForBudget,
            batchSize: max(1, batchSize),
            algorithm: resolvedAlgorithm
        )

        let workspaceEstimate = Int(Double(weightBytes) * workspaceFraction)

        return weightBytes + kvBytesEstimate + workspaceEstimate
    }

    /// Create a ticket from a static estimate (no measurement pass required).
    ///
    /// The ticket size is: `estimateBudget() × (1 + headroom)`, clamped to GPU capacity.
    public static func estimatedTicket(
        model: any LanguageModel,
        maxTokens: Int,
        parameters: GenerateParameters,
        batchSize: Int = 1,
        kvHeadsOverride: [Int]? = nil,
        headDimOverride: Int? = nil,
        headroom: Double = 0.1
    ) -> WiredMemoryTicket {
        let budget = estimateBudget(
            model: model,
            maxTokens: maxTokens,
            parameters: parameters,
            batchSize: batchSize,
            kvHeadsOverride: kvHeadsOverride,
            headDimOverride: headDimOverride
        )
        let total = Int(Double(budget) * (1.0 + headroom))

        let gpuCap = GPU.maxRecommendedWorkingSetBytes() ?? total
        let clampedTotal = min(total, gpuCap)

        return WiredMemoryTicket(
            size: clampedTotal,
            policy: WiredSumPolicy(cap: gpuCap)
        )
    }

    /// Convenience that resolves the wired ticket using environment overrides.
    ///
    /// Precedence:
    /// 1. `MLX_MEMORY_LIMIT` (bytes / `Ng` / `NM` / etc.) — used verbatim, clamped to GPU cap.
    /// 2. `MLX_SMART_MEMORY != "0"` — model-aware estimate via `estimatedTicket(...)`.
    /// 3. Fallback — `GPU.maxRecommendedWorkingSetBytes()`, no estimation.
    public static func resolveTicket(
        model: any LanguageModel,
        maxTokens: Int,
        parameters: GenerateParameters,
        batchSize: Int = 1,
        kvHeadsOverride: [Int]? = nil,
        headDimOverride: Int? = nil,
        headroom: Double = 0.1
    ) -> WiredMemoryTicket {
        let gpuCap = GPU.maxRecommendedWorkingSetBytes()

        if let explicit = envMemoryLimit() {
            let clamped = gpuCap.map { min(explicit, $0) } ?? explicit
            return WiredMemoryTicket(
                size: clamped,
                policy: WiredSumPolicy(cap: gpuCap)
            )
        }

        if envSmartMemoryEnabled() {
            return estimatedTicket(
                model: model,
                maxTokens: maxTokens,
                parameters: parameters,
                batchSize: batchSize,
                kvHeadsOverride: kvHeadsOverride,
                headDimOverride: headDimOverride,
                headroom: headroom
            )
        }

        let fallback = gpuCap ?? estimateBudget(
            model: model,
            maxTokens: maxTokens,
            parameters: parameters,
            batchSize: batchSize,
            kvHeadsOverride: kvHeadsOverride,
            headDimOverride: headDimOverride
        )
        return WiredMemoryTicket(
            size: fallback,
            policy: WiredSumPolicy(cap: gpuCap)
        )
    }
}
