// Copyright © 2024 Apple Inc.
//
// KV cache typed surface — see specs/006-kvcache-refactor.md.
//
// Two orthogonal axes: storage (raw / affine-quantized / turbo-compressed) and
// eviction (unbounded / windowed). Plus a user-facing string-parsed enum
// (`KVCache.CompressionAlgorithm`) and a runtime dispatch tag (`KVStorageKind`).
//
// The factory `makeKVCache(scheme:eviction:)` is in `KVCache.swift` so it can
// reference the concrete cache classes defined there.

import Foundation
import MLX

// MARK: - Storage axis

/// What the cache stores and (for quantized variants) how it encodes.
public enum KVStorage: Sendable, Equatable {
    /// Raw FP16 / BF16 K and V tensors.
    case raw

    /// Group-quantized K/V via MLX's affine quantization. `startOffset` is the
    /// token count at which the cache transitions from raw to quantized;
    /// keeps prefill fast.
    case affine(bits: Int, groupSize: Int = 64, startOffset: Int = 0)

    /// TurboQuant compression (MSE codec). `keyBits = 0` enables raw-key mode
    /// (only values compressed).
    case turbo(keyBits: Int, valueBits: Int, seed: UInt64 = 42)
}

// MARK: - Eviction axis

/// How the cache discards old tokens.
public enum KVEviction: Sendable, Equatable {
    /// Never evict; cache grows with every token.
    case unbounded

    /// Sliding window of `size` tokens. The first `keep` tokens are preserved
    /// across rotations (the attention-sink pattern; GPT-OSS uses keep = 4).
    case window(size: Int, keep: Int = 0)
}

// MARK: - Runtime dispatch tag

/// Reflects what a cache holds *right now*. Self-transitioning caches
/// (`AffineQuantizedKVCache`, `TurboQuantizedKVCache`) report their current
/// state, which may differ from how they were constructed.
///
/// Used by `AttentionUtils.attentionWithCacheUpdate` to dispatch to the right
/// attention path without `as?` downcasts on concrete cache types.
public enum KVStorageKind: Sendable, Equatable {
    case raw
    case affineQuantized(bits: Int, groupSize: Int)
    case turboCompressed(keyBits: Int, valueBits: Int)
    /// SSM state (e.g., GatedDeltaNet, Mamba). Not K/V-shaped.
    case ssm
    /// Composite of heterogeneous sub-caches (`CacheList`).
    case composite
}

// MARK: - Default storageKind for direct protocol conformers

extension KVCache {
    /// Default storage kind for any cache that doesn't explicitly override.
    /// `BaseKVCache` provides its own `open var storageKind` so subclasses can
    /// `override` it; this extension covers caches that conform to `KVCache`
    /// directly (without going through `BaseKVCache`), e.g. `DeepseekV4Cache`,
    /// `BatchedKVCache`, `PagedKVCache`.
    public var storageKind: KVStorageKind { .raw }
}

// MARK: - User-facing scheme parser

/// User-facing compression-scheme enum, parsed from CLI / `GenerateParameters.kvScheme`.
/// Single source of truth for the `"turbo4v2"` / `"affine4"` / etc. string format.
///
/// Aliased as `KVCache.CompressionAlgorithm` (see typealias below) so call sites can
/// read either form. The top-level definition lives here because Swift protocols
/// can't host nested type definitions in extensions; the typealias preserves the
/// scoped call-site syntax requested in spec 006.
public enum KVCacheCompressionAlgorithm: Sendable, Equatable, CustomStringConvertible {
    /// No compression — raw FP16/BF16 K/V.
    case none
    /// Affine group-quantized via MLX. Maps to `KVStorage.affine(...)`.
    case affine(bits: Int, groupSize: Int = 64)
    /// TurboQuant MSE codec. `keyBits = 0` enables raw-key mode.
    ///
    /// Boundary-layer skip preserves the most PPL-sensitive layers (first
    /// N and last N attention layers) at full FP precision when this
    /// algorithm is in effect — matches llama.cpp TurboQuant mode 7. Models
    /// that construct `TurboQuantizedKVCache` directly (Qwen3.5,
    /// NemotronH) honor these knobs in their `newCache(parameters:)`. The
    /// boundary skip only kicks in when there are at least
    /// `4 * boundaryLayersToSkip` convertible attention layers, so small
    /// models don't skip half their layers.
    ///
    /// - Parameters:
    ///   - keyBits: Per-key quantization bits (0 = raw FP key).
    ///   - valueBits: Per-value quantization bits.
    ///   - skipBoundaryLayerCompression: When `true` (default), skip the
    ///     first and last `boundaryLayersToSkip` attention layers.
    ///   - boundaryLayersToSkip: Number of layers at each end to leave
    ///     uncompressed when `skipBoundaryLayerCompression` is `true`.
    ///     Default `2` matches the v3 `maybeQuantizeKVCache` behavior.
    case turbo(
        keyBits: Int,
        valueBits: Int,
        skipBoundaryLayerCompression: Bool = true,
        boundaryLayersToSkip: Int = 2
    )

    public var description: String {
        switch self {
        case .none:
            return "none"
        case let .affine(bits, groupSize):
            // Default groupSize=64 emits the short form; otherwise the full form.
            return groupSize == 64 ? "affine\(bits)" : "affine\(bits)g\(groupSize)"
        case let .turbo(keyBits, valueBits, _, _):
            // Symmetric (keyBits == valueBits) emits the short form; asymmetric
            // emits the kvBits-vN form. keyBits=0 → "turbo0v\(valueBits)".
            // Boundary-skip knobs are not part of the wire format — they're
            // an API-level config that controls which layers get compressed,
            // not the compression scheme itself.
            return keyBits == valueBits ? "turbo\(keyBits)" : "turbo\(keyBits)v\(valueBits)"
        }
    }

    /// Parse a scheme string like `"none"` / `"turbo4"` / `"turbo4v2"` /
    /// `"turbo0v4"` / `"affine4"` / `"affine4g64"` / `"affine8g32"`.
    ///
    /// Returns `nil` for any string that doesn't match. Empty inputs map to `.none`.
    public init?(_ string: String) {
        let trimmed = string.trimmingCharacters(in: .whitespaces).lowercased()

        if trimmed.isEmpty || trimmed == "none" {
            self = .none
            return
        }

        if trimmed.hasPrefix("turbo") {
            let suffix = trimmed.dropFirst("turbo".count)
            if let vIdx = suffix.firstIndex(of: "v") {
                guard let kb = Int(suffix[suffix.startIndex ..< vIdx]),
                    let vb = Int(suffix[suffix.index(after: vIdx)...])
                else { return nil }
                self = .turbo(keyBits: kb, valueBits: vb)
                return
            } else {
                guard let b = Int(suffix) else { return nil }
                self = .turbo(keyBits: b, valueBits: b)
                return
            }
        }

        if trimmed.hasPrefix("affine") {
            let suffix = trimmed.dropFirst("affine".count)
            if let gIdx = suffix.firstIndex(of: "g") {
                guard let bits = Int(suffix[suffix.startIndex ..< gIdx]),
                    let groupSize = Int(suffix[suffix.index(after: gIdx)...])
                else { return nil }
                self = .affine(bits: bits, groupSize: groupSize)
                return
            } else {
                guard let bits = Int(suffix) else { return nil }
                self = .affine(bits: bits, groupSize: 64)
                return
            }
        }

        return nil
    }
}

// Scoped alias so call sites can read `KVCache.CompressionAlgorithm.turbo(...)` —
// the spec-006-preferred call-site syntax. (Swift protocols can't host nested type
// *definitions*, but they can host typealiases pointing to top-level types.)
extension KVCache {
    public typealias CompressionAlgorithm = KVCacheCompressionAlgorithm
}

// MARK: - Factory

/// Construct a `KVCache` for one attention layer, parameterised by the
/// compression scheme + eviction strategy. The 90% case for model
/// `newCache(parameters:)` factories.
///
/// `scheme` and `eviction` compose orthogonally with one constraint:
/// `.turbo(...)` + `.window(...)` is **not supported** and triggers a
/// precondition trap at construction time. TurboQuant's two-phase prefill→
/// decode design has no clean definition of "evict a token from the
/// compressed store"; for sliding-window quantization use `.affine(bits:)`,
/// which supports windowed eviction natively.
///
/// Compute the set of attention-layer indices that should stay uncompressed
/// when TurboQuant is in effect. Mirrors v3's `maybeQuantizeKVCache`
/// boundary-skip behavior — the first N and last N attention layers are the
/// most PPL-sensitive, so quantizing them costs the most quality. Caller
/// passes the *full ordered list* of attention-layer indices (which it
/// computes from its own layer-type discovery — hybrid models like NemotronH
/// have to thread Mamba/MLP/MoE layers around the attention ones); the
/// helper returns the subset to leave uncompressed.
///
/// The boundary-skip only kicks in when there are at least
/// `4 * boundaryLayersToSkip` attention layers, so small models (Qwen 3.5
/// 0.8B, etc.) don't end up with half their layers skipped. This matches
/// llama.cpp TurboQuant mode 7.
///
/// Returns an empty set when:
/// - `algorithm` is `nil` or not `.turbo`
/// - `skipBoundaryLayerCompression` on the algorithm is `false`
/// - `boundaryLayersToSkip` is `0`
/// - The model has fewer than `4 * boundaryLayersToSkip` attention layers
///
/// - Parameters:
///   - attentionLayerIndices: Ordered indices of the attention layers (in
///     the resulting cache list) that the model is about to construct.
///     Index space is the model's own — could be `model.layers`-based
///     (Qwen 3.5) or pattern-walk-based (NemotronH).
///   - algorithm: The active compression algorithm. Only `.turbo(...)`
///     contributes a non-empty set.
/// - Returns: Set of indices from `attentionLayerIndices` that should NOT
///   be compressed. The caller's factory passes those to
///   `makeAttentionCache(...)` and the rest to `TurboQuantizedKVCache(...)`.
public func turboBoundarySkipSet(
    attentionLayerIndices: [Int],
    algorithm: KVCache.CompressionAlgorithm?
) -> Set<Int> {
    guard case let .turbo(_, _, skip, count) = algorithm,
        skip, count > 0
    else { return [] }
    let n = attentionLayerIndices.count
    let actualSkip = (n >= 4 * count) ? count : 0
    return Set(
        attentionLayerIndices.prefix(actualSkip)
            + attentionLayerIndices.suffix(actualSkip)
    )
}

/// - Parameters:
///   - scheme: Compression algorithm (`.none` / `.affine(...)` / `.turbo(...)`).
///   - eviction: Eviction strategy (`.unbounded` or `.window(size:keep:)`).
/// - Returns: A `KVCache` instance ready to be used by an attention layer.
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded,
    affineStep: Int? = nil
) -> any KVCache {
    switch scheme {
    case .none:
        return StandardKVCache(eviction: eviction)
    case let .affine(bits, _):
        // AffineQuantizedKVCache doesn't currently honor windowed eviction;
        // this matches legacy `maybeQuantizeKVCache` behavior, where a
        // sliding-window cache is swapped for a non-rotating quantized
        // cache (per the comment on `StandardKVCache.toQuantized`).
        //
        // `affineStep` lets callers size the cache's growth increment to
        // their model's prefill chunk (defaults to 256 when omitted). Sized
        // to match `LanguageModel.defaultPrefillStepSize` collapses growth
        // events 1:1 with prefill chunks instead of `chunkSize/256`.
        return AffineQuantizedKVCache(
            groupSize: schemeGroupSize(scheme), bits: bits,
            step: affineStep ?? 256)
    case let .turbo(keyBits, valueBits, _, _):
        // TurboQuantizedKVCache supports windowed eviction natively via its
        // `rotatingMaxSize` + `rotatingIdx` write-position machinery — the
        // raw-prefill → compressed-decode pipeline rotates the compressed
        // store at `maxSize` once it transitions, and the SDPA path honors
        // windowed semantics for the mask. Pass `maxSize` through when
        // `.window` eviction is requested; `.keep` (the attention-sink
        // prefix on `StandardKVCache`) is not currently surfaced on the
        // TurboQuant codec — windowed turbo treats the buffer as a flat
        // rotating window. Boundary-skip is applied by the *model factory*
        // (which sees the full attention-layer list), not by this
        // single-cache helper.
        let maxSize: Int? = {
            if case let .window(size, _) = eviction { return size }
            return nil
        }()
        return TurboQuantizedKVCache(
            keyBits: keyBits, valueBits: valueBits, maxSize: maxSize)
    }
}

/// Build the right cache class for a single attention layer, given a model's
/// `newCache(parameters:)` context. Used by ~14 model factories to construct
/// caches up-front (eliminating the runtime `maybeQuantizeKVCache` swap).
///
/// Decision tree:
/// - `.affine(bits:groupSize:)` → `AffineQuantizedKVCache` (window eviction
///   ignored, matching the legacy swap behavior).
/// - `.turbo(...)` with `maxSize` set → `TurboQuantizedKVCache(maxSize:)`
///   (windowed turbo). Issue #185 fixed in this release: KV-shared
///   Gemma 4 variants (E2B / E4B) now read the donor's K/V correctly
///   from `lastReturnedKeys` / `lastReturnedValues` (regular `update()`
///   sets them, matching `updateAndDequant()`'s existing behaviour).
/// - `.turbo(...)` without `maxSize` → caller's responsibility (turbo
///   construction needs per-model `headDim` for kernel JIT pre-warm +
///   boundary-skip logic; models that support turbo construct
///   `TurboQuantizedKVCache` directly).
/// - `.none` / `nil` → `StandardKVCache(maxSize: maxSize, keep: keep)` if
///   `maxSize` set; else `StandardKVCache()` (unbounded).
///
/// Diagnostic env knob: `MLX_TURBO_WINDOWED=0` forces the legacy
/// fallback (`StandardKVCache`) for the windowed-turbo case. Useful
/// for A/B testing or regressions; not needed in normal operation.
/// Build the per-layer KV cache that matches `parameters.compressionAlgorithm`
/// + the layer's architectural shape.
///
/// **Two distinct caps drive rotating eviction:**
///
/// 1. `slidingWindow: Int?` — **architectural** cap. Set per-layer by the
///    model factory when the layer's attention path **must** attend over
///    only the last N tokens (Gemma 4 local layers, GPT-OSS sliding layers,
///    Mistral 3 sliding layers, etc.). When non-nil, the returned cache
///    rotates at this cap regardless of compression algorithm. This is
///    non-negotiable — the model architecture demands it.
///
/// 2. `parameters?.maxKVSize` — **user budget** cap. Read internally from
///    the parameters tuple. When the layer is **not** architecturally
///    sliding-window, this serves as the user's "bound the cache here for
///    memory budgeting" hint. Honored uniformly across all three schemes:
///    - `.none`: returns `StandardKVCache(maxSize:)` — rotating eviction.
///    - `.turbo`: returns `TurboQuantizedKVCache(maxSize:)` — rotating
///      eviction + compression.
///    - `.affine`: returns `AffineQuantizedKVCache(maxSize:)` — rotating
///      eviction + compression (via spec 041 phase 1.2 rotating-window
///      affine cache, same dispatch as the architectural-sliding case).
///
/// Precedence when both are present: `slidingWindow` wins. The architectural
/// cap is non-negotiable; the user cannot make a sliding-window layer
/// attend over more tokens than the architecture allows.
///
/// `forceRawKV: true` is a separate caller's-contract flag for KV-sharing
/// donor layers whose reader path can only consume raw FP16 K/V (Gemma 3n,
/// Gemma 4 VLM today). Under affine, such donors fall back to
/// `StandardKVCache` instead of the quantized donor tuple.
public func makeAttentionCache(
    parameters: GenerateParameters?,
    slidingWindow: Int? = nil,
    keep: Int = 0,
    affineStep: Int? = nil,
    forceRawKV: Bool = false,
    useBias: Bool = false
) -> KVCache {
    // The architectural cap takes precedence over the user budget cap. From
    // this point, all three schemes (`.none` / `.turbo` / `.affine`) treat
    // `effectiveMaxSize` uniformly — rotating eviction at that cap when set.
    let effectiveMaxSize: Int? = slidingWindow ?? parameters?.maxKVSize

    if case let .affine(bits, groupSize) = parameters?.compressionAlgorithm {
        // Affine path. Three branches:
        //
        // 1. `forceRawKV`: KV-shared donor whose reader path consumes raw
        //    FP16 K/V (not the quantised tuple). Fall back to `StandardKVCache`
        //    so the reader gets raw K/V via `lastReturnedKeys`. (Gemma 4 LLM
        //    Phase 5 reader handles the quantised tuple; Gemma 3n / Gemma 4
        //    VLM readers still need raw — they pass `forceRawKV: true`.)
        //
        // 2. Any cap is set (architectural sliding-window OR user budget):
        //    spec 041 phase 1.2 rotating-window `AffineQuantizedKVCache`.
        //    Affine compression retained alongside rotating eviction at the
        //    cap. Same dispatch for both cap sources — the cache class's
        //    rotation logic doesn't care whether the cap came from the
        //    model architecture or `parameters.maxKVSize`.
        //
        // 3. No cap: unbounded `AffineQuantizedKVCache`.
        if forceRawKV {
            if let cap = effectiveMaxSize {
                return StandardKVCache(maxSize: cap, keep: keep)
            }
            return StandardKVCache()
        }
        // Prefer the user-overridden prefill step over the per-model
        // default. `affineStep` carries the model's `defaultPrefillStepSize`;
        // combined with `parameters.prefillStepSize` override, lands on
        // `parameters.prefillStepSize ?? affineStep ?? 256`.
        let resolvedStep = parameters?.prefillStepSize ?? affineStep ?? 256
        if let cap = effectiveMaxSize {
            return AffineQuantizedKVCache(
                groupSize: groupSize, bits: bits,
                step: resolvedStep, maxSize: cap)
        }
        return AffineQuantizedKVCache(
            groupSize: groupSize, bits: bits, step: resolvedStep)
    }

    // Turbo path. Windowed turbo cache when ANY cap is set (architectural
    // sliding-window OR user budget). The codec has its own rotating-window
    // logic so both shapes route identically.
    let env = ProcessInfo.processInfo.environment
    let windowedTurboDisabled = env["MLX_TURBO_WINDOWED"] == "0"
    if case let .turbo(keyBits, valueBits, _, _) = parameters?.compressionAlgorithm,
        let cap = effectiveMaxSize, !windowedTurboDisabled
    {
        return TurboQuantizedKVCache(
            bits: max(keyBits, valueBits),
            keyBits: keyBits, valueBits: valueBits,
            maxSize: cap,
            useBias: useBias)
    }

    // `.none` path. Any cap (architectural or user) → rotating StandardKVCache.
    if let cap = effectiveMaxSize {
        return StandardKVCache(maxSize: cap, keep: keep)
    }
    return StandardKVCache()
}

/// Internal helper to extract groupSize from an affine compression scheme.
private func schemeGroupSize(_ scheme: KVCache.CompressionAlgorithm) -> Int {
    if case let .affine(_, groupSize) = scheme {
        return groupSize
    }
    return 64
}
