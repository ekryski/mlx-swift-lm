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
public func makeAttentionCache(
    parameters: GenerateParameters?,
    maxSize: Int? = nil,
    keep: Int = 0,
    affineStep: Int? = nil,
    forceRawKV: Bool = false,
    architecturalSlidingWindow: Bool = false,
    useBias: Bool = false
) -> KVCache {
    if case let .affine(bits, groupSize) = parameters?.compressionAlgorithm {
        // Two cases force a `StandardKVCache` fallback under affine:
        //
        // 1. **Architectural sliding window (`architecturalSlidingWindow ==
        //    true`).** Gemma 4 / Mistral3 / etc. define sliding-window layers
        //    in the model architecture itself — SDPA *must* attend over only
        //    the last `slidingWindow` tokens. `AffineQuantizedKVCache` has no
        //    rotating-buffer logic (grows unbounded), so without this
        //    fallback the model attends over the full prefill once context
        //    exceeds the window and produces gibberish.
        //
        // 2. **KV-sharing donors when the caller can't consume quantized
        //    donor state (`forceRawKV == true`).** Spec 041 Phase 5 added a
        //    reader path in `Gemma4TextAttention.callAsFunction(useSharedKV:...)`
        //    that consumes the donor's `getQuantizedState()` tuple directly
        //    via `quantizedScaledDotProductAttention`. That callsite passes
        //    `forceRawKV: false` so its donors stay compressed under affine.
        //    Other KV-sharing callers (Gemma 3n, Gemma 4 VLM — both still
        //    routing donors through FP16 SDPA in the reader) keep
        //    `forceRawKV: true`, falling back to `StandardKVCache` here. The
        //    flag is the caller's contract: "my reader can / cannot route
        //    quantized donor state."
        //
        // `maxSize` alone is NOT a fallback trigger — the bench harness
        // and library callers set `maxSize == contextSize` on full-attention
        // layers as an operational budgeting hint, not as a "wrap and evict
        // on overflow" instruction. For Qwen 3.5 (no sliding-window
        // architecture) those layers should keep their affine compression
        // even with a `maxSize` set.
        //
        // Trade-off where the sliding-window fallback still engages: those
        // specific layers stay FP16 instead of quantising. On Gemma 4
        // sliding-window layers the cache is bounded at
        // `slidingWindow ≈ 1024` tokens so the absolute memory cost is
        // small. Closing this gap fully needs a rotating-buffer-aware
        // affine cache — tracked in
        // https://github.com/ekryski/mlx-swift-lm/issues/202 and on spec
        // 041 Phase 1 (the kernel-level fix carries window semantics in
        // its mask).
        if forceRawKV {
            // Donor reader can't consume quantised state — fall back to
            // `StandardKVCache` so the reader gets raw FP16 K/V via
            // `lastReturnedKeys`. (Phase 5 callsites pass `false` because
            // their readers route through `quantizedScaledDotProductAttention`
            // on the donor's tuple — donors stay compressed.)
            if let maxSize {
                return StandardKVCache(maxSize: maxSize, keep: keep)
            }
            return StandardKVCache()
        }
        if architecturalSlidingWindow, let slidingMax = maxSize {
            // Spec 041 phase 1.2: rotating-window affine cache. Replaces
            // the previous `StandardKVCache(maxSize: slidingMax)` fallback
            // — affine compression is retained on sliding-window layers
            // (Gemma 4 sliding, GPT-OSS sliding) and the kernel-side
            // sliding mask runs against the rolling window.
            let resolvedStep =
                parameters?.prefillStepSize ?? affineStep ?? 256
            return AffineQuantizedKVCache(
                groupSize: groupSize, bits: bits,
                step: resolvedStep, maxSize: slidingMax)
        }
        if architecturalSlidingWindow {
            // Defensive: architecturalSlidingWindow without a maxSize
            // shouldn't happen (the caller would have passed slidingWindow
            // as the maxSize). Fall back to unbounded affine.
        }
        // Prefer the user-overridden prefill step over the per-model default
        // — when the caller resized prefill chunks, the cache should match.
        // `affineStep` carries the model's `defaultPrefillStepSize` from the
        // `newCache(parameters:)` call site; combined with the parameters
        // override here, we land on `parameters.prefillStepSize ??
        // model.defaultPrefillStepSize ?? 256`.
        let resolvedStep = parameters?.prefillStepSize ?? affineStep ?? 256
        return AffineQuantizedKVCache(
            groupSize: groupSize, bits: bits, step: resolvedStep)
    }
    let env = ProcessInfo.processInfo.environment
    let windowedTurboDisabled = env["MLX_TURBO_WINDOWED"] == "0"
    if case let .turbo(keyBits, valueBits, _, _) = parameters?.compressionAlgorithm,
        let maxSize, !windowedTurboDisabled
    {
        return TurboQuantizedKVCache(
            bits: max(keyBits, valueBits),
            keyBits: keyBits, valueBits: valueBits,
            maxSize: maxSize,
            useBias: useBias)
    }
    if let maxSize {
        return StandardKVCache(maxSize: maxSize, keep: keep)
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
