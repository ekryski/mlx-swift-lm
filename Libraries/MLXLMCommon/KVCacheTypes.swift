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
    case turbo(keyBits: Int, valueBits: Int)

    public var description: String {
        switch self {
        case .none:
            return "none"
        case let .affine(bits, groupSize):
            // Default groupSize=64 emits the short form; otherwise the full form.
            return groupSize == 64 ? "affine\(bits)" : "affine\(bits)g\(groupSize)"
        case let .turbo(keyBits, valueBits):
            // Symmetric (keyBits == valueBits) emits the short form; asymmetric
            // emits the kvBits-vN form. keyBits=0 → "turbo0v\(valueBits)".
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
/// - Parameters:
///   - scheme: Compression algorithm (`.none` / `.affine(...)` / `.turbo(...)`).
///   - eviction: Eviction strategy (`.unbounded` or `.window(size:keep:)`).
/// - Returns: A `KVCache` instance ready to be used by an attention layer.
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded
) -> any KVCache {
    if case .turbo = scheme {
        precondition(
            eviction == .unbounded,
            "TurboQuantizedKVCache does not support windowed eviction. Use `.affine(bits:)` for sliding-window-quantized caches."
        )
    }

    switch scheme {
    case .none:
        return StandardKVCache(eviction: eviction)
    case let .affine(bits, _):
        // AffineQuantizedKVCache doesn't currently honor windowed eviction;
        // this matches legacy `maybeQuantizeKVCache` behavior, where a
        // sliding-window cache is swapped for a non-rotating quantized
        // cache (per the comment on `StandardKVCache.toQuantized`).
        return AffineQuantizedKVCache(
            groupSize: schemeGroupSize(scheme), bits: bits)
    case let .turbo(keyBits, valueBits):
        return TurboQuantizedKVCache(
            keyBits: keyBits, valueBits: valueBits)
    }
}

/// Build the right cache class for a single attention layer, given a model's
/// `newCache(parameters:)` context. Used by ~14 model factories to construct
/// caches up-front (eliminating the runtime `maybeQuantizeKVCache` swap).
///
/// Decision tree:
/// - `.affine(bits:groupSize:)` → `AffineQuantizedKVCache` (window eviction
///   ignored, matching the legacy swap behavior).
/// - `.turbo(...)` → caller's responsibility (turbo construction needs
///   per-model `headDim` for kernel JIT pre-warm + boundary-skip logic;
///   models that support turbo construct `TurboQuantizedKVCache` directly).
/// - `.none` / `nil` → `StandardKVCache(maxSize: maxSize, keep: keep)` if
///   `maxSize` set; else `StandardKVCache()` (unbounded).
public func makeAttentionCache(
    parameters: GenerateParameters?,
    maxSize: Int? = nil,
    keep: Int = 0
) -> KVCache {
    if case let .affine(bits, groupSize) = parameters?.compressionAlgorithm {
        return AffineQuantizedKVCache(groupSize: groupSize, bits: bits)
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
