import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update. Routes by `cache.storageKind`:
///
/// - `.turboCompressed` — always `compressedAttention`. The cache itself
///   chooses its decode-time attention path: TurboFlash (default, true
///   compressed-domain) or dequant-then-SDPA (opt-in via
///   `TURBO_DEQUANT_SDPA=1`, materialises a per-layer FP16 K/V buffer).
///   The prefill (`L>1`) case still uses raw update + standard SDPA —
///   the cache compresses on first decode call.
/// - `.affineQuantized` — `quantizedScaledDotProductAttention` (sinks
///   folded in-Swift).
/// - `.raw` / `.ssm` / `.composite` — `MLXFast.scaledDotProductAttention(...
///   sinks:)`.
///
/// `sinks` defaults to `nil`; non-sinks-using models can omit it.
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    guard let cache else {
        // Cache-less path (rare). Wrap the SDPA call so it shows up in traces
        // alongside cache-backed paths for comparability.
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        }
    }
    // Dispatch on the typed `storageKind` enum (spec 006) rather than `as?`
    // downcasts. The downcast inside each arm is a class-identity assertion
    // — a mismatch indicates a programming error, not a runtime case.
    switch cache.storageKind {
    case .turboCompressed:
        guard let turboCache = cache as? TurboQuantizedKVCache else {
            fatalError(
                "storageKind .turboCompressed but cache is not TurboQuantizedKVCache: \(type(of: cache))"
            )
        }
        let L = queries.dim(2)
        if L > 1 {
            // Prefill (L>1): raw update + standard SDPA. Cache compresses
            // lazily on the first L=1 decode call via `compressRawCache`.
            let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            BenchmarkSignpost.end(updH)
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: mask, sinks: sinks
                )
            }
        }
        // Decode (L=1): compressed-domain attention. `compressedAttention`
        // emits its own tq_* sub-phase signposts and chooses TurboFlash
        // (default) or dequant-SDPA (opt-in) internally.
        let ws = Int(mask.windowSize)
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask,
            sinks: sinks,
            windowSize: ws
        )

    case .affineQuantized:
        guard let quantizedKVCache = cache as? AffineQuantizedKVCache else {
            fatalError(
                "storageKind .affineQuantized but cache is not AffineQuantizedKVCache: \(type(of: cache))"
            )
        }
        // Sinks fold lives in `quantizedScaledDotProductAttention`'s
        // numerically-stable manual softmax.
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.qsdpa) {
            quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: quantizedKeys,
                quantizedValues: quantizedValues,
                scale: scale,
                mask: mask,
                sinks: sinks,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
        }

    case .raw, .ssm, .composite:
        // Standard path. SSM caches (`SSMStateCache`) and composite
        // (`CacheList`) reach this arm via fallback — layer code typically
        // doesn't call `attentionWithCacheUpdate` for SSM layers.
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        }
    }
}
