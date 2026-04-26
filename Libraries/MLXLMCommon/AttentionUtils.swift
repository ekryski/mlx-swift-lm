import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update.
///
/// Routes to the right backend based on the cache type:
/// - `TurboQuantKVCache` (default α path): rotates Q, dequant K/V from rotated FP16
///   workspace, runs `MLXFast.scaledDotProductAttention(... sinks:)`, inverse-rotates output
/// - `TurboQuantKVCache` with `useCompressedAttention = true` (β opt-in): runs
///   `compressedAttention` directly on the packed buffer (sinks unsupported)
/// - `QuantizedKVCacheProtocol`: affine quantized SDPA (sinks unsupported)
/// - any other cache: standard `MLXFast.scaledDotProductAttention(... sinks:)`
///
/// `sinks` defaults to `nil`; non-sinks-using models can omit it.
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
///   - sinks: Optional per-head attention-sink logits ([nHeads]) — flows through
///     SDPA. fatalErrors if combined with a cache type that doesn't support sinks
///     (affine quantized, β compressed TurboQuant).
/// - Returns: Attention output [B, nHeads, L, D]
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
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 {
            // Prefill (L>1): raw update + standard SDPA. Zero overhead.
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask, sinks: sinks
            )
        }
        // β opt-in: compressed-domain Metal kernels. The sinks plumbing
        // (spec 010 B-1) is in place across all four repos — kernel, C++,
        // C ABI, Swift wrapper, dispatch — and pass2 folds the sink into
        // the softmax denominator. The end-to-end math is, however, not
        // numerically equivalent to MLXFast SDPA's sinks output on
        // GPT-OSS-20B today (output collapses to ellipses after a few
        // tokens; α with the same sinks tensor produces coherent text).
        // Until that's resolved, route sinks-using models through α — α's
        // rotation is already active in β-configured caches, so this is a
        // pure SDPA-vs-pass2 routing decision, not a behavioral cliff.
        // Set MLX_TURBO_FORCE_BETA_SINKS=1 to opt into the (currently
        // broken) β + sinks path for further debugging.
        let forceBetaSinks = ProcessInfo.processInfo.environment["MLX_TURBO_FORCE_BETA_SINKS"] == "1"
        if turboCache.useCompressedAttention && (sinks == nil || forceBetaSinks) {
            return turboCache.compressedAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask, sinks: sinks
            )
        }
        // α default: dequant-to-FP16 + standard SDPA(... sinks:).
        let (rotKeys, rotValues) = turboCache.updateAndDequant(keys: keys, values: values)
        let rotQueries = turboCache.prepareQueries(queries)
        let rotOutput = MLXFast.scaledDotProductAttention(
            queries: rotQueries, keys: rotKeys, values: rotValues,
            scale: scale, mask: mask, sinks: sinks
        )
        return turboCache.inverseRotateOutput(rotOutput)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        if sinks != nil {
            fatalError("Affine quantized attention does not support non-zero sinks.")
        }
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
}
