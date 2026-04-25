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
        // β opt-in: compressed-domain Metal kernels. Sinks not supported.
        if turboCache.useCompressedAttention {
            if sinks != nil {
                fatalError(
                    "TurboQuant compressed attention (β, useCompressedAttention=true) "
                    + "does not support attention sinks. Use the default α path "
                    + "(useCompressedAttention=false)."
                )
            }
            return turboCache.compressedAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask
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
