import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update.
///
/// Routes to the right backend based on the cache type:
/// - `TurboQuantKVCache` (default A path): raw-FP16 cache + standard
///   `MLXFast.scaledDotProductAttention(... sinks:)`. The TurboQuant rotation
///   is bypassed at decode (SDPA is invariant to a fixed orthogonal Π applied
///   to both Q and K), so `prepareQueries`/`inverseRotateOutput` are no-ops
///   and `updateAndDequant` keeps appending to the raw prefill buffer.
/// - `TurboQuantKVCache` with `useCompressedAttention = true` (B opt-in): runs
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
///     (affine quantized, B compressed TurboQuant).
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
    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 {
            // Prefill (L>1): raw update + standard SDPA. Zero overhead.
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
        // B (default): compressed-domain dequant + matrix-engine SDPA.
        // `compressedAttention` emits its own tq_encode/tq_score/tq_value/tq_rotate
        // sub-phase signposts internally — don't double-wrap here.
        // Sinks-using models (GPT-OSS family) auto-fallback to A — the
        // compressed-attention pass2 kernel doesn't yet incorporate the
        // sink-token logits in its online softmax (tracked in PR #99).
        if turboCache.useCompressedAttention && sinks == nil {
            return turboCache.compressedAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask
            )
        }
        // A path: raw-FP16 cache + standard SDPA(... sinks:). Used when the
        // user opts out via `TURBO_COMPRESSED_ATTENTION=0` /
        // `useCompressedAttention=false`, or when the model uses attention
        // sinks.
        // updateAndDequant returns raw K/V; prepareQueries/inverseRotateOutput
        // are no-ops in A — SDPA is invariant to the codec's orthogonal rotation
        // applied to both Q and K, so we skip the rotation entirely.
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (rotKeys, rotValues) = turboCache.updateAndDequant(keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        let rotQueries = turboCache.prepareQueries(queries)
        let rotOutput = BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: rotQueries, keys: rotKeys, values: rotValues,
                scale: scale, mask: mask, sinks: sinks
            )
        }
        return turboCache.inverseRotateOutput(rotOutput)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        if sinks != nil {
            fatalError("Affine quantized attention does not support non-zero sinks.")
        }
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
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
        }
    } else {
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
