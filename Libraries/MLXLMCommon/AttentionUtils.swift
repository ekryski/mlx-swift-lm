import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update.
///
/// Routes to the right backend based on the cache type:
/// - `TurboQuantizedKVCache` (default A path): raw-FP16 cache + standard
///   `MLXFast.scaledDotProductAttention(... sinks:)`. The TurboQuant rotation
///   is bypassed at decode (SDPA is invariant to a fixed orthogonal Π applied
///   to both Q and K), so `prepareQueries`/`inverseRotateOutput` are no-ops
///   and `updateAndDequant` keeps appending to the raw prefill buffer.
/// - `TurboQuantizedKVCache` with `useCompressedAttention = true` (B opt-in): runs
///   `compressedAttention` directly on the packed buffer (sinks-using models
///   auto-fallback to A — compressed-domain pass2 doesn't yet incorporate the
///   sink logits; tracked in PR #99)
/// - `AffineQuantizedKVCache`: affine quantized SDPA (sinks folded in-Swift via
///   numerically-stable manual softmax — see `quantizedScaledDotProductAttention`)
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
///     SDPA and (after this PR) through the affine and turbo-α quantized paths.
///     The β compressed-TurboQuant path silently routes sinks-using models to
///     α; see `AttentionUtils` arm for `.turboCompressed`.
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
    // Dispatch on `storageKind` rather than `as?` downcasts (spec 006). The
    // typed enum is extensible (new storage kinds don't touch this switch's
    // consumers) and self-documenting. The downcast inside each arm is a
    // class-identity assertion: storageKind is defined to mirror the concrete
    // class, so a mismatch indicates a programming error, not a runtime case.
    switch cache.storageKind {
    case .turboCompressed:
        guard let turboCache = cache as? TurboQuantizedKVCache else {
            fatalError(
                "storageKind .turboCompressed but cache is not TurboQuantizedKVCache: \(type(of: cache))"
            )
        }
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

    case .affineQuantized:
        guard let quantizedKVCache = cache as? AffineQuantizedKVCache else {
            fatalError(
                "storageKind .affineQuantized but cache is not AffineQuantizedKVCache: \(type(of: cache))"
            )
        }
        // Sinks support (GPT-OSS family): folded into the in-Swift softmax of
        // `quantizedScaledDotProductAttention`. Sinks add one implicit logit
        // per Q head to the softmax denominator; the V is implicit zero, so
        // they only affect normalization. See `quantizedScaledDotProductAttention`
        // for the math.
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
        // Standard path — raw FP16/BF16 K/V (StandardKVCache), SSM caches
        // (SSMStateCache — not actually K/V but routed through the same
        // default-update path; layer code typically doesn't call
        // attentionWithCacheUpdate for SSM layers anyway), and composite
        // (CacheList — same reasoning).
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
