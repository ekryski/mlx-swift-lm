// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode hooks for the shared Qwen2 layer stack. Adds
// `fullyBatchedSparseForward` to Attention, DecoderLayer, and ModelInner
// without touching the existing dense entry points. At decode L=1 on
// sparse-eligible layers (per RetrievalAttentionConfig.isSparseLayer),
// attention routes through BatchedRetrievalAttentionKVCache.sparseAttend.
// Dense path is used for L>1 prefill chunks AND for dense-band layers.

import Foundation
import MLX
import MLXNN

extension Qwen2.Attention {

    /// Batched sparse forward for a chunk of L queries (L >= 1) — handles
    /// both prefill chunks AND decode steps. The caller passes a per-layer
    /// `BatchedRetrievalAttentionKVCache` whose inner `BatchedKVCache` has
    /// B slots reserved and whose offsets reflect prior completion.
    ///
    /// Dispatch:
    ///   - L = 1: decode-step semantics — write K/V via `cache.update`
    ///     (advances offsets by 1), then `raCache.sparseAttend` for
    ///     sparse-eligible layers or dense fall-through.
    ///   - L > 1 + sparse-eligible + `sparsePrefillEnabled` (or env knob)
    ///     + priorLen > `sparsePrefillMinContext`: prefill-sparse via
    ///     `cache.updateChunk` (advances offsets by L), selector index
    ///     update, then `raCache.prefillSparseAttend`.
    ///   - L > 1 otherwise: dense via `cache.updateChunk` + `getCachedWithMask`.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // Q/K/V projections, transpose to [B, heads, L, headDim].
        var queries = wq(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        // RoPE — fast path when all slots share the same offset (the
        // common steady-state case, including rectangular prefill).
        let allSameOffset = cache.offsets[0 ..< cache.active]
            .allSatisfy { $0 == cache.offsets[0] }
        let preUpdateK: MLXArray
        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            preUpdateK = keys
        } else {
            // Ragged slot offsets — slot-by-slot RoPE then re-stack.
            let qSlices = split(queries, parts: B, axis: 0)
            let kSlices = split(keys, parts: B, axis: 0)
            var rotQ = [MLXArray]()
            var rotK = [MLXArray]()
            rotQ.reserveCapacity(B)
            rotK.reserveCapacity(B)
            for i in 0 ..< B {
                let off = cache.offsets[i]
                rotQ.append(rope(qSlices[i], offset: off))
                rotK.append(rope(kSlices[i], offset: off))
            }
            queries = concatenated(rotQ, axis: 0)
            keys = concatenated(rotK, axis: 0)
            preUpdateK = keys
        }

        // Cache update — L=1 uses the decode-step write; L>1 uses the
        // prefill-chunk write (advances offsets by L, vectorized across B).
        if L == 1 {
            cache.update(newKeys: keys, newValues: values)
        } else {
            cache.updateChunk(newKeys: keys, newValues: values)
        }

        // Feed selector index. L=1 decode tokens are covered by the
        // sliding window — skipped. L>1 prefill chunks populate the
        // block-feature buffer in bulk.
        raCache.updateIndex(newKeys: preUpdateK)

        // Sparse-prefill gate. Engages when:
        //   - L > 1 (prefill chunk)
        //   - layer is in the sparse band
        //   - sparsePrefillEnabled OR env override
        //   - priorLen exceeds sparsePrefillMinContext threshold
        let priorLen = cache.offsets[0] - L
        let sparsePrefillOn = raCache.raConfig.sparsePrefillEnabled
            || BatchedRetrievalAttentionKVCache.envSparsePrefillEnabled
        let canSparsePrefill = L > 1
            && raCache.isSparseEligible
            && sparsePrefillOn
            && priorLen > raCache.raConfig.sparsePrefillMinContext

        let output: MLXArray
        if L == 1 && raCache.isSparseEligible {
            // Decode-step sparse.
            output = raCache.sparseAttend(queries: queries, scale: scale)
        } else if canSparsePrefill {
            // Prefill-chunk sparse.
            output = raCache.prefillSparseAttend(queries: queries, scale: scale)
        } else {
            // Dense path — short prefill, dense-band layer, or sparse-prefill off.
            let (k, v, mask) = cache.getCachedWithMask()
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: k, values: v,
                scale: scale, mask: .array(mask))
        }
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

extension Qwen2.DecoderLayer {

    /// Layer-level wrapper: pre-norm → sparse attention → residual → MLP.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let r = attention.fullyBatchedSparseForward(
            inputLayerNorm(x), raCache: raCache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

extension Qwen2.ModelInner {

    /// Per-layer dispatch: walk the stack, route each layer through its
    /// own `BatchedRetrievalAttentionKVCache`. The cache list must have
    /// length equal to the number of decoder layers.
    public func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        precondition(raCaches.count == layers.count,
            "raCaches count (\(raCaches.count)) must match layers count (\(layers.count))")
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedSparseForward(h, raCache: raCaches[i])
        }
        return norm(h)
    }
}
