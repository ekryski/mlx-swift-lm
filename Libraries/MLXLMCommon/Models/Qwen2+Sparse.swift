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

    /// Batched sparse forward for a single decode step (L=1). The caller
    /// passes a per-layer `BatchedRetrievalAttentionKVCache` whose inner
    /// `BatchedKVCache` already has B slots reserved (via `addRequest()`)
    /// and whose offsets reflect prefill completion.
    ///
    /// Cache update writes post-RoPE K/V into the rectangular buffer; the
    /// selector index gets fed the post-RoPE K (skipped at L=1 — sliding
    /// window covers decode positions). Then either dense `getCachedWithMask`
    /// + MLXFast.SDPA or `raCache.sparseAttend` depending on layer eligibility.
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
        // common steady-state decode case).
        let allSameOffset = cache.offsets[0 ..< cache.active]
            .allSatisfy { $0 == cache.offsets[0] }
        let preUpdateK: MLXArray
        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            preUpdateK = keys
            cache.update(newKeys: keys, newValues: values)
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
            cache.update(newKeys: keys, newValues: values)
        }

        // Feed selector index (skipped at L=1; sliding window covers it).
        raCache.updateIndex(newKeys: preUpdateK)

        let output: MLXArray
        if L == 1 && raCache.isSparseEligible {
            output = raCache.sparseAttend(queries: queries, scale: scale)
        } else {
            // Dense path — prefill chunk or dense-band layer.
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
