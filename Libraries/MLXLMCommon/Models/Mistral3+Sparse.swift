// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode + prefill hooks for the shared Mistral 3 /
// Ministral 3 layer stack. Llama-style attention with two specifics:
//   1. Llama-4 attention scaling: caller passes a pre-computed `attnScale`
//      tensor (computed once per forward in ModelInner), multiplied into the
//      post-RoPE queries.
//   2. Per-layer sliding / full attention dispatch via `layer.useSliding`.
//      Both layer kinds route through `BatchedRetrievalAttentionKVCache` —
//      sparse-eligibility is decided at the `RetrievalAttentionConfig` level,
//      not per layer kind.

import Foundation
import MLX
import MLXNN

extension Mistral3.Attention {

    /// Batched sparse forward for a chunk of L queries (L >= 1) — handles
    /// both decode steps and prefill chunks. Takes the pre-computed
    /// Llama-4 `attnScale` tensor — caller (ModelInner) is responsible
    /// for either computing it from the rope params or passing a
    /// constant-1 tensor when the model isn't Llama-4-scaled.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        attnScale: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let allSameOffset = cache.offsets[0 ..< cache.active]
            .allSatisfy { $0 == cache.offsets[0] }
        let preUpdateK: MLXArray
        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            queries = queries * attnScale
            preUpdateK = keys
        } else {
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
            queries = concatenated(rotQ, axis: 0) * attnScale
            keys = concatenated(rotK, axis: 0)
            preUpdateK = keys
        }

        // Cache update — L=1 decode-write; L>1 prefill-chunk write.
        if L == 1 {
            cache.update(newKeys: keys, newValues: values)
        } else {
            cache.updateChunk(newKeys: keys, newValues: values)
        }

        raCache.updateIndex(newKeys: preUpdateK)

        // Sparse-prefill gate (mirrors Qwen2+Sparse).
        let priorLen = cache.offsets[0] - L
        let sparsePrefillOn = raCache.raConfig.sparsePrefillEnabled
            || BatchedRetrievalAttentionKVCache.envSparsePrefillEnabled
        let canSparsePrefill = L > 1
            && raCache.isSparseEligible
            && sparsePrefillOn
            && priorLen > raCache.raConfig.sparsePrefillMinContext

        let output: MLXArray
        if L == 1 && raCache.isSparseEligible {
            output = raCache.sparseAttend(queries: queries, scale: scale)
        } else if canSparsePrefill {
            output = raCache.prefillSparseAttend(queries: queries, scale: scale)
        } else {
            let (k, v, mask) = cache.getCachedWithMask()
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: k, values: v,
                scale: scale, mask: .array(mask))
        }
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

extension Mistral3.TransformerBlock {

    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        attnScale: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let r = attention.fullyBatchedSparseForward(
            inputLayerNorm(x), attnScale: attnScale, raCache: raCache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

extension Mistral3.ModelInner {

    /// Per-layer dispatch. Caller passes a per-layer `BatchedRetrievalAttentionKVCache`
    /// list whose length matches `layers.count`. Sliding vs full attention is
    /// reflected in each layer's `useSliding` — the dense fallback path picks
    /// the right mask from the inner cache.
    public func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        precondition(raCaches.count == layers.count,
            "raCaches count (\(raCaches.count)) must match layers count (\(layers.count))")
        var h = embedTokens(inputs)

        // Compute the per-token attention scale once (constant tensor when
        // Llama-4 scaling isn't configured).
        let offset = raCaches.first?.inner.offsets[0] ?? 0
        let attnScale: MLXArray
        if let ropeParams = args.ropeParameters,
            let beta = ropeParams["llama_4_scaling_beta"]?.asFloat(),
            let originalMaxPos = ropeParams["original_max_position_embeddings"]?.asInt()
        {
            attnScale = Self.scale(
                h: h, offset: offset, length: h.dim(1),
                beta: beta, maxPositionEmbeddings: originalMaxPos)
        } else {
            attnScale = MLXArray.ones([h.dim(1), 1]).asType(h.dtype)
        }

        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedSparseForward(
                h, attnScale: attnScale, raCache: raCaches[i])
        }
        return norm(h)
    }
}
