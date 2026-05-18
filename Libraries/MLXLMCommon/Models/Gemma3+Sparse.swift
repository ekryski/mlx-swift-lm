// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode + prefill hooks for the shared Gemma 3 layer stack.
// Gemma 3 specifics:
//   - Q/K RMSNorm applied AFTER the [B, H, L, D] transpose, BEFORE RoPE.
//   - Sliding-window vs global attention alternate per `slidingWindowPattern`
//     (typically 6: every 6th layer is global, the rest are sliding). Each
//     layer has its own RoPE — sliding uses local rope base, global uses
//     the standard rope base + scaling.
//   - Backbone scales embedded inputs by sqrt(hiddenSize) before the stack.
//   - Clip-residual (no plain addition) between attn / mlp residuals.
//   - Pre/post-feedforward layernorms wrap the MLP.

import Foundation
import MLX
import MLXNN

extension Gemma3.Attention {

    /// Batched sparse forward for a chunk of L queries (L >= 1) — handles
    /// both decode steps and prefill chunks. At L>1 dispatches to
    /// `prefillSparseAttend` when sparse-prefill is enabled.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // Gemma3-specific: norm BEFORE RoPE, on [B, H, L, D] tensors.
        queries = queryNorm(queries)
        keys = keyNorm(keys)

        let allSameOffset = cache.offsets[0 ..< cache.active]
            .allSatisfy { $0 == cache.offsets[0] }
        let preUpdateK: MLXArray
        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
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
            queries = concatenated(rotQ, axis: 0)
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
        return outputProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

extension Gemma3.TransformerBlock {

    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let r = selfAttention.fullyBatchedSparseForward(inputLayerNorm(x), raCache: raCache)
        let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
        let r2 = mlp(preFeedforwardLayerNorm(h))
        return Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
    }
}

extension Gemma3.Backbone {

    public func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        precondition(raCaches.count == layers.count,
            "raCaches count (\(raCaches.count)) must match layers count (\(layers.count))")
        var h = embedTokens(inputs)

        // sqrt(hiddenSize) scale (computed in bf16 then cast to runtime dtype).
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
            .asType(h.dtype)
        h = h * scale

        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedSparseForward(h, raCache: raCaches[i])
        }
        return norm(h)
    }
}
