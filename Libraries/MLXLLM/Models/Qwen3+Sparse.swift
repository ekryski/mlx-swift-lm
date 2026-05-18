// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode hooks for Qwen3. Mirrors Qwen3.fullyBatchedForward
// but routes through BatchedRetrievalAttentionKVCache.sparseAttend at L=1
// on sparse-eligible layers. Preserves Qwen3's pre-RoPE q_norm + k_norm
// ordering (the architectural difference vs Qwen2).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Qwen3Attention {

    /// Batched sparse forward. Same shape contract as `fullyBatchedForward`
    /// but takes a `BatchedRetrievalAttentionKVCache` and dispatches to
    /// sparse SDPA on sparse-eligible decode steps.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache,
        layerIndex: Int
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // Q/K/V projections.
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // q_norm / k_norm on per-head views, THEN transpose.
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // RoPE + cache update.
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

extension Qwen3TransformerBlock {

    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache,
        layerIndex: Int
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        var r = attention.fullyBatchedSparseForward(
            normed, raCache: raCache, layerIndex: layerIndex)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

extension Qwen3ModelInner {

    public func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        precondition(raCaches.count == layers.count,
            "raCaches count must match layers count")
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedSparseForward(h, raCache: raCaches[i], layerIndex: i)
        }
        return norm(h)
    }
}

extension Qwen3Model: BatchedSparseLLM {

    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }
}
