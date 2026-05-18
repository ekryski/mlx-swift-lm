// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode hooks for Gemma 4. Gemma 4 specifics handled here:
//   - Per-layer sliding / global attention mix via `layerTypes`. Each layer
//     owns its own RoPE (sliding uses local rope theta; global uses the
//     scaled rope theta + maxPositionEmbeddings).
//   - Sliding layers and global layers have DIFFERENT head dimensions and
//     KV-head counts (global uses `globalKvHeads` / `globalHeadDim`). The
//     per-layer `BatchedRetrievalAttentionKVCache` list must match each
//     layer's actual K shape — the caller is responsible.
//   - v_norm: Gemma 4 applies a value RMSNorm with no learned weight; we
//     replicate via `MLXFast.rmsNorm(values, weight: .mlxNone, eps: ...)`.
//   - attentionKEqV (some non-sliding layers re-use K as V — no v_proj):
//     handled by reading `vProj` optionality at runtime.
//   - Embeddings are pre-scaled by sqrt(hiddenSize).
//   - KV-shared layers (e2b: trailing `num_kv_shared_layers` layers reuse
//     the donor layer's K/V) are wired through the donor's
//     `BatchedRetrievalAttentionKVCache` — the caller passes the SAME
//     cache instance for each shared layer as the donor occupies in the
//     `raCaches` list (see `ModelInner.fullyBatchedSparseForward`). The
//     shared layer computes Q only and runs sparse attend against the
//     donor's already-written K/V, RoPE'ing Q at the donor's pre-update
//     offset (mirrors the dense path's `donorOffset`).
//   - MoE FFN blocks (`experts` / `router` non-nil) pass through the
//     existing dense expert-routing branch in `Gemma4TransformerBlock`
//     — sparse only affects attention; expert routing is orthogonal.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Gemma4Attention {

    /// Batched sparse forward for a donor (own-cache) layer.
    /// Selects sliding OR global at init; same call site handles both.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // Use the non-fused norm + RoPE path on the sparse decode site to keep
        // the math straightforward; the fused-kernel optimisation lives on the
        // existing dense path and is orthogonal to the sparse-attention swap.
        var queries = qProj(x).reshaped(B, L, nHeads, -1)
        var keys = kProj(x).reshaped(B, L, nKVHeads, -1)
        var values: MLXArray
        if attentionKEqV {
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, nKVHeads, -1)
        }
        // v_norm has no learned weight — keep parity with the dense path.
        values = MLXFast.rmsNorm(values, weight: MLXArray.mlxNone, eps: rmsNormEps)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

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
            let (k, v, mask) = cache.getCachedWithMask()
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: k, values: v,
                scale: scale, mask: .array(mask))
        }
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    /// KV-shared variant: compute Q only, run sparse attend against the
    /// donor's cache (which already wrote its K/V this step). RoPE for Q
    /// uses the donor's pre-update offset — mirrors the dense
    /// `useSharedKV: true` branch in `Gemma4Attention.callAsFunction`.
    ///
    /// Caller contract: `raCache` is the same instance the donor layer
    /// just wrote to; `donorPreUpdateOffset` is the offset captured BEFORE
    /// the donor's `cache.update` ran (otherwise Q is RoPE'd at the wrong
    /// position).
    public func fullyBatchedSparseSharedForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache,
        donorPreUpdateOffset: Int
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Compute Q only; K/V already live in donor's cache.
        var queries = qProj(x).reshaped(B, L, nHeads, -1)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: donorPreUpdateOffset)

        let output: MLXArray
        if L == 1 && raCache.isSparseEligible {
            output = raCache.sparseAttend(queries: queries, scale: scale)
        } else {
            // L > 1 prefill or dense-band fallback — read donor's full K/V
            // and run plain SDPA. Matches the donor branch's fallback path.
            let cache = raCache.inner
            let (k, v, mask) = cache.getCachedWithMask()
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: k, values: v,
                scale: scale, mask: .array(mask))
        }
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

extension Gemma4TransformerBlock {

    /// Sparse forward over Gemma 4's pre/post norm + rmsNormResidual fusion.
    /// `donorPreUpdateOffset` is `nil` for donor layers (this layer owns
    /// its cache) and `Some(offset)` for KV-shared layers (caller captured
    /// the donor's offset before the donor's attention ran).
    /// `mlp` MoE expert routing is preserved exactly as in the dense path.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache,
        perLayerInput: MLXArray? = nil,
        donorPreUpdateOffset: Int? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let attnOut: MLXArray
        if let offset = donorPreUpdateOffset {
            attnOut = selfAttention.fullyBatchedSparseSharedForward(
                inputNorm, raCache: raCache, donorPreUpdateOffset: offset)
        } else {
            attnOut = selfAttention.fullyBatchedSparseForward(
                inputNorm, raCache: raCache)
        }
        var h = MLXFast.rmsNormResidual(
            attnOut, residual: x,
            weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps)

        // FFN. MoE checkpoints route through the experts + shared-MLP fork;
        // dense checkpoints take the single shared-MLP branch. Expert
        // routing is orthogonal to sparse attention (lives in `mlp`-style
        // submodules, not the attention step) — mirrors Qwen3MoE+Sparse.
        if let experts, let router,
           let postNorm1 = postFeedforwardLayerNorm1,
           let preNorm2 = preFeedforwardLayerNorm2,
           let postNorm2 = postFeedforwardLayerNorm2
        {
            // MoE path — pre/post norms wrap both the shared MLP and the
            // expert MoE branch, then fuse via add.
            let preFFNNorm = preFeedforwardLayerNorm(h)
            var h1 = sharedMLP(preFFNNorm)
            h1 = postNorm1(h1)

            // Route: router gets h (pre-norm), experts get norm2(h).
            let routerLogits = router(h)
            let (topKLogits, topKIndices) = gemma4TopKSparse(
                routerLogits, k: topKExperts, axis: -1)
            let stopIndices = MLX.stopGradient(topKIndices)
            var expertWeights = softmax(topKLogits, axis: -1, precise: true)
            expertWeights = expertWeights * router.perExpertScale[topKIndices]
            let preFFNNorm2 = preNorm2(h)
            var h2 = experts(preFFNNorm2, stopIndices)
            h2 = h2 * expandedDimensions(expertWeights, axis: -1)
            h2 = h2.sum(axis: -2)
            h2 = postNorm2(h2)

            let ffnOut = h1 + h2
            h = MLXFast.rmsNormResidual(
                ffnOut, residual: h,
                weight: postFeedforwardLayerNorm.weight,
                eps: postFeedforwardLayerNorm.eps)
        } else {
            let preFFNNorm = preFeedforwardLayerNorm(h)
            let ffnOut = sharedMLP(preFFNNorm)
            h = MLXFast.rmsNormResidual(
                ffnOut, residual: h,
                weight: postFeedforwardLayerNorm.weight,
                eps: postFeedforwardLayerNorm.eps)
        }

        // Per-Layer Embeddings (PLE) gate + projection.
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let pli = perLayerInput
        {
            let residual = h
            var g = compiledGeluMulSparse(gate(h), pli)
            g = proj(g)
            g = norm(g)
            h = residual + g
        }

        return h * layerScalar
    }
}

/// Local copy of the compiled fused gelu*mul used by Gemma 4's PLE gate.
/// The shared dense-path one in Gemma4.swift is fileprivate, so we keep a
/// dedicated compiled closure here to avoid touching the source file.
private let compiledGeluMulSparse: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { gate, x in
        geluApproximate(gate) * x
    }

/// Local mirror of `gemma4TopK` (fileprivate to Gemma4.swift). Pure
/// argPartition-based top-k — duplicated here to keep this extension
/// additive (no source-file edits).
private func gemma4TopKSparse(
    _ a: MLXArray, k: Int, axis: Int = -1
) -> (values: MLXArray, indices: MLXArray) {
    let partitionedIndices = argPartition(a, kth: -k, axis: axis)
    let topKIndices = partitionedIndices[.ellipsis, (-k)...]
    let topKValues = takeAlong(a, topKIndices, axis: axis)
    return (topKValues, topKIndices)
}

extension Gemma4ModelInner {

    /// Per-layer sparse forward. Caller is responsible for:
    ///   - Building one `BatchedRetrievalAttentionKVCache` per layer whose
    ///     `inner.keys` shape matches the layer's K (sliding vs global K-shape).
    ///   - For KV-shared layers, the entry at index `j` in `raCaches` MUST be
    ///     the SAME `BatchedRetrievalAttentionKVCache` instance as the donor
    ///     layer's entry (`raCaches[previousKVs[j]] === raCaches[j]`). This
    ///     mirrors how the dense path keys all shared layers off the donor's
    ///     KV cache via `previousKVs`.
    public func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        precondition(raCaches.count == layers.count,
            "raCaches count (\(raCaches.count)) must match layers count (\(layers.count))")

        var h = embedTokens(inputs)
        // sqrt(hiddenSize) embedding scale.
        h = h * sqrt(Float(config.hiddenSize))

        // Per-Layer Embeddings (PLE) is computed exactly as the dense path,
        // but only when the model declares it. The synthetic smoke config
        // leaves `hiddenSizePerLayerInput == 0` which skips the entire block.
        var perLayerInputs: MLXArray? = nil
        if hiddenSizePerLayerInput > 0, let embedPL = embedTokensPerLayer {
            var pli = embedPL(inputs) * embedTokensPerLayerScale
            pli = pli.reshaped(
                pli.dim(0), pli.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)
            if let proj = perLayerModelProjection {
                var plProj = proj(h) * perLayerProjectionScale
                plProj = plProj.reshaped(
                    plProj.dim(0), plProj.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)
                if let norm = perLayerProjectionNorm {
                    plProj = norm(plProj)
                }
                pli = (plProj + pli) * perLayerInputScale
            }
            perLayerInputs = pli
        }

        // Per-layer donor-offset capture. For each donor layer we record
        // `raCaches[i].inner.offsets[0]` BEFORE its attention call so any
        // downstream shared layer can RoPE its Q at the same position the
        // donor used. (Mirrors `donorPreUpdateOffsets[i]` in the dense
        // path.) Defaults to 0; only donors write into this map.
        var donorPreUpdateOffsets = [Int](repeating: 0, count: layers.count)

        for (i, layer) in layers.enumerated() {
            let pli = perLayerInputs.map { $0[0..., 0..., i, 0...] }
            let donorIdx = previousKVs[i]
            let isShared = donorIdx != i

            if isShared {
                // Shared layer: reuse the donor's cache + donor's pre-update
                // offset for Q RoPE. The donor must have already run — this
                // is true because layer order is the donor index < shared
                // index by construction of `previousKVs` in the dense path.
                let donorOffset = donorPreUpdateOffsets[donorIdx]
                h = layer.fullyBatchedSparseForward(
                    h, raCache: raCaches[donorIdx],
                    perLayerInput: pli,
                    donorPreUpdateOffset: donorOffset)
            } else {
                // Donor: capture the cache's offset BEFORE running attention,
                // then run as normal. The cache's offset is identical across
                // active slots in the rectangular-T decode case.
                donorPreUpdateOffsets[i] = raCaches[i].inner.offsets[0]
                h = layer.fullyBatchedSparseForward(
                    h, raCache: raCaches[i], perLayerInput: pli)
            }
        }
        return norm(h)
    }
}

extension Gemma4TextModel: BatchedSparseLLM {

    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else {
            out = lmHead!(out)
        }
        // Final logit softcapping — only when configured (matches dense path).
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            out = compiledLogitSoftcapSparse(MLXArray(softcap), out)
        }
        return out
    }
}

/// Local copy of the compiled logit softcap kernel (fileprivate on the dense
/// path; duplicated here to keep this extension additive).
private let compiledLogitSoftcapSparse: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { softcap, x in
        tanh(x / softcap) * softcap
    }
