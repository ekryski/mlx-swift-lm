// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode hooks for the Qwen 3.5 / Qwen 3.6 hybrid family
// (attention + GatedDeltaNet). Per-layer-type dispatch routes attention
// layers through `BatchedRetrievalAttentionKVCache.sparseAttend` at L=1
// while GDN layers stay on the existing batched GDN path.
//
// Conformance lands on `Qwen35TextModel`, which is also the base class
// for the MoE variant (`Qwen35MoEModel`) — both checkpoints share the
// same hybrid layer pattern, so this single conformance serves both.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Qwen35Attention {

    /// Batched sparse forward over `BatchedRetrievalAttentionKVCache`.
    /// Preserves the Qwen 3.5 gated-Q split (q_proj outputs 2× heads —
    /// queries + gate, then sigmoid-multiplied at the output projection).
    /// Q/K RMSNorm is applied pre-RoPE on the per-head views.
    /// `denseMask` is used only on the dense fallback (prefill chunks /
    /// dense-band layers); sparse decode builds its own per-slot mask.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache,
        denseMask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // q_proj outputs 2x heads (queries + gate). Split before the head
        // reshape so the gate stays at hidden granularity.
        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

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
            let qSlices = MLX.split(queries, parts: B, axis: 0)
            let kSlices = MLX.split(keys, parts: B, axis: 0)
            var rotQ = [MLXArray]()
            var rotK = [MLXArray]()
            rotQ.reserveCapacity(B)
            rotK.reserveCapacity(B)
            for i in 0..<B {
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
            let maxOffset = cache.offsets[0..<cache.active].max() ?? 0
            let allK = cache.keys[..<cache.active, 0..., ..<maxOffset, 0...]
            let allV = cache.values[..<cache.active, 0..., ..<maxOffset, 0...]
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: allK, values: allV,
                scale: scale, mask: .array(denseMask)
            )
        }
        return oProj(sigmoidMultiply(output.transposed(0, 2, 1, 3).reshaped(B, L, -1), gate))
    }
}

extension Qwen35DecoderLayer {

    /// Per-layer-type batched sparse forward. Dispatches on `isLinear`:
    ///   - `(false, .sparseAttention)` → sparse attention path.
    ///   - `(false, .attention)`       → existing dense batched path
    ///     (fall-through when the caller mixes plain `.attention` slots in
    ///     for some attention layers, e.g. dense-band first/last layers).
    ///   - `(true,  .gdn)`             → existing batched GDN path.
    func fullyBatchedSparseForward(
        _ x: MLXArray,
        layerCache: BatchedHybridCache.BatchedLayerCache,
        attnMask: MLXArray
    ) -> MLXArray {
        let r: MLXArray
        switch (isLinear, layerCache) {
        case (true, .gdn(let mambaCache)):
            r = linearAttn!.fullyBatchedForward(inputLayerNorm(x), cache: mambaCache)
        case (false, .sparseAttention(let raCache)):
            r = selfAttn!.fullyBatchedSparseForward(
                inputLayerNorm(x), raCache: raCache, denseMask: attnMask)
        case (false, .attention(let kvCache)):
            r = selfAttn!.fullyBatchedForward(
                inputLayerNorm(x), cache: kvCache, mask: attnMask)
        default:
            fatalError("Qwen35DecoderLayer: layer/cache type mismatch (isLinear=\(isLinear))")
        }
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

extension Qwen35TextModelInner {

    /// Fully batched sparse single-step forward. Builds the dense fallback
    /// mask once from the first `.attention` (non-sparse) layer if one
    /// exists; otherwise emits a placeholder zero-size tensor (sparse
    /// layers compute their own mask internally via the selector index).
    func fullyBatchedSparseForward(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        precondition(caches.layers.count == layers.count,
                     "fullyBatchedSparseForward: cache layer count mismatch")
        var h = embedTokens(inputs)

        // Build a fallback dense attention mask. Sparse-attention layers
        // generate their per-slot mask internally, but dense-band layers
        // (including any sparse layer hitting the L>1 prefill path) still
        // need a plain `[B, 1, 1, T+1]` mask. Either `.attention` or
        // `.sparseAttention` can supply the shape — both wrap a
        // `BatchedKVCache`.
        var sampleAttnCache: BatchedKVCache?
        for layer in caches.layers {
            switch layer {
            case .attention(let c): sampleAttnCache = c
            case .sparseAttention(let ra): sampleAttnCache = ra.inner
            case .gdn: continue
            }
            if sampleAttnCache != nil { break }
        }

        let attnMask: MLXArray
        if let c = sampleAttnCache {
            let B = c.active
            let cacheDtype = c.keys.dtype
            let allSame = c.offsets[0..<B].allSatisfy { $0 == c.offsets[0] }
            let maxPostOffset = (c.offsets[0..<B].max() ?? 0) + 1
            if allSame {
                attnMask = MLXArray.zeros(
                    [B, 1, 1, maxPostOffset], dtype: cacheDtype)
            } else {
                let positions = MLXArray(0..<maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(c.offsets[0..<B].map { $0 + 1 }).reshaped(B, 1)
                let valid = positions .< offsetsArr
                attnMask = MLX.where(
                    valid,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            }
        } else {
            attnMask = MLXArray.zeros([0, 1, 1, 0], dtype: h.dtype)
        }

        let modelDtype = h.dtype
        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedSparseForward(
                h, layerCache: caches.layers[i], attnMask: attnMask)
            // Defensive cast: quantized ops can promote bf16 → fp32 inside
            // the lazy graph. asType is zero-cost when dtype already matches.
            h = h.asType(modelDtype)
        }
        return norm(h)
    }
}

extension Qwen35TextModel: BatchedHybridSparseLLM {

    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        var out = model.fullyBatchedSparseForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Build a `BatchedHybridCache` whose attention layers wrap a
    /// `BatchedRetrievalAttentionKVCache` for sparse decode while GDN
    /// layers stay on the standard `BatchedMambaCache` path.
    public func newBatchedHybridSparseCache(
        maxBatch: Int,
        parameters: GenerateParameters?,
        raConfig: RetrievalAttentionConfig
    ) -> BatchedHybridCache {
        let cfg = configuration
        let headDim = cfg.headDim ?? (cfg.hiddenSize / cfg.attentionHeads)
        let kernelMinusOne = cfg.linearConvKernelDim - 1
        let keyDim = cfg.linearKeyHeadDim * cfg.linearNumKeyHeads
        let valueDim = cfg.linearValueHeadDim * cfg.linearNumValueHeads
        let convDim = keyDim * 2 + valueDim
        let maxSeq = parameters?.maxKVSize ?? 2048

        // Count attention layers up front so we can pass the correct
        // `totalLayers` to each `BatchedRetrievalAttentionKVCache` —
        // RetrievalAttentionConfig's dense-band predicate is total-layer
        // -relative; passing model.layers.count would mis-bias the bands.
        let attentionTotal = model.layers.reduce(0) { acc, l in
            acc + (l.isLinear ? 0 : 1)
        }
        var attentionIdx = 0

        let layerCaches: [BatchedHybridCache.BatchedLayerCache] = model.layers.map { layer in
            if layer.isLinear {
                return .gdn(BatchedMambaCache(
                    maxBatch: maxBatch,
                    kernelMinusOne: kernelMinusOne,
                    convDim: convDim,
                    Hv: cfg.linearNumValueHeads,
                    Dv: cfg.linearValueHeadDim,
                    Dk: cfg.linearKeyHeadDim))
            } else {
                let inner = BatchedKVCache(
                    maxBatch: maxBatch,
                    kvHeads: cfg.kvHeads,
                    headDim: headDim,
                    maxSeq: maxSeq)
                let raCache = BatchedRetrievalAttentionKVCache(
                    inner: inner,
                    B: maxBatch,
                    nKVHeads: cfg.kvHeads,
                    dHead: headDim,
                    layerIdx: attentionIdx,
                    totalLayers: attentionTotal,
                    raConfig: raConfig)
                attentionIdx += 1
                return .sparseAttention(raCache)
            }
        }
        return BatchedHybridCache(layers: layerCaches)
    }
}
