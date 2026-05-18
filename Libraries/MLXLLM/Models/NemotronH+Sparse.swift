// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode + prefill hooks for the NemotronH hybrid family. NemotronH
// interleaves four block kinds via the `hybridOverridePattern` string:
//   - 'M' → Mamba2 mixer (selective-SSM; per-slot conv + recurrent state)
//   - '*' → attention (RoPE-free, no Q/K norm — Llama-style stripped)
//   - '-' → MLP (no cache)
//   - 'E' → MoE (no cache)
//
// Sparse hookup mirrors `Qwen35+Sparse.swift`:
//   - Attention layers run through `BatchedRetrievalAttentionKVCache`
//     (`.sparseAttention` slot). NemotronH's attention is the simplest
//     possible — no RoPE, no Q/K norm — so the sparse path is a strict
//     subset of Llama+Sparse.
//   - Mamba2 layers run through a batched `ssmUpdate` against a
//     `BatchedMambaCache`. `BatchedMambaCache.recDtype` is set to the
//     model's compute dtype so NemotronH's input-dtype SSM state contract
//     (the kernel returns state in input dtype, not fp32 as GDN does) is
//     preserved through writeback.
//   - MLP / MoE blocks pass through unchanged — sparse only touches the
//     attention SDPA step; expert routing is orthogonal.
//
// Per-layer cache list mirrors the dense `newCache` shape: ONE entry per
// mamba OR attention block (mlp / moe blocks contribute no cache, so they
// don't advance the cache index). `fullyBatchedSparseDecode` walks
// `backbone.layers` and uses a running `cacheIdx` that only advances on
// mamba/attention.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension NemotronHAttention {

    /// Llama-style batched sparse forward — no RoPE, no Q/K norm.
    /// Mirrors `LlamaAttention.fullyBatchedSparseForward` minus the
    /// RoPE rotations. Handles both decode steps (L=1) and prefill
    /// chunks (L>1). At L>1 dispatches to `prefillSparseAttend` when
    /// sparse-prefill is enabled.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        let queries = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        // No RoPE on NemotronH attention; cache update + index update sit
        // directly on the projected K / V.
        if L == 1 {
            cache.update(newKeys: keys, newValues: values)
        } else {
            cache.updateChunk(newKeys: keys, newValues: values)
        }
        raCache.updateIndex(newKeys: keys)

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

extension NemotronHMamba2Mixer {

    /// Batched single-step Mamba2 forward against `BatchedMambaCache`. Reads
    /// the live conv + recurrent state slices for the active prefix, runs
    /// the (already-B-aware) `ssmUpdate` kernel, and writes back.
    /// `BatchedMambaCache.recDtype` should match the model's compute dtype
    /// so the writeback round-trip is dtype-stable.
    public func fullyBatchedForward(
        _ inputs: MLXArray, cache: BatchedMambaCache
    ) -> MLXArray {
        let B = inputs.dim(0)
        precondition(B == cache.active,
            "NemotronH Mamba2 fullyBatchedForward: input B (\(B)) ≠ cache.active (\(cache.active))")

        let projected = inProj(inputs)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        // Slice the live conv + recurrent state. Views over the cache —
        // safe to mutate only via `cache.writeback(...)`.
        let (convStateSlice, recStateSlice) = cache.slice(active: B)

        // Pre-pend the rolling conv window, then convolve.
        let padded = concatenated([convStateSlice, convInput], axis: 1)
        let end = padded.dim(1)
        let start = max(0, end - (convKernelSize - 1))
        let newConvState = padded[0..., start ..< end, 0...].contiguous()
        let convOutput = silu(conv1d(padded))

        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1)
        var hidden = convSplits[0]
        var bMat = convSplits[1]
        var cMat = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        bMat = bMat.reshaped([bMat.dim(0), bMat.dim(1), numGroups, ssmStateSize])
        cMat = cMat.reshaped([cMat.dim(0), cMat.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        // Cast the cached recurrent state to the model's compute dtype so
        // `ssmUpdate`'s kernel-template `T` (which it reads from the input
        // dtype) matches. `BatchedMambaCache.writeback` casts back to
        // `recDtype` on commit.
        let inputDtype = inputs.dtype
        let prevState: MLXArray? =
            recStateSlice.dtype == inputDtype
                ? recStateSlice : recStateSlice.asType(inputDtype)

        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: bMat,
            C: cMat,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: prevState,
            timeStepLimit: timeStepLimit,
            mask: nil  // single-step decode (S == 1) — no SSM mask needed
        )

        cache.writeback(conv: newConvState, rec: nextState)

        let flattenedY = y.flattened(start: 2)
        return outProj(norm(flattenedY, gate: gate))
    }
}

extension NemotronHAttention {

    /// Dense batched forward against a plain `BatchedKVCache`. Used by the
    /// `BatchedHybridLLM` parent surface (sparse extension's parent
    /// protocol requires a dense path too).
    public func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let queries = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        cache.update(newKeys: keys, newValues: values)
        let (k, v, mask) = cache.getCachedWithMask()
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v,
            scale: scale, mask: .array(mask))
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

extension NemotronHBlock {

    /// Per-block-type batched dispatch — handles both sparse (when the
    /// slot is `.sparseAttention`) and dense (`.attention`). `cacheSlot`
    /// is `nil` for mlp/moe blocks (no cache).
    func fullyBatchedSparseForward(
        _ x: MLXArray,
        cacheSlot: BatchedHybridCache.BatchedLayerCache?
    ) -> MLXArray {
        let hidden = norm(x)
        let output: MLXArray

        switch (blockType, cacheSlot) {
        case (.attention, .sparseAttention(let raCache)?):
            guard let attn = mixer as? NemotronHAttention else {
                fatalError("NemotronH attention block: mixer is not NemotronHAttention")
            }
            output = attn.fullyBatchedSparseForward(hidden, raCache: raCache)

        case (.attention, .attention(let kv)?):
            // Dense-band path — used by `fullyBatchedDecode` (the parent
            // `BatchedHybridLLM` surface).
            guard let attn = mixer as? NemotronHAttention else {
                fatalError("NemotronH attention block: mixer is not NemotronHAttention")
            }
            output = attn.fullyBatchedForward(hidden, cache: kv)

        case (.mamba, .gdn(let mambaCache)?):
            guard let mamba = mixer as? NemotronHMamba2Mixer else {
                fatalError("NemotronH mamba block: mixer is not NemotronHMamba2Mixer")
            }
            output = mamba.fullyBatchedForward(hidden, cache: mambaCache)

        case (.mlp, nil), (.moe, nil):
            // MLP / MoE: no cache, just pass through the unary mixer.
            guard let m = mixer as? UnaryLayer else {
                fatalError("NemotronH mlp/moe block: mixer is not UnaryLayer")
            }
            output = m(hidden)

        default:
            fatalError(
                "NemotronH+Sparse: block/cache mismatch (blockType=\(blockType), cacheSlot present? \(cacheSlot != nil))")
        }

        return x + output
    }
}

extension NemotronHBackbone {

    /// Per-layer batched sparse forward. Walks every block; advances the
    /// hybrid cache index only for mamba/attention blocks (mlp/moe carry
    /// no cache slot, mirroring `newCache`).
    func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        caches: BatchedHybridCache
    ) -> MLXArray {
        var hidden = embeddings(inputs)
        var cacheIdx = 0

        for layer in layers {
            let slot: BatchedHybridCache.BatchedLayerCache?
            switch layer.blockType {
            case .mamba, .attention:
                precondition(cacheIdx < caches.layers.count,
                    "NemotronH+Sparse: hybrid cache too small for layer pattern")
                slot = caches.layers[cacheIdx]
                cacheIdx += 1
            case .mlp, .moe:
                slot = nil
            }
            hidden = layer.fullyBatchedSparseForward(hidden, cacheSlot: slot)
        }
        precondition(cacheIdx == caches.layers.count,
            "NemotronH+Sparse: cache had \(caches.layers.count) slots, but layer pattern consumed only \(cacheIdx)")
        return normF(hidden)
    }
}

extension NemotronHModel: BatchedHybridLLM {

    /// Dense batched decode — required by `BatchedHybridSparseLLM`'s
    /// parent protocol. Reuses `fullyBatchedSparseForward` since per-block
    /// dispatch already handles both `.attention` and `.sparseAttention`.
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        var out = backbone.fullyBatchedSparseForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        return out
    }

    /// Build a dense `BatchedHybridCache` — `.attention` + `.gdn` slots,
    /// one per mamba/attention block. mlp/moe blocks contribute no slot.
    public func newBatchedHybridCache(
        maxBatch: Int, parameters: GenerateParameters?
    ) -> BatchedHybridCache {
        let cfg = configuration
        let pattern = Array(cfg.hybridOverridePattern)

        let mambaIntermediate = cfg.mambaNumHeads * cfg.mambaHeadDim
        let mambaConvDim =
            mambaIntermediate + 2 * cfg.nGroups * cfg.ssmStateSize
        let mambaKernelMinusOne = cfg.convKernel - 1
        let attnHeadDim = cfg.headDim ?? (cfg.hiddenSize / cfg.numAttentionHeads)
        let maxSeq = parameters?.maxKVSize ?? 2048
        let modelDtype: DType = backbone.embeddings.weight.dtype

        var slots: [BatchedHybridCache.BatchedLayerCache] = []
        for ch in pattern {
            switch NemotronHBlockType(from: ch) {
            case .mamba:
                slots.append(.gdn(BatchedMambaCache(
                    maxBatch: maxBatch,
                    kernelMinusOne: mambaKernelMinusOne,
                    convDim: mambaConvDim,
                    Hv: cfg.mambaNumHeads,
                    Dv: cfg.mambaHeadDim,
                    Dk: cfg.ssmStateSize,
                    dtype: modelDtype,
                    recDtype: modelDtype)))
            case .attention:
                slots.append(.attention(BatchedKVCache(
                    maxBatch: maxBatch,
                    kvHeads: cfg.numKeyValueHeads,
                    headDim: attnHeadDim,
                    maxSeq: maxSeq,
                    dtype: modelDtype)))
            case .mlp, .moe:
                continue
            }
        }
        return BatchedHybridCache(layers: slots)
    }
}

extension NemotronHModel: BatchedHybridSparseLLM {

    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        var out = backbone.fullyBatchedSparseForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        return out
    }

    /// Build a `BatchedHybridCache` for NemotronH:
    ///   - Mamba blocks → `.gdn(BatchedMambaCache)`.
    ///   - Attention blocks → `.sparseAttention(BatchedRetrievalAttentionKVCache)`.
    ///   - mlp / moe blocks contribute NO slot — matches `newCache`'s shape.
    ///
    /// `recDtype` of each `BatchedMambaCache` is set to the model's compute
    /// dtype (bf16 default) so the input-dtype SSM state contract holds.
    public func newBatchedHybridSparseCache(
        maxBatch: Int,
        parameters: GenerateParameters?,
        raConfig: RetrievalAttentionConfig
    ) -> BatchedHybridCache {
        let cfg = configuration
        let pattern = Array(cfg.hybridOverridePattern)

        // NemotronH Mamba2 cache shape — derived from the dense
        // `NemotronHMamba2Mixer.init` plumbing.
        let mambaIntermediate = cfg.mambaNumHeads * cfg.mambaHeadDim
        let mambaConvDim =
            mambaIntermediate + 2 * cfg.nGroups * cfg.ssmStateSize
        let mambaKernelMinusOne = cfg.convKernel - 1

        // Attention shape.
        let attnHeadDim = cfg.headDim ?? (cfg.hiddenSize / cfg.numAttentionHeads)
        let maxSeq = parameters?.maxKVSize ?? 2048

        // Count attention blocks up front so each
        // `BatchedRetrievalAttentionKVCache` knows its
        // total-attention-layer-count for dense-band predicates.
        let attentionTotal = pattern.reduce(0) { acc, ch in
            acc + (NemotronHBlockType(from: ch) == .attention ? 1 : 0)
        }
        var attentionIdx = 0

        // Detect compute dtype from a model parameter sample. NemotronH's
        // `castPredicate` keeps `e_score_correction_bias` + `A_log` in fp32,
        // so we deliberately read a different parameter (embeddings) for
        // the model-wide compute dtype.
        let modelDtype: DType = backbone.embeddings.weight.dtype

        var slots: [BatchedHybridCache.BatchedLayerCache] = []
        slots.reserveCapacity(attentionTotal + pattern.filter { $0 == "M" }.count)
        for ch in pattern {
            switch NemotronHBlockType(from: ch) {
            case .mamba:
                slots.append(.gdn(BatchedMambaCache(
                    maxBatch: maxBatch,
                    kernelMinusOne: mambaKernelMinusOne,
                    convDim: mambaConvDim,
                    Hv: cfg.mambaNumHeads,
                    Dv: cfg.mambaHeadDim,
                    Dk: cfg.ssmStateSize,
                    dtype: modelDtype,
                    recDtype: modelDtype)))
            case .attention:
                let inner = BatchedKVCache(
                    maxBatch: maxBatch,
                    kvHeads: cfg.numKeyValueHeads,
                    headDim: attnHeadDim,
                    maxSeq: maxSeq,
                    dtype: modelDtype)
                let raCache = BatchedRetrievalAttentionKVCache(
                    inner: inner,
                    B: maxBatch,
                    nKVHeads: cfg.numKeyValueHeads,
                    dHead: attnHeadDim,
                    layerIdx: attentionIdx,
                    totalLayers: attentionTotal,
                    raConfig: raConfig)
                attentionIdx += 1
                slots.append(.sparseAttention(raCache))
            case .mlp, .moe:
                // No cache slot — mirror `newCache`.
                continue
            }
        }
        return BatchedHybridCache(layers: slots)
    }
}
