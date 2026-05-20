//
//  Qwen3.swift
//  LLM
//
//  Created by John Mai on 2025/4/28.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py

class Qwen3Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let triCache = cache as? TriAttentionKVCache
        if B == 1, let triCache {
            // V3 calibration consumes pre-RoPE Q shaped [tokens, heads, dim].
            // Qwen3 has queries as [B, heads, tokens, dim] at this point.
            let qForCalibration = queries[0].transposed(1, 0, 2).asType(.float32)
            triCache.engine.accumulateQ(qForCalibration, layerIdx: triCache.layerIdx)
        }

        // TriAttention physically compacts K/V storage. Keep RoPE position
        // tied to the original logical token stream, not the compacted
        // storage length (`cache.offset`).
        let rotaryOffset = triCache?.logicalOffset ?? (cache?.offset ?? 0)
        queries = rope(queries, offset: rotaryOffset)
        keys = rope(keys, offset: rotaryOffset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// Batched attention: B requests with separate KV caches.
    /// Projections batched, per-request RoPE + attention + cache update.
    public func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Batched projections: single matmul for all B
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Split into per-request slices once (avoid repeated indexing)
        let qSlices = split(queries, parts: B, axis: 0)
        let kSlices = split(keys, parts: B, axis: 0)
        let vSlices = split(values, parts: B, axis: 0)

        // Per-request: RoPE (different offsets) + cache update
        // Then batched SDPA if all caches are same length, else per-request
        var allSameLen = true
        var firstLen = -1

        var rotQ = [MLXArray]()
        var allKeys = [MLXArray]()
        var allVals = [MLXArray]()
        rotQ.reserveCapacity(B)
        allKeys.reserveCapacity(B)
        allVals.reserveCapacity(B)

        for i in 0..<B {
            let cache_i = caches[i]
            let offset = cache_i?.offset ?? 0
            let q_rot = rope(qSlices[i], offset: offset)
            let k_rot = rope(kSlices[i], offset: offset)

            let (aK, aV) = cache_i?.update(keys: k_rot, values: vSlices[i])
                ?? (k_rot, vSlices[i])

            rotQ.append(q_rot)
            allKeys.append(aK)
            allVals.append(aV)

            let sLen = aK.dim(2)
            if firstLen < 0 { firstLen = sLen }
            if sLen != firstLen { allSameLen = false }
        }

        let output: MLXArray
        if allSameLen && B > 1 {
            // Fast path: all caches same length → single batched SDPA
            // Batched SDPA: B=\(B) seq=\(firstLen)
            let bQ = concatenated(rotQ, axis: 0)      // [B, heads, 1, dim]
            let bK = concatenated(allKeys, axis: 0)    // [B, kvHeads, seq, dim]
            let bV = concatenated(allVals, axis: 0)    // [B, kvHeads, seq, dim]

            output = MLXFast.scaledDotProductAttention(
                queries: bQ, keys: bK, values: bV,
                scale: scale, mask: .none
            )
        } else {
            // Slow path: different lengths → per-request SDPA
            var outputs = [MLXArray]()
            outputs.reserveCapacity(B)
            for i in 0..<B {
                let attn = MLXFast.scaledDotProductAttention(
                    queries: rotQ[i], keys: allKeys[i], values: allVals[i],
                    scale: scale, mask: .none
                )
                outputs.append(attn)
            }
            output = concatenated(outputs, axis: 0)
        }

        return wo(
            output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        )
    }

    /// Fully batched forward with BatchedKVCache — ZERO per-request loops.
    /// Mask is pre-computed once and shared across all layers.
    public func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
        mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Batched projections
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // RoPE + cache update
        let allSameOffset = cache.offsets[0..<cache.active]
            .allSatisfy { $0 == cache.offsets[0] }

        if allSameOffset {
            // Fast path: single batched RoPE
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            cache.update(newKeys: keys, newValues: values)
        } else {
            // Mixed offsets: per-request RoPE then cache update
            let qSlices = split(queries, parts: B, axis: 0)
            let kSlices = split(keys, parts: B, axis: 0)
            var rotQ = [MLXArray]()
            var rotK = [MLXArray]()
            for i in 0..<B {
                let off = cache.offsets[i]
                rotQ.append(rope(qSlices[i], offset: off))
                rotK.append(rope(kSlices[i], offset: off))
            }
            queries = concatenated(rotQ, axis: 0)
            keys = concatenated(rotK, axis: 0)
            cache.update(newKeys: keys, newValues: values)
        }

        let maxOffset = cache.offsets[0..<cache.active].max() ?? 0
        let allK = cache.keys[..<cache.active, 0..., ..<maxOffset, 0...]
        let allV = cache.values[..<cache.active, 0..., ..<maxOffset, 0...]

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allK, values: allV,
            scale: scale, mask: .array(mask)
        )

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// `Qwen3MLP` is now an alias for the shared `Qwen3.MLP` in MLXLMCommon.
// Issue #168 consolidation pass — the SwiGLU MLP is bit-identical between
// the Qwen 3 LLM and Qwen3VL.
typealias Qwen3MLP = Qwen3.MLP

class Qwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    let mlp: Qwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(args)
        self.mlp = Qwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }

    /// Fully batched forward with shared BatchedKVCache — zero loops.
    public func fullyBatchedForward(_ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
                                     mask: MLXArray) -> MLXArray {
        let normed = inputLayerNorm(x)
        var r = attention.fullyBatchedForward(normed, cache: cache, layerIndex: layerIndex,
                                              mask: mask)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }

    /// Batched forward: B requests, batched norms + MLP, per-request attention.
    public func batchedForward(_ x: MLXArray, caches: [KVCache?]) -> MLXArray {
        let normed = inputLayerNorm(x)                // [B, 1, hidden] batched
        var r = attention.batchedForward(normed, caches: caches)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))            // [B, 1, hidden] batched
        return h + r
    }
}

public class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    /// Fully batched forward: shared per-layer BatchedKVCaches.
    public func fullyBatchedForward(_ inputs: MLXArray, caches: [BatchedKVCache]) -> MLXArray {
        var h = embedTokens(inputs)

        // When all requests have same offset (continuous decode),
        // all cache positions are valid — no mask needed.
        // For mixed offsets, would need per-request mask.
        let allSame = caches[0].offsets[0..<caches[0].active]
            .allSatisfy { $0 == caches[0].offsets[0] }

        let B = caches[0].active
        let cacheDtype = caches[0].keys.dtype
        // Post-update max offset (all advance by 1)
        let maxPostOffset = (caches[0].offsets[0..<B].max() ?? 0) + 1
        let mask: MLXArray
        if allSame {
            mask = MLXArray.zeros([B, 1, 1, maxPostOffset], dtype: cacheDtype)
        } else {
            let positions = MLXArray(0..<maxPostOffset).reshaped(1, maxPostOffset)
            let offsetsArr = MLXArray(caches[0].offsets[0..<B].map { $0 + 1 }).reshaped(B, 1)
            let valid = positions .< offsetsArr
            mask = MLX.where(valid,
                             MLXArray(Float(0)).asType(cacheDtype),
                             MLXArray(Float(-1e9)).asType(cacheDtype))
                .reshaped(B, 1, 1, maxPostOffset)
        }

        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedForward(h, cache: caches[i], layerIndex: i, mask: mask)
        }
        return norm(h)
    }

    /// Batched forward: B requests with separate per-layer caches.
    /// caches: [[KVCache]] — outer is per-request, inner is per-layer.
    public func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var h = embedTokens(inputs)  // [B, 1, hidden] batched

        for (i, layer) in layers.enumerated() {
            let layerCaches = caches.map { $0[i] as KVCache? }
            h = layer.batchedForward(h, caches: layerCaches)
        }

        return norm(h)
    }
}

public class Qwen3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3ModelInner
    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Fully batched decode: zero per-request loops.
    public func fullyBatchedDecode(_ inputs: MLXArray, caches: [BatchedKVCache]) -> MLXArray {
        var out = model.fullyBatchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Batched decode: B requests, batched projections + MLP, per-request attention.
    /// inputs: [B, 1] token IDs. caches: B arrays of per-layer KVCache.
    /// Returns: [B, 1, vocab] logits.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var out = model.batchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = configuration.hiddenLayers
        let env = ProcessInfo.processInfo.environment
        let enabled = env["VLLM_TRIATT_ENABLED"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

        if enabled, parameters?.maxKVSize == nil {
            let engine = TriAttentionV3Engine(
                cfg: .fromEnv(),
                nLayers: configuration.hiddenLayers,
                nHeads: configuration.attentionHeads,
                nKVHeads: configuration.kvHeads,
                headDim: configuration.headDim,
                ropeTheta: configuration.ropeTheta
            )
            TriAttentionRescue.shared.install(on: engine)
            return (0..<numLayers).map { layerIdx in
                TriAttentionKVCache(layerIdx: layerIdx, engine: engine)
            }
        }

        // Eric's spec-006 cleanup: factories use `makeAttentionCache`
        // which routes to StandardKVCache (or eviction-windowed variant)
        // based on parameters + maxSize, instead of instantiating
        // KVCacheSimple/RotatingKVCache directly.
        return (0..<numLayers).map { _ in
            makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
        }
    }
}

public struct Qwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension Qwen3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
