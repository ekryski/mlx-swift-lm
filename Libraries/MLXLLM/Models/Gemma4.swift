//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Created by Claude on 2026/4/6.
//

// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let globalHeadDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let globalKvHeads: Int
    let numExperts: Int
    let topKExperts: Int
    let slidingWindow: Int
    let layerTypes: [String]
    let enableMoeBlock: Bool
    let attentionKEqV: Bool
    let useDoubleWideMlp: Bool
    let numKvSharedLayers: Int
    let finalLogitSoftcapping: Float?
    let tieWordEmbeddings: Bool
    let ropeTheta: Float
    let globalRopeTheta: Float
    let partialRotaryFactor: Float
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case globalKvHeads = "num_global_key_value_heads"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case enableMoeBlock = "enable_moe_block"
        case attentionKEqV = "attention_k_eq_v"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case numKvSharedLayers = "num_kv_shared_layers"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"

        // RoPE config — may be at top level or nested in rope_parameters
        case ropeTheta = "rope_theta"
        case globalRopeTheta = "global_rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"

        // Per-Layer Embeddings (PLE)
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 30
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 704
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        // globalKvHeads: null/missing in E2B/E4B → fall back to kvHeads
        let decodedGlobalKvHeads = try container.decodeIfPresent(Int.self, forKey: .globalKvHeads)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        enableMoeBlock = try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        // MoE params: null/missing for dense models
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Derive globalKvHeads: explicit value → kvHeads fallback
        globalKvHeads = decodedGlobalKvHeads ?? kvHeads

        // RoPE: try nested rope_parameters first (actual config format), then top-level fallbacks
        if let ropeParams = try container.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        {
            let slidingRope = ropeParams["sliding_attention"] ?? [:]
            let fullRope = ropeParams["full_attention"] ?? [:]

            ropeTheta = slidingRope["rope_theta"]?.asFloat() ?? 10_000.0
            globalRopeTheta = fullRope["rope_theta"]?.asFloat() ?? 1_000_000.0
            partialRotaryFactor = fullRope["partial_rotary_factor"]?.asFloat() ?? 0.25
        } else {
            ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
            globalRopeTheta =
                try container.decodeIfPresent(Float.self, forKey: .globalRopeTheta) ?? 1_000_000.0
            partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
        }

        // Per-Layer Embeddings (PLE)
        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(globalHeadDim, forKey: .globalHeadDim)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(globalKvHeads, forKey: .globalKvHeads)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(topKExperts, forKey: .topKExperts)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(enableMoeBlock, forKey: .enableMoeBlock)
        try container.encode(attentionKEqV, forKey: .attentionKEqV)
        try container.encode(useDoubleWideMlp, forKey: .useDoubleWideMlp)
        try container.encode(numKvSharedLayers, forKey: .numKvSharedLayers)
        try container.encodeIfPresent(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(globalRopeTheta, forKey: .globalRopeTheta)
        try container.encode(partialRotaryFactor, forKey: .partialRotaryFactor)
        try container.encode(hiddenSizePerLayerInput, forKey: .hiddenSizePerLayerInput)
        try container.encode(vocabSizePerLayerInput, forKey: .vocabSizePerLayerInput)
    }
}

// MARK: - Top-K helper

private func gemma4TopK(_ a: MLXArray, k: Int, axis: Int = -1) -> (
    values: MLXArray, indices: MLXArray
) {
    let partitionedIndices = argPartition(a, kth: -k, axis: axis)
    let topKIndices = partitionedIndices[.ellipsis, (-k)...]
    let topKValues = takeAlong(a, topKIndices, axis: axis)
    return (topKValues, topKIndices)
}

// v_norm uses MLXFast.rmsNorm with a ones weight (no learnable scale).
// This fuses the 3-dispatch manual implementation (square, mean, rsqrt*mul)
// into a single optimized kernel dispatch.

// MARK: - ProportionalRoPE

/// Proportional RoPE: rotates `rotatedDims` out of `dims` dimensions, using the
/// FULL `dims` for both frequency computation and dimension pairing.
///
/// Matches Python's `ProportionalRoPE` in `rope_utils.py`.
///
/// **Why this exists:** `RoPE(dimensions: 128)` on a 512-dim head pairs `(x[0], x[64])`,
/// `(x[1], x[65])`, etc. — within the first 128 dims. But the model weights were trained
/// with `ProportionalRoPE(dims=512, rotatedDims=128)` which pairs `(x[0], x[256])`,
/// `(x[1], x[257])`, etc. — across the first and second halves of the full 512-dim head.
/// Non-rotated dimensions get `inf` frequency → zero rotation → pass-through.
class ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(dims: Int, rotatedDims: Int, traditional: Bool = false, base: Float, factor: Float = 1.0) {
        precondition(rotatedDims <= dims, "rotatedDims must be ≤ dims")
        self.dims = dims
        self.traditional = traditional

        // Exponents use the FULL dims as denominator (the "proportional" aspect)
        let exponents = MLXArray(
            stride(from: Float(0), to: Float(rotatedDims), by: 2)
        ) / Float(dims)
        let realFreqs = factor * pow(MLXArray(base), exponents)

        // Pad with inf for non-rotated pairs (zero rotation angle per position)
        let paddingCount = (dims - rotatedDims) / 2
        let infPadding = MLXArray(Array(repeating: Float.infinity, count: paddingCount))
        self._freqs = concatenated([realFreqs, infPadding], axis: 0)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dims, traditional: traditional, base: nil,
            scale: 1.0, offset: offset, freqs: _freqs)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, offset: 0)
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let isSliding: Bool
    let attentionKEqV: Bool
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    // Always has k_proj. When attention_k_eq_v and NOT sliding, values = keys (no v_proj).
    // When attention_k_eq_v is False OR this is a sliding layer, v_proj is separate.
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rmsNormEps: Float
    let rope: any OffsetLayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        // attention_k_eq_v only applies to non-sliding (global) layers
        self.attentionKEqV = config.attentionKEqV && !isSliding

        self.nHeads = config.attentionHeads

        if isSliding {
            self.headDim = config.headDim
            self.nKVHeads = config.kvHeads
        } else {
            self.headDim = config.globalHeadDim
            self.nKVHeads = config.globalKvHeads
        }

        self.repeats = nHeads / nKVHeads
        // Gemma4 uses scale=1.0, NOT 1/sqrt(head_dim)
        self.scale = 1.0

        let dim = config.hiddenSize
        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        // k_proj always exists; v_proj only when NOT using k_eq_v
        self._kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        if !attentionKEqV {
            self._vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        }

        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rmsNormEps = config.rmsNormEps

        if isSliding {
            self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        } else {
            // ProportionalRoPE: pairs are formed across the full head_dim (e.g., 512),
            // not just the rotated portion. Using plain RoPE(dims=128) would pair
            // (0,64),(1,65),... instead of the correct (0,256),(1,257),...
            let ropeDim = Int(Float(headDim) * config.partialRotaryFactor)
            self.rope = ProportionalRoPE(
                dims: headDim, rotatedDims: ropeDim,
                traditional: false, base: config.globalRopeTheta)
        }

        super.init()
    }

    /// Forward pass with KV sharing support.
    /// - Parameters:
    ///   - useSharedKV: When true, skip K/V projection and read cached K/V from the
    ///     donor's cache (passed as `cache`). Only compute Q, apply RoPE, and run SDPA.
    ///   - donorOffset: The donor layer's pre-update cache offset, captured before
    ///     `attentionWithCacheUpdate` ran. Shared layers must use this — NOT `cache.offset`,
    ///     which is the post-update value and would produce wrong RoPE positions.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        useSharedKV: Bool = false,
        donorOffset: Int? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        queries = qNorm(queries)

        if useSharedKV, let cache, let (cachedKeys, cachedValues) = cache.peek() {
            // KV sharing: skip K/V projection, read from donor's cache (already updated).
            // Use donorOffset (pre-update) so query positions match the donor's queries —
            // cache.offset is post-update here and would place queries one step too far.
            queries = rope(queries, offset: donorOffset ?? cache.offset)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(output)
        }

        // Normal path: compute own K/V
        var keys: MLXArray
        var values: MLXArray

        keys = kProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        if attentionKEqV {
            // K and V share the same projection (values = keys before norms)
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        }

        // Apply v_norm via fused kernel (mlxNone weight = no learnable scale)
        values = MLXFast.rmsNorm(values, weight: MLXArray.mlxNone, eps: rmsNormEps)

        keys = kNorm(keys)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Shared MLP

class Gemma4SharedMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int, isDoubleWide: Bool = false) {
        let effectiveHidden = isDoubleWide ? hiddenDimensions * 2 : hiddenDimensions
        self._gateProj.wrappedValue = Linear(dimensions, effectiveHidden, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, effectiveHidden, bias: false)
        self._downProj.wrappedValue = Linear(effectiveHidden, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Router

class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let rootSize: Float
    let eps: Float

    init(dimensions: Int, numExperts: Int, eps: Float) {
        self.rootSize = pow(Float(dimensions), -0.5)
        self.eps = eps
        self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([dimensions])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normWeight = scale * MLXArray(rootSize)
        let normed = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
        return proj(normed)
    }
}

// MARK: - Transformer Block

class Gemma4TransformerBlock: Module {
    let enableMoeBlock: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo(key: "mlp") var sharedMLP: Gemma4SharedMLP
    // MoE components live directly on the layer (no moe_block wrapper) to match weight keys
    @ModuleInfo(key: "experts") var experts: FusedGateUpSwitchGLU?
    @ModuleInfo(key: "router") var router: Gemma4Router?
    let topKExperts: Int
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?

    // Per-layer scalar (learnable output scaling, loaded from weights if present)
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    // Per-Layer Embeddings (PLE)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.enableMoeBlock = config.enableMoeBlock
        self.topKExperts = config.topKExperts

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        // use_double_wide_mlp only applies to KV-shared layers (near the end of the model)
        let firstKvSharedIdx = config.hiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = config.numKvSharedLayers > 0 && layerIdx >= firstKvSharedIdx
        let isDoubleWide = config.useDoubleWideMlp && isKvSharedLayer

        self._sharedMLP.wrappedValue = Gemma4SharedMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize,
            isDoubleWide: isDoubleWide)

        if config.enableMoeBlock {
            self._experts.wrappedValue = FusedGateUpSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                activation: geluApproximate,
                bias: false
            )
            self._router.wrappedValue = Gemma4Router(
                dimensions: config.hiddenSize, numExperts: config.numExperts, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // layer_scalar: initialized to [1.0] shape [1], overwritten by checkpoint if present
        self._layerScalar.wrappedValue = MLXArray([Float(1.0)])

        // PLE components (only when enabled)
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        useSharedKV: Bool = false,
        donorOffset: Int? = nil
    ) -> MLXArray {
        // Attention with pre/post norms and residual
        let inputNorm = inputLayerNorm(x)
        let attnOut = selfAttention(inputNorm, mask: mask, cache: cache, useSharedKV: useSharedKV, donorOffset: donorOffset)
        let attnNorm = postAttentionLayerNorm(attnOut)
        var h = x + attnNorm

        // FFN with pre/post norms: shared MLP + MoE
        if let experts, let router,
           let postNorm1 = postFeedforwardLayerNorm1,
           let preNorm2 = preFeedforwardLayerNorm2,
           let postNorm2 = postFeedforwardLayerNorm2
        {
            // MoE path: separate norms for shared MLP and expert MoE
            let preFFNNorm = preFeedforwardLayerNorm(h)
            var h1 = sharedMLP(preFFNNorm)
            h1 = postNorm1(h1)

            // Route through experts: router gets h (pre-norm), experts get normed input
            let routerLogits = router(h)  // router has its own internal RMS norm
            let (topKLogits, topKIndices) = gemma4TopK(routerLogits, k: topKExperts, axis: -1)
            let stopIndices = MLX.stopGradient(topKIndices)
            var expertWeights = softmax(topKLogits, axis: -1, precise: true)
            expertWeights = expertWeights * router.perExpertScale[topKIndices]
            let preFFNNorm2 = preNorm2(h)
            var h2 = experts(preFFNNorm2, stopIndices)
            h2 = h2 * expandedDimensions(expertWeights, axis: -1)
            h2 = h2.sum(axis: -2)
            h2 = postNorm2(h2)

            var ffnOut = h1 + h2
            ffnOut = postFeedforwardLayerNorm(ffnOut)
            h = h + ffnOut
        } else {
            // Non-MoE path: single shared MLP
            let preFFNNorm = preFeedforwardLayerNorm(h)
            var ffnOut = sharedMLP(preFFNNorm)
            ffnOut = postFeedforwardLayerNorm(ffnOut)
            h = h + ffnOut
        }

        // Per-Layer Embeddings (PLE)
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let pli = perLayerInput
        {
            let residual = h
            var g = gate(h)
            g = geluApproximate(g)
            g = g * pli
            g = proj(g)
            g = norm(g)
            h = residual + g
        }

        // Per-layer scalar
        h = h * layerScalar

        return h
    }
}

// MARK: - Inner Model

public class Gemma4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    // Per-Layer Embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // PLE scaling factors — computed constants, NOT learned parameters
    let embedTokensPerLayerScale: Float   // sqrt(hidden_size_per_layer_input)
    let perLayerProjectionScale: Float    // hidden_size^(-0.5)
    let perLayerInputScale: Float         // 2^(-0.5) ≈ 0.707

    let config: Gemma4TextConfiguration
    let layerTypes: [String]
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int
    let hiddenSizePerLayerInput: Int
    /// KV sharing: maps each layer to its KV donor layer index.
    /// previousKVs[i] == i means the layer computes its own KV.
    /// previousKVs[i] != i means it reuses KV from layer previousKVs[i].
    let previousKVs: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.layerTypes = config.layerTypes

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Cache representative layer indices for mask creation
        self.slidingAttentionIndex = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.fullAttentionIndex = layerTypes.firstIndex(of: "full_attention") ?? 0

        // KV sharing: build donor mapping
        let N = config.hiddenLayers
        let M = N - config.numKvSharedLayers
        var mapping = Array(0 ..< N)
        if config.numKvSharedLayers > 0 {
            var kvsByType: [String: Int] = [:]
            for i in 0 ..< M {
                kvsByType[config.layerTypes[i]] = i
            }
            for j in M ..< N {
                if let donor = kvsByType[config.layerTypes[j]] {
                    mapping[j] = donor
                }
            }
        }
        self.previousKVs = mapping

        // PLE scaling constants (computed, not learned)
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.embedTokensPerLayerScale = sqrt(Float(max(config.hiddenSizePerLayerInput, 1)))
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
        self.perLayerInputScale = pow(2.0, -0.5)  // ≈ 0.707

        // Per-Layer Embeddings (PLE)
        if config.hiddenSizePerLayerInput > 0 {
            let plEmbedDim = config.hiddenLayers * config.hiddenSizePerLayerInput
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput, dimensions: plEmbedDim)
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize, plEmbedDim, bias: false)
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)

        // Gemma embedding scaling
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        let cache: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)

        // Per-Layer Embeddings (PLE): compute per-layer inputs from token IDs
        var perLayerInputs: MLXArray? = nil
        if hiddenSizePerLayerInput > 0, let embedPL = embedTokensPerLayer {
            // 1. Embed token IDs through per-layer embedding table
            var pli = embedPL(inputs)
            pli = pli * MLXArray(embedTokensPerLayerScale)
            // Reshape: [B, T, numLayers * plDim] -> [B, T, numLayers, plDim]
            pli = pli.reshaped(
                pli.dim(0), pli.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)

            // 2. Project hidden states and combine with per-layer embeddings
            if let proj = perLayerModelProjection {
                var plProj = proj(h)
                // Scale the projection output BEFORE reshaping
                plProj = plProj * MLXArray(perLayerProjectionScale)
                plProj = plProj.reshaped(
                    plProj.dim(0), plProj.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)
                if let norm = perLayerProjectionNorm {
                    plProj = norm(plProj)
                }
                // Combine: (projection + embedding) * scale
                pli = (plProj + pli) * MLXArray(perLayerInputScale)
            }
            perLayerInputs = pli
        }

        // Pre-compute masks once per forward pass (GPTOSS pattern)
        let seqLen = h.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        // Pre-update offsets captured from each donor layer before attentionWithCacheUpdate runs.
        // Python threads this explicitly via an `intermediates` array; here we replicate that by
        // snapshotting cache.offset just before each donor layer executes.
        // Shared layers need these to apply query RoPE at the same positions as their donor —
        // by the time a shared layer runs, cache.offset has already been incremented by the donor.
        var donorPreUpdateOffsets = [Int](repeating: 0, count: layers.count)

        for (i, layer) in layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                maskMode = mask
            } else if layerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = makeAttentionMask(
                        n: seqLen,
                        cache: cache[fullAttentionIndex],
                        windowSize: nil
                    )
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(
                        n: seqLen,
                        cache: cache[slidingAttentionIndex],
                        windowSize: config.slidingWindow
                    )
                }
                maskMode = slidingMask!
            }

            let pli = perLayerInputs?[0..., 0..., i, 0...]

            // KV sharing: shared layers use the donor's cache to read K/V
            let donorIdx = previousKVs[i]
            let isShared = donorIdx != i

            if isShared {
                // Pass the donor's pre-update offset so query RoPE uses the correct positions.
                h = layer(h, mask: maskMode, cache: cache[donorIdx], perLayerInput: pli,
                          useSharedKV: true, donorOffset: donorPreUpdateOffsets[donorIdx])
            } else {
                // Snapshot cache.offset BEFORE the donor runs; attentionWithCacheUpdate will
                // increment it, so any later read would give the wrong (post-update) value.
                donorPreUpdateOffsets[i] = cache[i]?.offset ?? 0
                h = layer(h, mask: maskMode, cache: cache[i], perLayerInput: pli)
            }
        }

        return norm(h)
    }
}

// MARK: - Gemma4 Text Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo public var model: Gemma4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: Gemma4TextConfiguration

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.model = Gemma4ModelInner(config)

        // Per-layer KV head counts for KVCacheDimensionProvider
        self.kvHeads = config.layerTypes.map { layerType in
            layerType == "full_attention" ? config.globalKvHeads : config.kvHeads
        }

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    /// Pure attention model — use larger prefill chunks (4096) since there's no
    /// GatedDeltaNet sequential bottleneck. Reduces TTFT by processing more tokens per step.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = max(windowSize ?? 512, 2048)
        var y = input.text

        while y.tokens.size > prefillStepSize {
            let input = y[.newAxis, ..<prefillStepSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            y = y[prefillStepSize...]
        }

        return .tokens(y)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else {
            out = lmHead!(out)
        }

        // Final logit softcapping
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            let scale = MLXArray(softcap)
            out = tanh(out / scale) * scale
        }

        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Strip language_model.model. prefix from VLM weights
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Trim vocabulary tensors to expected size
        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]

        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        // When tied, remove any lm_head weights (we use embed_tokens.asLinear instead)
        if config.tieWordEmbeddings {
            processedWeights.keys
                .filter { $0.hasPrefix("lm_head.") }
                .forEach { processedWeights.removeValue(forKey: $0) }
        }

        // Remap expert weight names: experts.switch_glu.X -> experts.X
        let expertKeys = processedWeights.keys.filter { $0.contains(".switch_glu.") }
        for key in expertKeys {
            let newKey = key.replacingOccurrences(of: ".switch_glu.", with: ".")
            processedWeights[newKey] = processedWeights.removeValue(forKey: key)
        }

        // Fuse gate_proj + up_proj into gate_up_proj for FusedGateUpSwitchGLU.
        // Handles weight, scales, and biases (quantized models).
        let gateKeys = processedWeights.keys.filter { $0.contains(".experts.gate_proj.") }
        for gateKey in gateKeys {
            let upKey = gateKey.replacingOccurrences(of: "gate_proj", with: "up_proj")
            guard let gateVal = processedWeights[gateKey],
                  let upVal = processedWeights[upKey] else { continue }

            let fusedKey = gateKey.replacingOccurrences(of: "gate_proj", with: "gate_up_proj")
            // Concat on output dimension (axis 1): [E, outDim, ...] → [E, 2*outDim, ...]
            processedWeights[fusedKey] = concatenated([gateVal, upVal], axis: 1)
            processedWeights.removeValue(forKey: gateKey)
            processedWeights.removeValue(forKey: upKey)
        }

        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()

        for layerType in config.layerTypes {
            if layerType == "full_attention" {
                if let maxKVSize = parameters?.maxKVSize {
                    caches.append(RotatingKVCache(maxSize: maxKVSize, keep: 0))
                } else {
                    caches.append(StandardKVCache())
                }
            } else {
                caches.append(
                    RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
