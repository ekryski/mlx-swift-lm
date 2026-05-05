import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of Gemma 4 VLM: SigLIP vision encoder + Gemma 4 text decoder
// Based on the Gemma3 VLM wrapper pattern in this repo

// MARK: - Text Configuration

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
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    let finalLogitSoftcapping: Float?
    let tieWordEmbeddings: Bool
    let ropeTheta: Float
    let globalRopeTheta: Float
    let partialRotaryFactor: Float

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
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case globalRopeTheta = "global_rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 30
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 704
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        // globalKvHeads: null/missing in E2B/E4B -> fall back to kvHeads
        let decodedGlobalKvHeads = try container.decodeIfPresent(
            Int.self, forKey: .globalKvHeads)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        globalKvHeads = decodedGlobalKvHeads ?? kvHeads

        // RoPE: try nested rope_parameters first, then top-level fallbacks
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
        try container.encode(hiddenSizePerLayerInput, forKey: .hiddenSizePerLayerInput)
        try container.encode(vocabSizePerLayerInput, forKey: .vocabSizePerLayerInput)
        try container.encodeIfPresent(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(globalRopeTheta, forKey: .globalRopeTheta)
        try container.encode(partialRotaryFactor, forKey: .partialRotaryFactor)
    }
}

// MARK: - Vision Configuration

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let patchSize: Int
    public let imageSize: Int

    public let numChannels: Int = 3
    public let layerNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case imageSize = "image_size"
    }
}

// MARK: - Model Configuration

public struct Gemma4Configuration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String
    public let mmTokensPerImage: Int
    public let quantization: BaseConfiguration.Quantization?

    private let _vocabularySize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    public var hiddenSize: Int {
        textConfiguration.hiddenSize
    }

    public var padTokenId: Int {
        _padTokenId ?? 0
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case mmTokensPerImage = "mm_tokens_per_image"
        case quantization

        case _vocabularySize = "vocab_size"
        case _padTokenId = "pad_token_id"
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

// MARK: - ProportionalRoPE

/// Proportional RoPE for Gemma 4 global attention layers.
/// Rotates `rotatedDims` out of `dims` dimensions using the full `dims` as
/// the frequency denominator. Non-rotated dimensions get inf frequency (pass-through).
private class Gemma4ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(dims: Int, rotatedDims: Int, traditional: Bool = false, base: Float) {
        precondition(rotatedDims <= dims, "rotatedDims must be <= dims")
        self.dims = dims
        self.traditional = traditional

        let exponents = MLXArray(
            stride(from: Float(0), to: Float(rotatedDims), by: 2)
        ) / Float(dims)
        let realFreqs = pow(MLXArray(base), exponents)

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

// MARK: - Text Attention

private class Gemma4TextAttention: Module {
    let isSliding: Bool
    let attentionKEqV: Bool
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rmsNormEps: Float
    let rope: any OffsetLayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
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
            let ropeDim = Int(Float(headDim) * config.partialRotaryFactor)
            self.rope = Gemma4ProportionalRoPE(
                dims: headDim, rotatedDims: ropeDim,
                traditional: false, base: config.globalRopeTheta)
        }

        super.init()
    }

    /// Forward pass with KV sharing support.
    /// - Parameters:
    ///   - useSharedKV: When true, skip K/V projection and use donor's cached K/V.
    ///   - sharedKVArrays: The donor layer's cached (keys, values) from peek().
    ///   - donorOffset: The donor layer's cache offset for correct RoPE positioning.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        useSharedKV: Bool = false,
        sharedKVArrays: (MLXArray, MLXArray)? = nil,
        donorOffset: Int? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, -1)

        if useSharedKV, let (cachedKeys, cachedValues) = sharedKVArrays {
            let offset = donorOffset ?? cache?.offset ?? 0
            queries = qNorm(queries)
            queries = queries.transposed(0, 2, 1, 3)
            queries = rope(queries, offset: offset)

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
        var keys = kProj(x).reshaped(B, L, nKVHeads, -1)
        var values: MLXArray
        if attentionKEqV {
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, nKVHeads, -1)
        }

        // v_norm: RMSNorm with no learnable scale (mlxNone weight)
        values = MLXFast.rmsNorm(values, weight: MLXArray.mlxNone, eps: rmsNormEps)

        queries = qNorm(queries)
        keys = kNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        values = values.transposed(0, 2, 1, 3)

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

// MARK: - Text Shared MLP (GeGLU)

private class Gemma4TextSharedMLP: Module {
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

// MARK: - Text Router

private class Gemma4TextRouter: Module {
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

// MARK: - Text Transformer Block

private class Gemma4TextTransformerBlock: Module {
    let enableMoeBlock: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4TextAttention
    @ModuleInfo(key: "mlp") var sharedMLP: Gemma4TextSharedMLP
    @ModuleInfo(key: "experts") var experts: SwitchGLU?
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    let topKExperts: Int
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?

    // Per-layer scalar (learnable output scaling)
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    // Per-Layer Embeddings (PLE)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.enableMoeBlock = config.enableMoeBlock
        self.topKExperts = config.topKExperts

        self._selfAttention.wrappedValue = Gemma4TextAttention(config, layerIdx: layerIdx)

        // use_double_wide_mlp only applies to KV-shared layers
        let firstKvSharedIdx = config.hiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = config.numKvSharedLayers > 0 && layerIdx >= firstKvSharedIdx
        let isDoubleWide = config.useDoubleWideMlp && isKvSharedLayer

        self._sharedMLP.wrappedValue = Gemma4TextSharedMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize,
            isDoubleWide: isDoubleWide)

        if config.enableMoeBlock {
            self._experts.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                activation: geluApproximate,
                bias: false
            )
            self._router.wrappedValue = Gemma4TextRouter(
                dimensions: config.hiddenSize, numExperts: config.numExperts,
                eps: config.rmsNormEps)
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
        sharedKVArrays: (MLXArray, MLXArray)? = nil,
        donorOffset: Int? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let attnOut = selfAttention(
            inputNorm, mask: mask, cache: cache,
            useSharedKV: useSharedKV, sharedKVArrays: sharedKVArrays,
            donorOffset: donorOffset)

        var h = postAttentionLayerNorm(attnOut) + x

        // FFN: shared MLP + optional MoE
        if let experts, let router,
           let postNorm1 = postFeedforwardLayerNorm1,
           let preNorm2 = preFeedforwardLayerNorm2,
           let postNorm2 = postFeedforwardLayerNorm2
        {
            let preFFNNorm = preFeedforwardLayerNorm(h)
            var h1 = sharedMLP(preFFNNorm)
            h1 = postNorm1(h1)

            let routerLogits = router(h)
            let (topKLogits, topKIndices) = gemma4TopK(routerLogits, k: topKExperts, axis: -1)
            let stopIndices = MLX.stopGradient(topKIndices)
            var expertWeights = softmax(topKLogits, axis: -1, precise: true)
            expertWeights = expertWeights * router.perExpertScale[topKIndices]
            let preFFNNorm2 = preNorm2(h)
            var h2 = experts(preFFNNorm2, stopIndices)
            h2 = h2 * expandedDimensions(expertWeights, axis: -1)
            h2 = h2.sum(axis: -2)
            h2 = postNorm2(h2)

            let ffnOut = h1 + h2
            h = postFeedforwardLayerNorm(ffnOut) + h
        } else {
            let preFFNNorm = preFeedforwardLayerNorm(h)
            let ffnOut = sharedMLP(preFFNNorm)
            h = postFeedforwardLayerNorm(ffnOut) + h
        }

        // Per-Layer Embeddings (PLE)
        if let gate = perLayerInputGate,
           let proj = perLayerProjection,
           let norm = postPerLayerInputNorm,
           let pli = perLayerInput
        {
            let residual = h
            var g = geluApproximate(gate(h)) * pli
            g = proj(g)
            g = norm(g)
            h = residual + g
        }

        h = h * layerScalar

        return h
    }
}

// MARK: - Text Inner Model

private class Gemma4TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4TextTransformerBlock]
    @ModuleInfo var norm: RMSNorm

    // Per-Layer Embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // PLE scaling factors -- computed constants, NOT learned parameters
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    let perLayerInputScale: Float

    let config: Gemma4TextConfiguration
    let hiddenSizePerLayerInput: Int
    let layerTypes: [String]
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int
    let previousKVs: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.layerTypes = config.layerTypes

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TextTransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

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

        // PLE scaling constants
        self.embedTokensPerLayerScale = sqrt(Float(max(config.hiddenSizePerLayerInput, 1)))
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
        self.perLayerInputScale = pow(2.0, -0.5)

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
        _ inputs: MLXArray? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding
        } else if let inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Either inputs or inputEmbedding must be provided")
        }

        // Gemma embedding scaling
        let embedScale = sqrt(Float(config.hiddenSize))
        h = h * embedScale

        let cache: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)

        // Per-Layer Embeddings (PLE): compute per-layer inputs from token IDs
        var perLayerInputs: MLXArray? = nil
        if hiddenSizePerLayerInput > 0, let embedPL = embedTokensPerLayer, let inputs {
            var pli = embedPL(inputs)
            pli = pli * embedTokensPerLayerScale
            pli = pli.reshaped(
                pli.dim(0), pli.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)

            if let proj = perLayerModelProjection {
                var plProj = proj(h)
                plProj = plProj * perLayerProjectionScale
                plProj = plProj.reshaped(
                    plProj.dim(0), plProj.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)
                if let norm = perLayerProjectionNorm {
                    plProj = norm(plProj)
                }
                pli = (plProj + pli) * perLayerInputScale
            }
            perLayerInputs = pli
        }

        // Pre-compute masks once per forward pass
        let seqLen = h.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        // Track donor offsets and K/V for shared layers
        var donorPreUpdateOffsets = [Int](repeating: 0, count: layers.count)
        var donorKVs: [(MLXArray, MLXArray)?] = Array(repeating: nil, count: layers.count)

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

            let pli: MLXArray? = perLayerInputs.map { $0[0..., 0..., i, 0...] }

            let donorIdx = previousKVs[i]
            let isShared = donorIdx != i

            if isShared {
                h = layer(
                    h, mask: maskMode, cache: nil, perLayerInput: pli,
                    useSharedKV: true, sharedKVArrays: donorKVs[donorIdx],
                    donorOffset: donorPreUpdateOffsets[donorIdx])
            } else {
                donorPreUpdateOffsets[i] = cache[i]?.offset ?? 0
                h = layer(h, mask: maskMode, cache: cache[i], perLayerInput: pli)

                if let c = cache[i] {
                    donorKVs[i] = c.peek()
                }
            }
        }

        return norm(h)
    }
}

// MARK: - Text Language Model

private class Gemma4LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: Gemma4TextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: Gemma4TextConfiguration
    var kvHeads: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4TextModelInner(config)

        self.kvHeads = config.layerTypes.map { layerType in
            layerType == "full_attention" ? config.globalKvHeads : config.kvHeads
        }

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches = [KVCache]()

        for layerType in config.layerTypes {
            let maxSize: Int? =
                (layerType == "full_attention") ? parameters?.maxKVSize : config.slidingWindow
            caches.append(makeAttentionCache(parameters: parameters, maxSize: maxSize))
        }

        return caches
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> LMOutput {
        let optionalCache = cache?.map { $0 as KVCache? }
        var out = model(
            inputs, inputEmbedding: inputEmbedding, mask: mask, cache: optionalCache)

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

        return LMOutput(logits: out)
    }

    func sanitize(
        weights: [String: MLXArray], quantizationConfig: BaseConfiguration.Quantization? = nil
    ) -> [String: MLXArray] {
        var processedWeights = weights

        // Handle weight tying
        if config.tieWordEmbeddings {
            processedWeights.keys
                .filter { $0.hasPrefix("language_model.lm_head.") }
                .forEach { processedWeights.removeValue(forKey: $0) }
        }

        // Trim vocabulary tensors to expected size
        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
            "language_model.lm_head.weight",
            "language_model.lm_head.scales",
            "language_model.lm_head.biases",
        ]

        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        // Remap expert weight names: experts.switch_glu.X -> experts.X
        let expertKeys = processedWeights.keys.filter { $0.contains(".switch_glu.") }
        for key in expertKeys {
            let newKey = key.replacingOccurrences(of: ".switch_glu.", with: ".")
            processedWeights[newKey] = processedWeights.removeValue(forKey: key)
        }

        // Remove unused precomputed rotary freqs
        return processedWeights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

// MARK: - Vision Model Components (SigLIP-based)

private class Gemma4VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outputProj: Linear

    let numHeads: Int
    let scale: Float

    init(dimensions: Int, numHeads: Int, bias: Bool = true) {
        self.numHeads = numHeads
        let headDim = dimensions / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._queryProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._keyProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._valueProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        self._outputProj.wrappedValue = Linear(dimensions, dimensions, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        let (B, L, _) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let S = keys.dim(1)

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

private class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo var activationFn: GELU

    init(config: Gemma4VisionConfiguration) {
        self.activationFn = GELU(approximation: .precise)
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = activationFn(x)
        return fc2(x)
    }
}

private class Gemma4VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    let embedDim: Int

    init(config: Gemma4VisionConfiguration) {
        self.embedDim = config.hiddenSize

        self._selfAttention.wrappedValue = Gemma4VisionAttention(
            dimensions: config.hiddenSize,
            numHeads: config.attentionHeads,
            bias: true
        )

        self._layerNorm1.wrappedValue = LayerNorm(dimensions: embedDim, eps: config.layerNormEps)
        self.mlp = Gemma4VisionMLP(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: embedDim, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none)
        -> MLXArray
    {
        let r = selfAttention(layerNorm1(x), mask: mask)
        let h = x + r
        let r2 = mlp(layerNorm2(h))
        return h + r2
    }
}

private class Gemma4VisionEncoder: Module {
    @ModuleInfo var layers: [Gemma4VisionEncoderLayer]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4VisionEncoderLayer(config: config)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> (MLXArray, [MLXArray]?) {
        var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
        var h = x

        for layer in layers {
            h = layer(h, mask: mask)
            if outputHiddenStates {
                encoderStates?.append(h)
            }
        }

        return (h, encoderStates)
    }
}

private class Gemma4VisionEmbeddings: Module, UnaryLayer {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let config: Gemma4VisionConfiguration
    let embedDim: Int
    let imageSize: Int
    let patchSize: Int
    let numPatches: Int
    let numPositions: Int

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.embedDim = config.hiddenSize
        self.imageSize = config.imageSize
        self.patchSize = config.patchSize

        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize)
        )

        self.numPatches = (imageSize / patchSize) * (imageSize / patchSize)
        self.numPositions = numPatches

        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPositions,
            dimensions: embedDim
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var patchEmbeddings = patchEmbedding(x)
        patchEmbeddings = flattened(patchEmbeddings, start: 1, end: 2)

        let actualNumPatches = patchEmbeddings.dim(1)
        let useNumPositions = min(actualNumPatches, numPositions)

        let positionIds = MLXArray(Array(0 ..< useNumPositions))[.newAxis, 0...]
        var embeddings = patchEmbeddings

        if useNumPositions == actualNumPatches {
            embeddings = embeddings + positionEmbedding(positionIds)
        } else {
            let positionedPatches =
                embeddings[0..., ..<useNumPositions, 0...] + positionEmbedding(positionIds)
            let remainingPatches = embeddings[0..., useNumPositions..., 0...]
            embeddings = concatenated([positionedPatches, remainingPatches], axis: 1)
        }

        return embeddings
    }
}

private class Gemma4SigLipVisionModel: Module {
    @ModuleInfo var embeddings: Gemma4VisionEmbeddings
    @ModuleInfo var encoder: Gemma4VisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(config: Gemma4VisionConfiguration) {
        self.embeddings = Gemma4VisionEmbeddings(config: config)
        self.encoder = Gemma4VisionEncoder(config: config)
        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> (MLXArray, MLXArray, [MLXArray]?) {
        let x = embeddings(x)
        let (encoderOutput, encoderStates) = encoder(
            x, outputHiddenStates: outputHiddenStates, mask: .none)
        let poolerOutput = postLayerNorm(encoderOutput)
        return (poolerOutput, x, encoderStates)
    }
}

private class Gemma4VisionModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: Gemma4SigLipVisionModel

    let modelType: String

    init(config: Gemma4VisionConfiguration) {
        self.modelType = config.modelType
        self._visionModel.wrappedValue = Gemma4SigLipVisionModel(config: config)
    }

    func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> (MLXArray, MLXArray, [MLXArray]?) {
        visionModel(x, outputHiddenStates: outputHiddenStates)
    }

    private func checkArrayShape(_ arr: MLXArray) -> Bool {
        let shape = arr.shape
        guard shape.count == 4 else { return false }
        let (outChannels, kH, kW, _) = (shape[0], shape[1], shape[2], shape[3])
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        for (k, v) in weights {
            if k.contains("patch_embedding.weight") {
                // PyTorch conv2d: [out_channels, in_channels, kH, kW]
                // MLX conv2d:     [out_channels, kH, kW, in_channels]
                if checkArrayShape(v) {
                    sanitizedWeights[k] = v
                } else {
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                }
            } else {
                sanitizedWeights[k] = v
            }
        }

        return sanitizedWeights
    }
}

// MARK: - Multimodal Projector

class Gemma4MultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "mm_input_projection_weight") var mmInputProjectionWeight: MLXArray
    @ModuleInfo(key: "mm_soft_emb_norm") var mmSoftEmbNorm: RMSNorm
    @ModuleInfo var avgPool: AvgPool2d

    let config: Gemma4Configuration
    let patchesPerImage: Int
    let tokensPerSide: Int
    let kernelSize: Int

    init(config: Gemma4Configuration) {
        self.config = config

        self._mmInputProjectionWeight.wrappedValue = ones([
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
        ])

        self._mmSoftEmbNorm.wrappedValue = RMSNorm(
            dimensions: config.visionConfiguration.hiddenSize,
            eps: config.visionConfiguration.layerNormEps
        )

        self.patchesPerImage =
            config.visionConfiguration.imageSize / config.visionConfiguration.patchSize

        self.tokensPerSide = Int(sqrt(Double(config.mmTokensPerImage)))
        self.kernelSize = patchesPerImage / tokensPerSide

        self.avgPool = AvgPool2d(
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(kernelSize)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, _, l) = (x.dim(0), x.dim(1), x.dim(2))

        var reshapedVisionOutputs = x.transposed(0, 2, 1)
        reshapedVisionOutputs = reshapedVisionOutputs.reshaped(
            b, l, patchesPerImage, patchesPerImage
        )

        reshapedVisionOutputs = reshapedVisionOutputs.transposed(0, 2, 3, 1)
        var pooledVisionOutputs = avgPool(reshapedVisionOutputs)
        pooledVisionOutputs = pooledVisionOutputs.transposed(0, 3, 1, 2).flattened(start: 2)
        pooledVisionOutputs = pooledVisionOutputs.transposed(0, 2, 1)

        let normedVisionOutputs = mmSoftEmbNorm(pooledVisionOutputs)

        let projectedVisionOutputs = einsum(
            "btm,md->btd",
            normedVisionOutputs,
            mmInputProjectionWeight
        )

        return projectedVisionOutputs.asType(x.dtype)
    }
}

/// Inserts image features into text embeddings at specified token positions
private func gemma4MaskedScatter(
    finalEmbedding: MLXArray,
    imageMaskExpanded: MLXArray,
    scaledImageFeatures: MLXArray
) -> MLXArray {
    let finalEmbeddingShape = finalEmbedding.shape
    let scaledImageFeaturesFlattened = scaledImageFeatures.flattened()
    let finalEmbeddingFlattened = finalEmbedding.flattened()
    let imageMaskExpandedFlattened = imageMaskExpanded.flattened()

    let expectedCount = scaledImageFeaturesFlattened.shape[0]
    let actualTrueCount = imageMaskExpandedFlattened
        .asType(.int32).sum().item(Int.self)
    guard actualTrueCount == expectedCount else {
        if actualTrueCount == 0 {
            return finalEmbedding
        }
        fatalError(
            """
            gemma4MaskedScatter: Size mismatch between image features and positions.
            Image features: \(expectedCount)
            Image positions: \(actualTrueCount)
            """)
    }
    guard expectedCount > 0 else {
        return finalEmbedding
    }

    // argWhere stays on GPU; one .item() for the count replaces a full
    // .asArray(Bool.self) readback of the entire mask.
    let rawIndices = argWhere(
        imageMaskExpandedFlattened.asType(.bool), count: expectedCount)
    let imagePositions = rawIndices.asType(DType.uint32)
    finalEmbeddingFlattened[imagePositions] = scaledImageFeaturesFlattened
    return finalEmbeddingFlattened.reshaped(finalEmbeddingShape)
}

// MARK: - Gemma4 VLM Model

public class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Gemma4MultiModalProjector

    public let config: Gemma4Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

    public init(_ config: Gemma4Configuration) {
        self.config = config

        self._visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4LanguageModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = Gemma4MultiModalProjector(config: config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        guard let pixelValues else {
            return (languageModel.model.embedTokens(inputIds!), nil)
        }

        let inputsEmbeds = languageModel.model.embedTokens(inputIds!)

        // Process image through vision tower
        let processedPixels = pixelValues.transposed(0, 2, 3, 1).asType(inputsEmbeds.dtype)

        let (hiddenState, _, _) = visionTower(
            processedPixels,
            outputHiddenStates: true
        )

        let imageFeatures = multiModalProjector(hiddenState)

        let (finalEmbedding, finalAttentionMask4d) = prepareInputsForMultimodal(
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds!,
            attentionMask: mask
        )

        return (finalEmbedding, finalAttentionMask4d)
    }

    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let embedDim = inputsEmbeds.dim(2)

        // Scale image features to match text embedding magnitude
        let scaledImageFeatures =
            imageFeatures / sqrt(Float(config.textConfiguration.hiddenSize))

        var finalEmbedding = inputsEmbeds

        let padTokenId = config.padTokenId
        let imageTokenId = 262144  // Image token used after expansion

        // Create masks for different token types
        let imageMask = MLX.equal(inputIds, MLXArray(imageTokenId))
        let padMask = MLX.equal(inputIds, MLXArray(padTokenId))

        // Expand masks to match embedding dimension
        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = repeated(imageMaskExpanded, count: embedDim, axis: -1)

        var padMaskExpanded = expandedDimensions(padMask, axis: -1)
        padMaskExpanded = repeated(padMaskExpanded, count: embedDim, axis: -1)
        finalEmbedding = MLX.where(
            padMaskExpanded, MLXArray.zeros(like: finalEmbedding), finalEmbedding)

        // Insert image token embeddings using masked scatter
        finalEmbedding = gemma4MaskedScatter(
            finalEmbedding: finalEmbedding,
            imageMaskExpanded: imageMaskExpanded,
            scaledImageFeatures: scaledImageFeatures
        )

        var finalAttentionMask4d: MLXArray? = nil
        if let attentionMask = attentionMask {
            let attentionMaskExpanded1 = expandedDimensions(attentionMask, axis: 1)
            let attentionMaskExpanded2 = expandedDimensions(attentionMask, axis: 2)
            finalAttentionMask4d = attentionMaskExpanded1 * attentionMaskExpanded2
            finalAttentionMask4d = expandedDimensions(finalAttentionMask4d!, axis: 1)
        }

        return (finalEmbedding.asType(inputsEmbeds.dtype), finalAttentionMask4d)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let imagePixels = input.image?.pixels else {
            // Text-only input
            let convertedCache = cache.compactMap { $0 as KVCache }
            let result = languageModel(
                input.text.tokens, cache: convertedCache, inputEmbedding: nil, mask: nil)
            return .logits(result)
        }

        let (inputEmbeddings, _) = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: imagePixels,
            mask: input.text.mask
        )

        let convertedCache = cache.compactMap { $0 as KVCache }
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = .causal

        let result = languageModel(
            nil,
            cache: convertedCache,
            inputEmbedding: inputEmbeddings,
            mask: maskMode
        )

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        return languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = languageModel.sanitize(
            weights: weights, quantizationConfig: config.quantization)
        processedWeights = visionTower.sanitize(weights: processedWeights)
        return processedWeights
    }
}

// MARK: - Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        var userProcessing = processing ?? UserInput.Processing()
        let targetSize = CGSize(width: config.imageSize, height: config.imageSize)

        userProcessing.resize = targetSize

        let processedImages = images.map { image in
            let processedImage = MediaProcessing.apply(image, processing: userProcessing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
            let normalizedImage = MediaProcessing.normalize(
                resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
            return MediaProcessing.asMLXArray(normalizedImage)
        }

        let pixelValues = concatenated(processedImages)

        return (pixelValues, THW(images.count, config.imageSize, config.imageSize))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?

        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated,
                frames: imagePixelsAndFrames.map { $0.1 }
            )

            // Expand single <start_of_image> token to multiple image tokens
            let startOfImageTokenId = 255999
            let imageTokenId = 262144
            let numImageTokens = config.imageSeqLength

            var expandedTokens: [Int] = []

            for token in promptTokens {
                if token == startOfImageTokenId {
                    expandedTokens.append(
                        contentsOf: Array(repeating: imageTokenId, count: numImageTokens))
                } else {
                    expandedTokens.append(token)
                }
            }

            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let imageProcessorType: String
    public let doNormalize: Bool
    public let doRescale: Bool
    public let doResize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let resample: Int
    public let rescaleFactor: Float
    public let size: ImageSize

    public let doConvertRgb: Bool?
    public let doPanAndScan: Bool?
    public let panAndScanMaxNumCrops: Int?
    public let panAndScanMinCropSize: Int?
    public let panAndScanMinRatioToActivate: Float?

    public let imageTokenId: Int = 262144

    public struct ImageSize: Codable, Sendable {
        public let height: Int
        public let width: Int
    }

    public var imageSize: Int { size.height }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessorType = "image_processor_type"
        case doNormalize = "do_normalize"
        case doRescale = "do_rescale"
        case doResize = "do_resize"
        case doConvertRgb = "do_convert_rgb"
        case doPanAndScan = "do_pan_and_scan"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case resample
        case rescaleFactor = "rescale_factor"
        case size
        case panAndScanMaxNumCrops = "pan_and_scan_max_num_crops"
        case panAndScanMinCropSize = "pan_and_scan_min_crop_size"
        case panAndScanMinRatioToActivate = "pan_and_scan_min_ratio_to_activate"
    }
}

// MARK: - LoRA

extension Gemma4: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
