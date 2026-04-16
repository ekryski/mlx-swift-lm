// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py
// MiniMax M2: MoE architecture with 62 layers, 256 local experts, 8 active per token.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MiniMaxM2Configuration: Codable, Sendable {
    var modelType: String = "minimax_m2"
    var hiddenSize: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var numExpertsPerTok: Int
    var numLocalExperts: Int
    var sharedIntermediateSize: Int
    var hiddenLayers: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var rotaryDim: Int
    var vocabularySize: Int
    var tieWordEmbeddings: Bool = false
    var scoringFunc: String = "sigmoid"
    var headDim: Int
    var useQkNorm: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numExpertsPerTok = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case sharedIntermediateSize = "shared_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m2"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1536
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 48
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 196608
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        numLocalExperts = try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 256
        sharedIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? 0
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 62
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 5_000_000
        rotaryDim = try container.decodeIfPresent(Int.self, forKey: .rotaryDim) ?? 64
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200064
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
    }
}

// MARK: - Attention

class MiniMaxM2Attention: Module {
    let scale: Float
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let useQkNorm: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: MiniMaxM2Configuration) {
        self.numAttentionHeads = args.attentionHeads
        self.numKeyValueHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.useQkNorm = args.useQkNorm

        _wq.wrappedValue = Linear(args.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(numAttentionHeads * headDim, args.hiddenSize, bias: false)

        if useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: numAttentionHeads * headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(
                dimensions: numKeyValueHeads * headDim, eps: args.rmsNormEps)
        }

        self.rope = RoPE(
            dimensions: args.rotaryDim,
            traditional: false,
            base: args.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        let values = wv(x)

        // QK norm applied BEFORE reshape (per-layer norm)
        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        var q = queries.reshaped(B, L, numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = keys.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        let v = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MoE

class MiniMaxM2SparseMoeBlock: Module {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: MiniMaxM2Configuration) {
        self.numExpertsPerTok = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts
        )
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x.asType(.float32))

        // Sigmoid scoring (not softmax)
        var scores = sigmoid(gates)
        let originalScores = scores
        scores = scores + eScoreCorrectionBias

        // Top-k expert selection
        let k = numExpertsPerTok
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        scores = takeAlong(originalScores, inds, axis: -1)

        // Normalize scores
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        scores = scores.asType(x.dtype)

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - Decoder Layer

class MiniMaxM2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM2Attention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxM2SparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MiniMaxM2Configuration) {
        _selfAttn.wrappedValue = MiniMaxM2Attention(args)
        _blockSparseMoe.wrappedValue = MiniMaxM2SparseMoeBlock(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var hidden = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + blockSparseMoe(postAttentionLayerNorm(hidden))
        return hidden
    }
}

// MARK: - Model

public class MiniMaxM2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxM2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: MiniMaxM2Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MiniMaxM2DecoderLayer(args) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - LLM Model

public class MiniMaxM2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxM2ModelInner
    let configuration: MiniMaxM2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxM2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = MiniMaxM2ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Restructure MoE expert weights: experts.N.wX.weight -> switch_mlp.{gate,up,down}_proj.weight
        if sanitizedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex)"
            for (orig, updated) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).block_sparse_moe.experts.0.\(orig).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        let toJoin = (0 ..< configuration.numLocalExperts).map { expertIndex in
                            sanitizedWeights.removeValue(
                                forKey:
                                    "\(prefix).block_sparse_moe.experts.\(expertIndex).\(orig).\(key)"
                            )!
                        }
                        sanitizedWeights[
                            "\(prefix).block_sparse_moe.switch_mlp.\(updated).\(key)"
                        ] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension MiniMaxM2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
