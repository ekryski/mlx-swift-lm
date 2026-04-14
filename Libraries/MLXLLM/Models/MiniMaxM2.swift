// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NativePrefillBridge

// MARK: - Generic Prefill Bridge (SPM C++ target — shared allocator)

final class GenericPrefillBridge {
    static let shared = GenericPrefillBridge()

    private var initializedModelType: String? = nil

    func ensureInitialized(modelType: String, model: Module, config configJSON: String) -> Bool {
        FileHandle.standardError.write(Data("[GenericPrefill] ensureInitialized: requested=\(modelType) cached=\(String(describing: initializedModelType))\n".utf8))
        if initializedModelType == modelType { return true }

        // If switching model types, cleanup first
        if initializedModelType != nil {
            gp_cleanup()
            initializedModelType = nil
        }

        let rc = configJSON.withCString { gp_init($0) }
        if rc != 0 { print("[GenericPrefill] init failed"); return false }

        let params = model.parameters().flattened()
        var weightCount = 0
        for (key, arr) in params {
            let bridgeKey = "model." + key
            let rawPtr = arr.ctx.ctx
            let status = bridgeKey.withCString { cKey in
                gp_set_weight(cKey, rawPtr!)
            }
            if status == 0 { weightCount += 1 }
        }
        print("[GenericPrefill] Passed \(weightCount) weights")

        let finRC = gp_finalize()
        if finRC != 0 { print("[GenericPrefill] finalize failed"); return false }

        initializedModelType = modelType
        print("[GenericPrefill] Initialized for \(modelType)")

        // Warmup: run a dummy forward pass to materialize all weight GPU buffers.
        // Without this, the first real forward pass produces garbage because the
        // Metal allocator reclaims unevaluated weight buffers during graph construction.
        var warmMs: Double = 0
        let warmTokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let _ = gp_run(warmTokens.ctx.ctx!, &warmMs)
        print(String(format: "[GenericPrefill] Pre-warmed in %.0fms", warmMs))

        return true
    }

    // Convenience for MiniMax
    func ensureInitialized(model: MiniMaxM2ModelInner, config: MiniMaxM2Configuration) -> Bool {
        let json = """
        {"model_type":"minimax_m2","hidden_size":\(config.hiddenSize),"num_hidden_layers":\(config.hiddenLayers),"num_attention_heads":\(config.attentionHeads),"num_key_value_heads":\(config.kvHeads),"head_dim":\(config.headDim),"intermediate_size":\(config.intermediateSize),"vocab_size":\(config.vocabularySize),"rms_norm_eps":\(String(format:"%.0e",Double(config.rmsNormEps))),"rope_theta":\(String(format:"%.0f",Double(config.ropeTheta))),"rotary_dim":\(config.rotaryDim),"tie_word_embeddings":\(config.tieWordEmbeddings),"use_qk_norm":\(config.useQkNorm),"num_local_experts":\(config.numLocalExperts),"num_experts_per_tok":\(config.numExpertsPerTok),"scoring_func":"\(config.scoringFunc)"}
        """
        return ensureInitialized(modelType: "minimax_m2", model: model, config: json)
    }

    func runAndInjectKV(tokenArray: MLXArray, cache: [KVCache], numLayers: Int) -> (Double, Bool) {
        guard initializedModelType != nil else { return (0, false) }

        let tokens2d = tokenArray.dim(0) == 1 ? tokenArray : tokenArray.reshaped(1, tokenArray.size)
        var ms: Double = 0
        let rc = gp_run(tokens2d.ctx.ctx!, &ms)
        if rc != 0 { return (0, false) }

        for i in 0..<min(numLayers, cache.count) {
            guard let kPtr = gp_get_k_ptr(Int32(i)),
                  let vPtr = gp_get_v_ptr(Int32(i)) else {
                return (ms, false)
            }
            let kArr = MLXArray.fromCppArray(kPtr).contiguous()
            let vArr = MLXArray.fromCppArray(vPtr).contiguous()

            let _ = cache[i].update(keys: kArr, values: vArr)
        }

        return (ms, true)
    }
}

// MARK: - Configuration

public struct MiniMaxM2Configuration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let maxPositionEmbeddings: Int
    let numExpertsPerTok: Int
    let numLocalExperts: Int
    let sharedIntermediateSize: Int
    let hiddenLayers: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let rotaryDim: Int
    let vocabularySize: Int
    let tieWordEmbeddings: Bool
    let scoringFunc: String
    let headDim: Int
    let useQkNorm: Bool

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
        modelType = try container.decode(String.self, forKey: .modelType)
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
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let useQkNorm: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ config: MiniMaxM2Configuration) {
        let dim = config.hiddenSize
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.useQkNorm = config.useQkNorm

        _qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        if useQkNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim * numHeads, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim * numKVHeads, eps: config.rmsNormEps)
        }

        rope = RoPE(dimensions: config.rotaryDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // QK norm applied BEFORE reshape (per-layer norm)
        if useQkNorm {
            queries = qNorm!(queries)
            keys = kNorm!(keys)
        }

        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask)

        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MoE

class MiniMaxM2SparseMoeBlock: Module {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ config: MiniMaxM2Configuration) {
        self.numExpertsPerTok = config.numExpertsPerTok
        _gate.wrappedValue = Linear(config.hiddenSize, config.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.numLocalExperts
        )
        _correctionBias.wrappedValue = MLXArray.zeros([config.numLocalExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x.asType(.float32))

        // Sigmoid scoring (not softmax)
        let scores = sigmoid(gates)
        let origScores = scores
        let correctedScores = scores + correctionBias

        // Top-k expert selection
        let k = numExpertsPerTok
        let inds = argPartition(-correctedScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var selectedScores = takeAlong(origScores, inds, axis: -1)

        // Normalize scores
        selectedScores = selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
        selectedScores = selectedScores.asType(x.dtype)

        var y = switchMLP(x, inds)
        y = (y * selectedScores[.ellipsis, .newAxis]).sum(axis: -2)

        return y
    }
}

// MARK: - Decoder Layer

class MiniMaxM2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM2Attention
    @ModuleInfo(key: "block_sparse_moe") var moe: MiniMaxM2SparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiniMaxM2Configuration) {
        _selfAttn.wrappedValue = MiniMaxM2Attention(config)
        _moe.wrappedValue = MiniMaxM2SparseMoeBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: KVCache? = nil
    ) -> MLXArray {
        let r = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return r + moe(postAttentionLayerNorm(r))
    }
}

// MARK: - Model

class MiniMaxM2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxM2DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: MiniMaxM2Configuration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0..<config.hiddenLayers).map { _ in MiniMaxM2DecoderLayer(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = makeAttentionMask(n: h.dim(1), cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - LLM Model

public class MiniMaxM2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var loraLayers: [Module] { model.layers.map { $0 } }
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo var model: MiniMaxM2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let config: MiniMaxM2Configuration

    public init(_ config: MiniMaxM2Configuration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.model = MiniMaxM2ModelInner(config)
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        var y = input.text
        // Native prefill offload (opt-in via NATIVE_PREFILL=1)
        if ProcessInfo.processInfo.environment["NATIVE_PREFILL"] == "1" {
            let bridge = GenericPrefillBridge.shared
            if bridge.ensureInitialized(model: model, config: config) {
                let allTokens = input.text.tokens
                let prefillCount = allTokens.size - 1
                if prefillCount > 0 {
                    let tokenSlice = allTokens[0 ..< prefillCount].reshaped(-1)
                    let (ms, ok) = bridge.runAndInjectKV(
                        tokenArray: tokenSlice, cache: cache, numLayers: config.hiddenLayers)
                    if ok {
                        print(String(format: "[GenericPrefill] %d tokens in %.1fms (%.0f t/s)",
                            prefillCount, ms, Double(prefillCount) / (ms / 1000)))
                        let lastToken = allTokens[prefillCount ..< allTokens.size]
                        return .tokens(LMInput.Text(tokens: lastToken))
                    }
                }
            }
        }

        // Default Swift prefill
        let prefillStepSize = max(windowSize ?? 512, 4096)
        while y.tokens.size > 1 {
            let chunkSize = min(prefillStepSize, y.tokens.size - 1)
            let input = y[.newAxis, ..<chunkSize]
            _ = self(input.tokens, cache: cache.isEmpty ? nil : cache)
            var cacheArrays: [MLXArray] = []
            for c in cache {
                cacheArrays.append(contentsOf: c.innerState())
            }
            asyncEval(cacheArrays)
            y = y[chunkSize...]
        }
        MLX.Memory.clearCache()

        return .tokens(y)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else {
            out = lmHead!(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Restructure MoE expert weights: experts.N.wX.weight → switch_mlp.{gate,up,down}_proj.weight
        if processedWeights["model.layers.0.block_sparse_moe.experts.0.w1.weight"] != nil {
            let mapping = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]
            for l in 0..<config.hiddenLayers {
                let prefix = "model.layers.\(l)"
                for (origName, newName) in mapping {
                    let firstKey = "\(prefix).block_sparse_moe.experts.0.\(origName).weight"
                    if processedWeights[firstKey] != nil {
                        var toJoin: [MLXArray] = []
                        for e in 0..<config.numLocalExperts {
                            let key = "\(prefix).block_sparse_moe.experts.\(e).\(origName).weight"
                            if let w = processedWeights.removeValue(forKey: key) {
                                toJoin.append(w)
                            }
                        }
                        if !toJoin.isEmpty {
                            processedWeights["\(prefix).block_sparse_moe.switch_mlp.\(newName).weight"] = stacked(toJoin)
                        }
                    }
                }
            }
        }

        return processedWeights
    }
}
