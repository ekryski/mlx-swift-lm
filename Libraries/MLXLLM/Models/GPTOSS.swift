//
//  GPTOSS.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/8/6.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gpt_oss.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Tunables
// Audited optima on M1 Max (4-bit, summarization). See issue #80 for the
// full sweep.
private enum GPTOSSDefaults {
    /// GPT-OSS-20B prefill chunk size. 2048 is the audited optimum;
    /// going larger (4096+) regresses prefill at long contexts due to
    /// activation memory pressure during the chunked forward.
    static let prefillStepSize = 2048
}

// MARK: - Configuration

public struct GPTOSSConfiguration: Codable, Sendable {
    public var modelType: String = "gpt_oss"
    public var hiddenLayers: Int = 36
    public var localExperts: Int = 128
    public var expertsPerToken: Int = 4
    public var vocabularySize: Int = 201088
    public var rmsNormEps: Float = 1e-5
    public var hiddenSize: Int = 2880
    public var intermediateSize: Int = 2880
    public var headDim: Int = 64
    public var attentionHeads: Int = 64
    public var kvHeads: Int = 8
    public var slidingWindow: Int = 128
    public var ropeTheta: Float = 150000
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var layerTypes: [String]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.localExperts = try container.decode(Int.self, forKey: .localExperts)
        self.expertsPerToken = try container.decode(Int.self, forKey: .expertsPerToken)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 150000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
    }
}

private func mlxTopK(_ a: MLXArray, k: Int, axis: Int = -1) -> (values: MLXArray, indices: MLXArray)
{
    let partitionedIndices = argPartition(a, kth: -k, axis: axis)
    let topKIndices = partitionedIndices[.ellipsis, (-k)...]
    let topKValues = takeAlong(a, topKIndices, axis: axis)
    return (topKValues, topKIndices)
}

private func swiglu(_ xLinear: MLXArray, _ xGlu: MLXArray, alpha: Float = 1.702, limit: Float = 7.0)
    -> MLXArray
{
    var xLinear = xLinear
    var xGlu = xGlu
    xGlu = clip(xGlu, max: MLXArray(limit))
    xLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

    let gluScaled = alpha * xGlu
    let sig = sigmoid(gluScaled)

    let outGlu = xGlu * sig
    return outGlu * (xLinear + 1)
}

private let compiledSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { xLinear, xGlu in
    swiglu(xLinear, xGlu)
}

class AttentionBlock: Module {
    let headDim: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numKeyValueGroups: Int
    let smScale: Float

    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: YarnRoPE
    private var cachedSinksActive: Bool?

    public init(_ config: GPTOSSConfiguration) {
        self.headDim = config.headDim
        self.numAttentionHeads = config.attentionHeads
        self.numKeyValueHeads = config.kvHeads
        self.numKeyValueGroups = config.attentionHeads / config.kvHeads

        _sinks.wrappedValue = zeros([config.attentionHeads])
        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: true)
        _oProj.wrappedValue = Linear(
            config.headDim * config.attentionHeads, config.hiddenSize, bias: true)

        self.smScale = 1.0 / sqrt(Float(config.headDim))

        if let ropeScaling = config.ropeScaling {
            self.rope = YarnRoPE(
                dimensions: headDim,
                base: config.ropeTheta,
                scalingFactor: ropeScaling["factor"]?.asFloat() ?? 32.0,
                originalMaxPositionEmbeddings: ropeScaling["original_max_position_embeddings"]?
                    .asInt() ?? 4096,
                betaFast: ropeScaling["beta_fast"]?.asFloat() ?? 32.0,
                betaSlow: ropeScaling["beta_slow"]?.asFloat() ?? 1.0
            )
        } else {
            self.rope = YarnRoPE(
                dimensions: headDim,
                base: config.ropeTheta
            )
        }
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let D = headDim

        var q = qProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        var k = kProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        let v = vProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)

        let sinksActive =
            cachedSinksActive
            ?? {
                let active = (sinks * sinks).max().item(Float.self) > 0
                cachedSinksActive = active
                return active
            }()

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        let vHat = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache,
            scale: smScale,
            mask: mask,
            sinks: sinksActive ? sinks : nil
        )

        return oProj(vHat.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

class MLPBlock: Module {
    let hiddenSize: Int
    let numLocalExperts: Int
    let numExpertsPerTok: Int

    @ModuleInfo(key: "experts") var experts: FusedGateUpSwitchGLU
    @ModuleInfo(key: "router") var router: Linear

    public init(_ config: GPTOSSConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.numLocalExperts = config.localExperts
        self.numExpertsPerTok = config.expertsPerToken

        _experts.wrappedValue = FusedGateUpSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.localExperts,
            twoArgActivation: compiledSwiglu,
            activationKind: .clippedSwiglu,
            bias: true
        )
        _router.wrappedValue = Linear(config.hiddenSize, config.localExperts, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = router(x)
        let (experts, indices) = mlxTopK(g, k: numExpertsPerTok, axis: -1)
        let stopIndices = MLX.stopGradient(indices)
        let expertWeights = softmax(experts, axis: -1, precise: true)

        var x = self.experts(x, stopIndices)

        x = x * expandedDimensions(expertWeights, axis: -1)
        return x.sum(axis: -2)
    }
}

class GPTOSSTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AttentionBlock
    @ModuleInfo(key: "mlp") var mlp: MLPBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ config: GPTOSSConfiguration) {
        _selfAttn.wrappedValue = AttentionBlock(config)
        _mlp.wrappedValue = MLPBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        var residual = x
        var x = inputLayerNorm(x)
        x = selfAttn(x, mask: mask, cache: cache)
        x = residual + x

        residual = x
        x = postAttentionLayerNorm(x)
        x = mlp(x)
        x = residual + x
        return x
    }
}

public class GPTOSSModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layerTypes: [String]
    fileprivate let layers: [GPTOSSTransformerBlock]
    let windowSize: Int
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int

    public init(_ config: GPTOSSConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layerTypes =
            config.layerTypes
            ?? Array(
                repeating: [
                    "sliding_attention",
                    "full_attention",
                ], count: config.hiddenLayers / 2
            ).flatMap { $0 }
        self.layers = (0 ..< config.hiddenLayers).map { _ in GPTOSSTransformerBlock(config) }
        self.windowSize = config.slidingWindow
        self.slidingAttentionIndex =
            self.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.fullAttentionIndex =
            self.layerTypes.firstIndex(of: "full_attention") ?? 0
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var x: MLXArray
        if let inputEmbeddings {
            x = inputEmbeddings
        } else {
            x = embedTokens(inputs)
        }

        let cache: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)

        let seqLen = x.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

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
                        windowSize: windowSize
                    )
                }
                maskMode = slidingMask!
            }

            x = layer(x, mask: maskMode, cache: cache[i])
        }

        x = norm(x)

        return x
    }
}

private func convertMoePackedTensors(blocks: MLXArray, scales: MLXArray) -> MLXArray {
    precondition(
        blocks.shape.dropLast() == scales.shape,
        "blocks.shape=\(blocks.shape) does not match scales.shape=\(scales.shape)"
    )

    var scales = scales.asType(.int32) - 127
    let lut = MLXArray([
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]).asType(.bfloat16)

    let (prefixShape, G, B) = (Array(blocks.shape.dropLast(2)), blocks.dim(-2), blocks.dim(-1))

    let blocks = blocks.reshaped(-1, B)
    scales = scales.reshaped(-1, 1)

    let idxLo = blocks & 0x0F
    let idxHi = blocks >> 4

    var out = stacked([lut[idxLo], lut[idxHi]], axis: -1).flattened(start: -2)
    out = (2.0 ** scales) * out
    out = out.reshaped(prefixShape.count, G * B * 2)
    return out.asType(.bfloat16)
}

public class GPTOSSModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: GPTOSSModelInner
    private let configuration: GPTOSSConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    /// Coherent on `--kv turbo*` via the bias-correction + hybrid
    /// sliding-FP16 policy in `newCache(...)` below. Closes #171 / #130.
    public var supportsTurboQuantization: Bool { true }

    public init(_ config: GPTOSSConfiguration) {
        self.configuration = config
        self.modelType = config.modelType
        self.model = GPTOSSModelInner(config)
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        // Already-quantized repos ship separate gate_proj / up_proj tensors.
        // Fuse them into the single gate_up_proj that FusedGateUpSwitchGLU expects.
        if weights.keys.contains(where: { $0.contains(".experts.gate_proj.weight") }) {
            return fuseGateUpWeights(weights)
        }

        // Packed (blocks + scales) format — unpack to raw tensors first.
        if weights.keys.contains(where: { $0.contains("gate_up_proj_scales") }) {
            var newWeights: [String: MLXArray] = [:]
            for (k, v) in weights {
                if k.hasSuffix("_scales") {
                    continue
                } else if k.hasSuffix("_blocks") {
                    let scaleKey = k.replacingOccurrences(of: "_blocks", with: "_scales")
                    if let scales = weights[scaleKey] {
                        let newV = convertMoePackedTensors(blocks: v, scales: scales)
                        let newK = k.replacingOccurrences(of: "_blocks", with: "")
                        newWeights[newK] = newV
                    }
                } else {
                    newWeights[k] = v
                }
            }
            weights = newWeights
        }

        // De-interleave the shipped `gate_up_proj` (stride-2 interleaved gate/up)
        // into a concatenated [gate; up] layout along the output axis, matching
        // what `MLX.split(gateUp, parts: 2, axis: -1)` expects in
        // FusedGateUpSwitchGLU.
        var finalWeights: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("gate_up_proj"), !k.contains("bias") {
                let gate = v[.ellipsis, .stride(by: 2), 0...]
                let up = v[.ellipsis, .stride(from: 1, by: 2), 0...]
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "gate_up_proj.weight")
                ] = contiguous(concatenated([gate, up], axis: -2))
            } else if k.contains("down_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj", with: "down_proj.weight")
                ] = contiguous(v)
            } else if k.contains("gate_up_proj_bias") {
                let gateBias = v[.ellipsis, .stride(by: 2)]
                let upBias = v[.ellipsis, .stride(from: 1, by: 2)]
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "gate_up_proj.bias")
                ] = contiguous(concatenated([gateBias, upBias], axis: -1))
            } else if k.contains("down_proj_bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj_bias", with: "down_proj.bias")
                ] = contiguous(v)
            } else {
                finalWeights[k] = v
            }
        }

        return finalWeights
    }

    /// Fuse pre-quantized checkpoints' separate `.experts.gate_proj.*` and
    /// `.experts.up_proj.*` tensors into a single `.experts.gate_up_proj.*` set
    /// (weight, scales, biases) on the output axis. Called when the checkpoint
    /// already has `gate_proj.weight` keys — i.e. skipped the interleaved
    /// `gate_up_proj` packing path above.
    private func fuseGateUpWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        var gateTensors: [String: MLXArray] = [:]
        var upTensors: [String: MLXArray] = [:]

        for (k, v) in weights {
            if k.contains(".experts.gate_proj.") {
                gateTensors[k] = v
            } else if k.contains(".experts.up_proj.") {
                upTensors[k] = v
            } else {
                result[k] = v
            }
        }

        for (gateKey, gateVal) in gateTensors {
            let upKey = gateKey.replacingOccurrences(of: "gate_proj", with: "up_proj")
            guard let upVal = upTensors[upKey] else {
                // Unpaired gate tensor — leave as-is rather than silently drop.
                result[gateKey] = gateVal
                continue
            }
            let fusedKey = gateKey.replacingOccurrences(of: "gate_proj", with: "gate_up_proj")
            // weight: [E, outDim, packedIn] → [E, 2*outDim, packedIn]
            // scales/biases: [E, outDim, groups] → [E, 2*outDim, groups]
            // bias: [E, outDim] → [E, 2*outDim]
            result[fusedKey] = concatenated([gateVal, upVal], axis: 1)
        }

        return result
    }

    /// Per-model audited prefill chunk default.
    public var defaultPrefillStepSize: Int { GPTOSSDefaults.prefillStepSize }

    /// Pure attention model — uses larger prefill chunks (2048) since there's no
    /// GatedDeltaNet sequential bottleneck. Reduces TTFT by processing more
    /// tokens per step.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? defaultPrefillStepSize
        var y = input.text

        while y.tokens.size > 1 {
            let chunkSize = min(prefillStepSize, y.tokens.size - 1)
            let input = y[.newAxis, ..<chunkSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            y = y[chunkSize...]
            MLX.Memory.clearCache()
        }

        return .tokens(y)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [KVCache] = []
        let prefillStep = defaultPrefillStepSize
        // GPT-OSS turbo* policy (spec 041 phase 1.1 follow-up):
        // - Full-attention layers: TurboQuant with DC-bias correction
        //   (`useBias: true`) — model's K/V projections have `bias=True`,
        //   so the rotated vectors carry a structured DC offset the
        //   zero-mean Lloyd-Max codebook can't represent.
        // - Sliding-window layers (128-token cap): raw FP16 — the codec's
        //   per-token error compounds in the sinks softmax; the FP16
        //   fallback adds ~1.5 MB across all sliding layers vs the
        //   full-attention layers' 8K-token compressed caches.
        //
        // `--kv affine*` and `--kv none` keep both layer types on their
        // default cache (no silent algorithm substitution).
        let isTurboScheme: Bool
        if case .turbo = parameters?.compressionAlgorithm {
            isTurboScheme = true
        } else {
            isTurboScheme = false
        }
        for lt in model.layerTypes {
            if lt == "full_attention" {
                // keep: 4 preserves attention-sink tokens for full-attention layers.
                // No `slidingWindow` — function reads the user's
                // `parameters?.maxKVSize` internally as the budget cap.
                caches.append(makeAttentionCache(
                    parameters: parameters,
                    keep: 4,
                    prefillStep: prefillStep,
                    useBias: isTurboScheme))
            } else if isTurboScheme {
                caches.append(StandardKVCache(
                    maxSize: configuration.slidingWindow, keep: 0))
            } else {
                caches.append(makeAttentionCache(
                    parameters: parameters,
                    slidingWindow: configuration.slidingWindow,
                    keep: 0,
                    prefillStep: prefillStep))
            }
        }
        return caches
    }
}

extension GPTOSSModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
