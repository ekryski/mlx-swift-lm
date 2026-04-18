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

class SwiGLUSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            compiledSwiglu(xUp, xGate),
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return x.squeezed(axis: -2)
    }
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
        var v = vProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        let sinksActive = resolveSinksActive()

        // Quantized cache path
        if let qcache = cache as? QuantizedKVCacheProtocol {
            if sinksActive {
                fatalError("Quantized attention does not support non-zero sinks.")
            }
            q = applyRotaryPosition(rope, to: q, cache: cache)
            k = applyRotaryPosition(rope, to: k, cache: cache)

            let (qKeys, qValues) = qcache.updateQuantized(keys: k, values: v)
            let vHat = quantizedScaledDotProductAttention(
                queries: q,
                quantizedKeys: qKeys,
                quantizedValues: qValues,
                scale: smScale,
                mask: mask,
                groupSize: qcache.groupSize,
                bits: qcache.bits,
                mode: qcache.mode
            )

            return oProj(vHat.swappedAxes(1, 2).reshaped(B, L, -1))
        }

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let vHat = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: smScale,
            mask: mask,
            sinks: sinksActive ? sinks : nil)

        return oProj(vHat.swappedAxes(1, 2).reshaped(B, L, -1))
    }

    // MARK: - D1 pre/post split (for ICB integration)
    //
    // The call sites below decompose `callAsFunction` into three stages:
    //
    //   preAttention(x)                → (q_pre, k_pre, v)    // qProj + kProj + vProj + reshape/swap
    //   attention(q_pre, k_pre, v, ...) → attn_v              // RoPE + cache.update + SDPA
    //   postAttention(attn_v, B, L)    → out                  // oProj + shape fixup
    //
    // RoPE + cache.update + SDPA must stay live because they depend on
    // cache.offset and/or current T_k — both are setBytes-baked and can't
    // be overridden per-step in an ICB replay (see
    // benchmarks/notes/argument-buffers-adoption-plan-2026-04-17.md §E2–E3).
    // The pre- and post- halves are shape-stable across decode steps and
    // are the replayable portion in D1.
    //
    // Equivalence to callAsFunction is enforced in a unit test.

    /// Pre-attention: `x → (q_pre, k_pre, v)`. Runs qProj, kProj, vProj and
    /// their reshape/swap. No cache access, no RoPE, no attention — safe
    /// to capture into an ICB because shape + offset are invariant across
    /// decode steps for a fixed prompt length.
    public func preAttention(_ x: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let D = headDim
        let q = qProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        let k = kProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        let v = vProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        return (q, k, v)
    }

    /// Live attention: applies RoPE, updates cache, runs SDPA. Must stay
    /// outside an ICB — RoPE reads cache.offset and SDPA's setBytes bake
    /// T_k, both of which grow per decode step.
    public func attention(
        qPre: MLXArray,
        kPre: MLXArray,
        v: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let sinksActive = resolveSinksActive()

        if let qcache = cache as? QuantizedKVCacheProtocol {
            if sinksActive {
                fatalError("Quantized attention does not support non-zero sinks.")
            }
            let q = applyRotaryPosition(rope, to: qPre, cache: cache)
            let k = applyRotaryPosition(rope, to: kPre, cache: cache)
            let (qKeys, qValues) = qcache.updateQuantized(keys: k, values: v)
            return quantizedScaledDotProductAttention(
                queries: q,
                quantizedKeys: qKeys,
                quantizedValues: qValues,
                scale: smScale,
                mask: mask,
                groupSize: qcache.groupSize,
                bits: qcache.bits,
                mode: qcache.mode)
        }

        let q = applyRotaryPosition(rope, to: qPre, cache: cache)
        var k = applyRotaryPosition(rope, to: kPre, cache: cache)
        var vCur = v
        if let cache {
            (k, vCur) = cache.update(keys: k, values: vCur)
        }
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: vCur,
            scale: smScale,
            mask: mask,
            sinks: sinksActive ? sinks : nil)
    }

    /// Post-attention: `attn_v → out`. Runs the o_proj linear and
    /// its shape fixup. Shape-stable across decode steps, safe to
    /// capture into an ICB.
    public func postAttention(_ attnV: MLXArray) -> MLXArray {
        let B = attnV.dim(0)
        // attnV comes back as [B, numHeads, L, headDim]; swap back to
        // [B, L, numHeads, headDim] then flatten the last two.
        let L = attnV.dim(2)
        return oProj(attnV.swappedAxes(1, 2).reshaped(B, L, -1))
    }

    private func resolveSinksActive() -> Bool {
        cachedSinksActive
            ?? {
                let active = (sinks * sinks).max().item(Float.self) > 0
                cachedSinksActive = active
                return active
            }()
    }
}

class MLPBlock: Module {
    let hiddenSize: Int
    let numLocalExperts: Int
    let numExpertsPerTok: Int

    @ModuleInfo(key: "experts") var experts: SwiGLUSwitchGLU
    @ModuleInfo(key: "router") var router: Linear

    public init(_ config: GPTOSSConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.numLocalExperts = config.localExperts
        self.numExpertsPerTok = config.expertsPerToken

        _experts.wrappedValue = SwiGLUSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.localExperts,
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

    /// D1 split path: structurally identical to `callAsFunction` but
    /// explicitly factored into (pre-attention, live attention,
    /// post-attention) stages so the pre and post halves can be
    /// captured into ICBs while the attention call stays live.
    ///
    /// This function produces the same output as `callAsFunction` for
    /// identical inputs — a correctness prerequisite before any ICB
    /// wrapping. Used by `GPTOSSModelInner` when D1 mode is enabled.
    public func callAsFunctionSplit(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let pre = preBlock(x)
        let attnV = selfAttn.attention(
            qPre: pre.q, kPre: pre.k, v: pre.v, mask: mask, cache: cache)
        return postBlock(attnV: attnV, residual: pre.residual)
    }

    /// ICB-capturable "pre" half of a decode step:
    /// `x → (residual, q_pre, k_pre, v)`. Covers:
    /// input_layernorm + qProj + kProj + vProj + reshape/swap.
    /// Shape-stable across decode steps.
    public func preBlock(
        _ x: MLXArray
    ) -> (residual: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray) {
        let residual = x
        let xNorm = inputLayerNorm(x)
        let (q, k, v) = selfAttn.preAttention(xNorm)
        return (residual, q, k, v)
    }

    /// ICB-capturable "post" half of a decode step:
    /// `(attn_v, residual) → out`. Covers:
    /// oProj + attn residual + post_attention_layernorm + MLP (MoE) + MLP residual.
    /// Shape-stable across decode steps.
    public func postBlock(
        attnV: MLXArray,
        residual: MLXArray
    ) -> MLXArray {
        let attnOut = selfAttn.postAttention(attnV)
        var h = residual + attnOut
        let residualMLP = h
        h = postAttentionLayerNorm(h)
        h = mlp(h)
        return residualMLP + h
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

    /// D1 decode-path integration state:
    ///   - MLX_ICB_D1 unset or 0: default live path
    ///   - MLX_ICB_D1=1          : split pre/attention/post (still all live),
    ///                             for correctness parity testing
    ///   - MLX_ICB_D1=2          : single-layer ICB POC (layer 0 only),
    ///                             for record/replay mechanism validation
    ///   - MLX_ICB_D1=3          : all-layer ICB (the full D1 target)
    public var d1Mode: Int = {
        let v = ProcessInfo.processInfo.environment["MLX_ICB_D1"]
        return Int(v ?? "0") ?? 0
    }()
    public var useD1SplitPath: Bool { d1Mode >= 1 }
    public var useD1ICB: Bool { d1Mode >= 2 }

    // Per-layer ICB state. Populated on first decode-step record pass;
    // replayed on subsequent steps. Parallel arrays across `layers`.
    private var preICBs: [IndirectCommandBuffer?] = []
    private var postICBs: [IndirectCommandBuffer?] = []

    // Cached scratch buffers for D1 replay — allocated once on first
    // replay, reused across subsequent steps. Without caching, each
    // replay allocates + materializes fresh scratch MLXArrays, and
    // the per-step allocation overhead cancels out the ICB encoding
    // savings on small-coverage POCs.
    //
    // `*Data` arrays are the CONTIGUOUS underlying storage that
    // qProj/kProj/vProj write to; `scratchQ/K/V` are swappedAxes
    // views with the stride layout attention expects.
    private var scratchQData: [MLXArray?] = []
    private var scratchKData: [MLXArray?] = []
    private var scratchVData: [MLXArray?] = []
    private var scratchQ: [MLXArray?] = []
    private var scratchK: [MLXArray?] = []
    private var scratchV: [MLXArray?] = []
    private var scratchH: [MLXArray?] = []

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

            // D1 mode 2: single-layer (layer 0) ICB for mechanism
            //           validation.
            // D1 mode 3: all-layer ICB for full pre-half coverage.
            let icbThisLayer = seqLen == 1 && (
                (d1Mode == 2 && i == 0) || d1Mode == 3)
            if icbThisLayer {
                x = d1Step(x: x, block: layer, layerIndex: i,
                           mask: maskMode, cache: cache[i])
            } else if useD1SplitPath {
                x = layer.callAsFunctionSplit(x, mask: maskMode, cache: cache[i])
            } else {
                x = layer(x, mask: maskMode, cache: cache[i])
            }
        }

        x = norm(x)

        return x
    }

    // MARK: - D1 ICB record/replay helpers

    /// Binding-name tags used by the D1 pre- and post-ICBs.
    private enum D1Tag {
        static let input: BindingName = "d1.input"
        static let qOut: BindingName = "d1.q_out"
        static let kOut: BindingName = "d1.k_out"
        static let vOut: BindingName = "d1.v_out"
        static let attnVIn: BindingName = "d1.attn_v_in"
        static let residualIn: BindingName = "d1.residual_in"
        static let hOut: BindingName = "d1.h_out"
    }

    /// One decode-step iteration for a single layer under D1 (record
    /// on first call, replay on subsequent calls).
    ///
    /// On first call per layer: runs pre + attention + post LIVE to
    /// produce the real result, then runs pre and post a SECOND time
    /// inside record blocks so the ICBs are captured against the
    /// same-shape inputs. Two-pass record is a one-time warmup cost
    /// (step 1 is ~2x the normal live time for this layer).
    ///
    /// On subsequent calls: replay the pre ICB, run live attention,
    /// replay the post ICB. No re-recording.
    private func d1Step(
        x: MLXArray,
        block: GPTOSSTransformerBlock,
        layerIndex: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        // Lazy-init the per-layer ICB + scratch slots on first call.
        if preICBs.count < layers.count {
            preICBs = Array(repeating: nil, count: layers.count)
            postICBs = Array(repeating: nil, count: layers.count)
            scratchQData = Array(repeating: nil, count: layers.count)
            scratchKData = Array(repeating: nil, count: layers.count)
            scratchVData = Array(repeating: nil, count: layers.count)
            scratchQ = Array(repeating: nil, count: layers.count)
            scratchK = Array(repeating: nil, count: layers.count)
            scratchV = Array(repeating: nil, count: layers.count)
            scratchH = Array(repeating: nil, count: layers.count)
        }

        if let icbPre = preICBs[layerIndex], let icbPost = postICBs[layerIndex] {
            return d1Replay(
                x: x, block: block, layerIndex: layerIndex,
                mask: mask, cache: cache,
                icbPre: icbPre, icbPost: icbPost)
        }

        // --- RECORD PATH (first decode step for this layer) ---
        // Live forward first so the caller receives real values.
        let pre = block.preBlock(x)
        eval(pre.q, pre.k, pre.v)
        let attnV = block.selfAttn.attention(
            qPre: pre.q, kPre: pre.k, v: pre.v, mask: mask, cache: cache)
        eval(attnV)
        let hLive = block.postBlock(attnV: attnV, residual: pre.residual)
        eval(hLive)

        // Second: record pre and post into ICBs. Dispatches inside the
        // record closure go to the recorder (no GPU execution), so we
        // don't rely on their outputs — only on the captured commands
        // and the binding tags against the SAME buffers as above.
        do {
            let icbPre = try IndirectCommandBuffer.recordWithBindings { tagger in
                tagger.tag(x, as: D1Tag.input)
                let rec = block.preBlock(x)
                // Force evaluation so mlx emits the actual dispatches
                // into the recorder — without this, the graph stays
                // lazy and nothing is captured.
                eval(rec.q, rec.k, rec.v)
                tagger.tag(rec.q, as: D1Tag.qOut)
                tagger.tag(rec.k, as: D1Tag.kOut)
                tagger.tag(rec.v, as: D1Tag.vOut)
            }
            let icbPost = try IndirectCommandBuffer.recordWithBindings { tagger in
                tagger.tag(attnV, as: D1Tag.attnVIn)
                tagger.tag(pre.residual, as: D1Tag.residualIn)
                let rec = block.postBlock(attnV: attnV, residual: pre.residual)
                eval(rec)
                tagger.tag(rec, as: D1Tag.hOut)
            }
            preICBs[layerIndex] = icbPre
            postICBs[layerIndex] = icbPost
            print(
                "[BENCH] D1 layer \(layerIndex) recorded: "
                    + "pre=\(icbPre.totalCommands) cmds, "
                    + "post=\(icbPost.totalCommands) cmds")
        } catch {
            print("[BENCH] D1 layer \(layerIndex) record failed: \(error) — falling back to live")
        }

        return hLive
    }

    /// Allocate scratch MLXArrays matching the recorded Q/K/V/H shapes
    /// if not already cached for `layerIndex`. Reused across replays
    /// so per-step allocation overhead doesn't cancel the ICB win.
    private func ensureScratchBuffers(
        for x: MLXArray,
        block: GPTOSSTransformerBlock,
        layerIndex: Int
    ) {
        if scratchQ[layerIndex] != nil { return }

        let (B, L) = (x.dim(0), x.dim(1))
        let D = block.selfAttn.headDim
        let nAttn = block.selfAttn.numAttentionHeads
        let nKV = block.selfAttn.numKeyValueHeads
        let H = x.dim(2)

        // Contiguous storage — qProj/kProj/vProj write [B, L, nHeads, D]
        // contiguous layout.
        let qData = MLXArray.zeros([B, L, nAttn, D], dtype: x.dtype)
        let kData = MLXArray.zeros([B, L, nKV, D], dtype: x.dtype)
        let vData = MLXArray.zeros([B, L, nKV, D], dtype: x.dtype)
        let h = MLXArray.zeros([B, L, H], dtype: x.dtype)
        eval(qData, kData, vData, h)

        // Views with the stride layout attention expects. Materialize
        // the views so `buffer().ptr()` is resolved when the C layer
        // extracts it during override.
        let qView = qData.swappedAxes(1, 2)
        let kView = kData.swappedAxes(1, 2)
        let vView = vData.swappedAxes(1, 2)
        eval(qView, kView, vView)

        scratchQData[layerIndex] = qData
        scratchKData[layerIndex] = kData
        scratchVData[layerIndex] = vData
        scratchQ[layerIndex] = qView
        scratchK[layerIndex] = kView
        scratchV[layerIndex] = vView
        scratchH[layerIndex] = h
    }

    private func d1Replay(
        x: MLXArray,
        block: GPTOSSTransformerBlock,
        layerIndex: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        icbPre: IndirectCommandBuffer,
        icbPost: IndirectCommandBuffer
    ) -> MLXArray {
        // Cached scratch buffers are allocated on first replay for this
        // layer and reused across subsequent replays. Shape is stable
        // across decode steps (L=1 for decode), so a single allocation
        // amortizes per-step overhead.
        ensureScratchBuffers(
            for: x, block: block, layerIndex: layerIndex)

        let scratchQData = self.scratchQData[layerIndex]!
        let scratchKData = self.scratchKData[layerIndex]!
        let scratchVData = self.scratchVData[layerIndex]!
        let scratchQ = self.scratchQ[layerIndex]!
        let scratchK = self.scratchK[layerIndex]!
        let scratchV = self.scratchV[layerIndex]!
        let scratchH = self.scratchH[layerIndex]!
        _ = (scratchQData, scratchKData, scratchVData)  // silence unused

        // Schedule materialization of `x` without a CPU-side wait
        // (eval would sync and kill tok/s when replayed 24× per step).
        // The barrier-tracking fix in
        // CommandEncoder::replay_icb_with_overrides ensures GPU
        // ordering with subsequent dispatches.
        asyncEval(x)

        // ISOLATION MODE: MLX_ICB_D1_PART:
        //   "pre"  → replay pre, live attention, LIVE post
        //   "post" → live pre, live attention, replay post
        //   "both" (default) → full D1 replay
        // Set via env var; tests whether pre or post (or both) are
        // causing the "!!!!" garbage output.
        let part = ProcessInfo.processInfo.environment["MLX_ICB_D1_PART"] ?? "both"

        if part == "pre" || part == "both" {
            icbPre.replay(overrides: [
                D1Tag.input: x,
                D1Tag.qOut: scratchQ,
                D1Tag.kOut: scratchK,
                D1Tag.vOut: scratchV,
            ])
            // No explicit synchronize: the mlx-side barrier-tracking
            // patch (`replay_icb_with_overrides` registers override
            // buffers in `next_outputs_`) ensures the subsequent live
            // attention call gets a memoryBarrier before reading
            // scratchQ/K/V.
        }

        let qPre: MLXArray
        let kPre: MLXArray
        let vPre: MLXArray
        if part == "pre" || part == "both" {
            qPre = scratchQ
            kPre = scratchK
            vPre = scratchV
        } else {
            let pre = block.preBlock(x)
            qPre = pre.q
            kPre = pre.k
            vPre = pre.v
        }

        let attnV = block.selfAttn.attention(
            qPre: qPre, kPre: kPre, v: vPre, mask: mask, cache: cache)
        eval(attnV)

        if part == "post" || part == "both" {
            icbPost.replay(overrides: [
                D1Tag.attnVIn: attnV,
                D1Tag.residualIn: x,
                D1Tag.hOut: scratchH,
            ])
            return scratchH
        } else {
            return block.postBlock(attnV: attnV, residual: x)
        }
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

        if weights.keys.contains(where: { $0.contains("gate_proj.weight") }) {
            return weights
        }

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

        var finalWeights: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("gate_up_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "gate_proj.weight")
                ] = contiguous(v[.ellipsis, .stride(by: 2), 0...])
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "up_proj.weight")
                ] = contiguous(v[.ellipsis, .stride(from: 1, by: 2), 0...])
            } else if k.contains("down_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj", with: "down_proj.weight")
                ] = contiguous(v)
            } else if k.contains("gate_up_proj_bias") {
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "gate_proj.bias")
                ] = contiguous(v[.ellipsis, .stride(by: 2)])
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "up_proj.bias")
                ] = contiguous(v[.ellipsis, .stride(from: 1, by: 2)])
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

    /// Pure attention model — use larger prefill chunks (2048+) since there's no
    /// GatedDeltaNet sequential bottleneck. Reduces TTFT by processing more tokens per step.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = max(windowSize ?? 512, 2048)
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

        for lt in model.layerTypes {
            if lt == "full_attention" {
                if let maxKVSize = parameters?.maxKVSize {
                    // Bound full-attention KV cache to prevent unbounded growth
                    // in multi-turn sessions. keep: 4 preserves attention sink tokens.
                    caches.append(RotatingKVCache(maxSize: maxKVSize, keep: 4))
                } else {
                    caches.append(StandardKVCache())
                }
            } else {
                caches.append(
                    RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
                )
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
