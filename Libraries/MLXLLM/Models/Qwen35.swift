//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Tunables
// Audited optima on M1 Max (4-bit, summarization). See
// `benchmarks/notes/prefill-step-audit-2026-04-24.md` (local) and issue #80
// for the full sweep. Override per checkpoint via
// `GenerateParameters.prefillStepSize`.
private enum Qwen35Defaults {
    /// Qwen3.5 / 3.6 dense hybrids (0.8B / 2B / 4B / 9B / 27B). 1024-token
    /// chunks beat 2048+ by 2–4% prefill while keeping peak memory bounded.
    static let densePrefillStepSize = 1024

    /// Qwen3.5 35B-A3B and other GDN-MoE variants. Larger chunks amortize
    /// per-chunk dispatch overhead across the 40-layer expert routing —
    /// 4096 wins by 12% over 1024 at ctx=4k.
    static let moePrefillStepSize = 4096
}

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: SSMStateCache? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            // .contiguous() breaks the lazy chain: without it, each decode step
            // builds concat(prev_slice, qkv) -> slice, keeping ALL prior qkv arrays
            // alive in the graph. This causes ~9GB/run memory growth for Qwen3.5-9B.
            cache[0] = convInput[0..., (-(convKernelSize - 1))...].contiguous()
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        // GDN Metal kernel reads q/k/v with raw pointer arithmetic and
        // assumed contiguous strides — slices from MLX.split are not.
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim).contiguous()
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim).contiguous()
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim).contiguous()

        var state = cache?[1]
        var out: MLXArray

        if S == 1, state != nil {
            // Decode (T=1): fused kernel absorbs rmsNorm(q), rmsNorm(k),
            // sigmoid(b) → beta, and g = exp(-exp(aLog) * softplus(a + dtBias))
            // into a single Metal dispatch. Eliminates ~4-6 separate dispatches
            // per GDN layer — dispatch overhead dominates at decode batch size.
            (out, state) = fusedGatedDeltaUpdate(
                qRaw: q,
                kRaw: k,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask
            )
        } else {
            // Prefill (T>1): pre-compute norms as separate MLX ops.
            // The fused kernel's extra register pressure hurts GPU occupancy at
            // larger batch sizes; the ops-based gatedDeltaUpdate is faster there.
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            (out, state) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask
            )
        }

        if let cache {
            cache[1] = state
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }

    /// Fully batched single-step decode against `BatchedMambaCache`. Reads
    /// conv + recurrent state slices for the active prefix, runs the fused
    /// GDN kernel (B>1 capable), and writes back. Mirrors
    /// `Qwen3NextGatedDeltaNet.fullyBatchedForward` but accounts for Qwen3.5's
    /// split QKV/Z/B/A projections.
    public func fullyBatchedForward(
        _ inputs: MLXArray, cache: BatchedMambaCache
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(B == cache.active,
                     "GDN fullyBatchedForward: input B (\(B)) ≠ cache.active (\(cache.active))")

        let qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        // Slice live conv + rec state for the active prefix. These are views.
        let (convStateSlice, recStateSlice) = cache.slice(active: B)

        // No mask in single-step decode (S == 1); skip the where().
        let convInput = concatenated([convStateSlice, qkv], axis: 1)

        // New conv state: trailing (kernel-1) tokens of the combined window.
        // .contiguous() breaks the lazy chain so prior qkv arrays don't leak
        // (matches the rationale in callAsFunction above).
        let newConvState = convInput[0..., (-(convKernelSize - 1))...].contiguous()

        let convOut = silu(conv1d(convInput))
        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        // GDN Metal kernel reads q/k/v with raw pointer arithmetic and
        // assumed contiguous strides — slices from MLX.split are not.
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim).contiguous()
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim).contiguous()
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim).contiguous()

        // Decode (S == 1) with non-nil state — use the fused kernel that
        // absorbs rmsNorm(q), rmsNorm(k), sigmoid(b)→beta, and
        // g = exp(-exp(aLog) * softplus(a + dtBias)) into one Metal dispatch.
        // The kernel is already B>1-capable.
        let (out, newRecState) = fusedGatedDeltaUpdate(
            qRaw: q,
            kRaw: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: recStateSlice,
            mask: nil
        )

        // Commit both pieces of state.
        cache.writeback(conv: newConvState, rec: newRecState)

        let normalized = norm(out, gate: z)
        return outProj(normalized.reshaped(B, S, -1))
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

        return oProj(sigmoidMultiply(output, gate))
    }

    /// Fully batched single-step decode against `BatchedKVCache`. Mirrors
    /// `Qwen3NextAttention.fullyBatchedForward` — same gated-output Q split
    /// and sigmoid-multiplied output projection. Runs the same fast/slow path
    /// branching on whether all active slots share the same offset.
    public func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache, mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // qProj outputs 2x heads (queries + gate). Split before the head
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

        // RoPE + cache update — same fast/slow path branching as Qwen3Next.
        let allSameOffset = cache.offsets[0..<cache.active]
            .allSatisfy { $0 == cache.offsets[0] }

        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
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
            cache.update(newKeys: keys, newValues: values)
        }

        let maxOffset = cache.offsets[0..<cache.active].max() ?? 0
        let allK = cache.keys[..<cache.active, 0..., ..<maxOffset, 0...]
        let allV = cache.values[..<cache.active, 0..., ..<maxOffset, 0...]

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allK, values: allV,
            scale: scale, mask: .array(mask)
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? SSMStateCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// Fully batched decode: dispatches by `isLinear` to the right batched
    /// attention or GDN path. Dense MLP / SparseMoeBlock already conform to
    /// `UnaryLayer` and run over the batched `[B, 1]` tensor without changes.
    func fullyBatchedForward(
        _ x: MLXArray,
        layerCache: BatchedHybridCache.BatchedLayerCache,
        attnMask: MLXArray
    ) -> MLXArray {
        let r: MLXArray
        switch (isLinear, layerCache) {
        case (true, .gdn(let mambaCache)):
            r = linearAttn!.fullyBatchedForward(inputLayerNorm(x), cache: mambaCache)
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

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    /// Whether this checkpoint is eligible for the batched prefill-eval
    /// optimization. Empirically the batched path wins on small dense
    /// hybrids (0.8B, 2B, 4B, 9B — hidden_size 1024 / 2048 / 2560 / 4096)
    /// and is neutral or regressive on larger dense variants (27B,
    /// hidden_size 5120) and MoE variants (35B A3B has hidden_size 2048
    /// but is MoE and excluded by the `numExperts` check).
    fileprivate let batchedPrefillEvalEligible: Bool

    /// Layers per `asyncEval` batch when `batchedPrefillEvalEligible` is
    /// true. Picked from the A/B matrix on M1 Max.
    private static let prefillEvalBatchSize = 8

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1
        self.batchedPrefillEvalEligible = (args.numExperts == 0) && (args.hiddenSize <= 4096)

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? SSMStateCache)

        let modelDtype = hiddenStates.dtype
        let isPrefill = hiddenStates.dim(1) > 1
        // Batched prefill eval is only on for small dense hybrids where
        // it wins on the benchmark (0.8B / 2B / 4B / 9B). Larger dense
        // variants and MoE variants stay on the per-layer path.
        let batchEval = batchedPrefillEvalEligible
        let evalEvery = Self.prefillEvalBatchSize
        var pendingEval: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
            // Force dtype after every layer — quantized operations can promote
            // bf16 -> fp32 inside the lazy graph (invisible to .dtype property).
            // The unconditional asType adds a cast node that ensures downstream
            // GDN/attention kernels receive bf16. When dtype is already correct,
            // asType is a no-op (zero overhead).
            hiddenStates = hiddenStates.asType(modelDtype)

            // During prefill, two paths:
            //
            // - Eligible (small dense hybrids): batch `asyncEval` every
            //   `evalEvery` layers. Each call still splits the lazy graph
            //   and commits a command buffer, so reducing splits lets MLX
            //   fuse kernels across layer boundaries. Only cache inner
            //   states accumulate — intermediate `hiddenStates` references
            //   would keep dead activations alive across the window and
            //   regress prefill, so the current hidden state is added only
            //   at flush time.
            // - Ineligible (27B dense, MoE): fall through to per-layer
            //   `asyncEval`. Alpha's tuned baseline — the batched path
            //   doesn't pay off and sometimes regresses.
            //
            // Skip during decode (T=1) where the graph is tiny and eval overhead hurts tok/s.
            if isPrefill, let c = cacheArray?[i] {
                if batchEval {
                    pendingEval.append(contentsOf: c.innerState())
                    let atBoundary = (i + 1) % evalEvery == 0
                    let atEnd = (i + 1) == layers.count
                    if atBoundary || atEnd {
                        pendingEval.append(hiddenStates)
                        asyncEval(pendingEval)
                        pendingEval.removeAll(keepingCapacity: true)
                    }
                } else {
                    var toEval: [MLXArray] = [hiddenStates]
                    toEval.append(contentsOf: c.innerState())
                    asyncEval(toEval)
                }
            }
        }

        return norm(hiddenStates)
    }

    /// Fully batched single-step decode forward pass. Builds the shared
    /// attention mask once from the first attention layer's `BatchedKVCache`
    /// and reuses it across every attention layer. GDN layers don't take a
    /// mask in single-step decode (S == 1).
    func fullyBatchedForward(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        precondition(caches.layers.count == layers.count,
                     "fullyBatchedForward: cache layer count mismatch")

        var hiddenStates = embedTokens(inputs)

        // Build the shared attention mask from the first attention layer's
        // BatchedKVCache. This is only used for attention layers; GDN passes
        // skip it.
        var sampleAttnCache: BatchedKVCache?
        for layer in caches.layers {
            if case .attention(let c) = layer {
                sampleAttnCache = c
                break
            }
        }

        let attnMask: MLXArray
        if let c = sampleAttnCache {
            let B = c.active
            let cacheDtype = c.keys.dtype
            let allSame = c.offsets[0..<B].allSatisfy { $0 == c.offsets[0] }
            // Post-update max offset (each step advances by 1).
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
            // No attention layers? (Shouldn't happen for Qwen3.5 hybrid, but
            // compose a placeholder so type-checking is straightforward.)
            attnMask = MLXArray.zeros([0, 1, 1, 0], dtype: hiddenStates.dtype)
        }

        let modelDtype = hiddenStates.dtype
        for (i, layer) in layers.enumerated() {
            hiddenStates = layer.fullyBatchedForward(
                hiddenStates, layerCache: caches.layers[i], attnMask: attnMask)
            // Same defensive cast as the per-request path: quantized ops can
            // promote bf16 → fp32 inside the lazy graph.
            hiddenStates = hiddenStates.asType(modelDtype)
        }

        return norm(hiddenStates)
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public var defaultPrefillStepSize: Int {
        configuration.numExperts > 0
            ? Qwen35Defaults.moePrefillStepSize
            : Qwen35Defaults.densePrefillStepSize
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

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Detect turbo from the typed compressionAlgorithm. Other algorithms
        // (.affine, .none) are handled by makeAttentionCache below — affine
        // gets an AffineQuantizedKVCache up-front, .none gets a StandardKVCache.
        let turbo: (keyBits: Int, valueBits: Int)?
        if case let .turbo(kb, vb, _, _) = parameters?.compressionAlgorithm {
            turbo = (kb, vb)
        } else {
            turbo = nil
        }

        // Boundary-skip: leave the first N and last N attention layers
        // uncompressed under turbo (most PPL-sensitive). Shared helper
        // computes the skip set; Qwen 3.5's index space is `model.layers`-
        // based, so attention layers are the non-linear (non-GDN) ones.
        let attentionLayerIndices: [Int] = model.layers.enumerated().compactMap {
            (i, layer) in layer.isLinear ? nil : i
        }
        let skipSet = turboBoundarySkipSet(
            attentionLayerIndices: attentionLayerIndices,
            algorithm: parameters?.compressionAlgorithm)

        return model.layers.enumerated().map { (i, layer) in
            if layer.isLinear {
                return SSMStateCache()
            }
            if let turbo, !skipSet.contains(i) {
                // TurboQuantizedKVCache: Phase 1 stores raw fp16 (zero prefill overhead),
                // Phase 2 compresses and uses compressedAttention (no fp16 copy).
                // Pass headDim so the cache can pre-warm MLX kernel JIT at
                // model load time — without it, the first turbo decode pays
                // ~80ms JIT cost that lands inside TTFT.
                return TurboQuantizedKVCache(
                    bits: max(turbo.keyBits, turbo.valueBits),
                    keyBits: turbo.keyBits, valueBits: turbo.valueBits,
                    maxSize: parameters?.maxKVSize,
                    headDim: configuration.headDim ?? (configuration.hiddenSize / configuration.attentionHeads))
            }
            // Either turbo is off, or this is a boundary attention layer
            // that the user opted to keep uncompressed. Either way, hand
            // off to the standard factory (raw `StandardKVCache` for .none,
            // `AffineQuantizedKVCache` for .affine).
            return makeAttentionCache(
                parameters: parameters,
                maxSize: parameters?.maxKVSize)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasMTPWeights || hasUnsanitizedConv1d

        var weights = weights.filter { !$0.key.contains("mtp.") }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        // Fuse gate_proj + up_proj into gate_up_proj for dense Qwen3NextMLP
        // (both per-layer .mlp and the shared_expert under Qwen35SparseMoeBlock).
        // Tight substrings avoid touching .mlp.switch_mlp.gate_proj on 35B MoE.
        let alreadyFused = weights.keys.contains {
            $0.contains(".mlp.gate_up_proj.") || $0.contains(".shared_expert.gate_up_proj.")
        }
        if !alreadyFused {
            fuseGateUpWeights(&weights, keyFilter: ".mlp.gate_proj.", outputAxis: 0)
            fuseGateUpWeights(&weights, keyFilter: ".shared_expert.gate_proj.", outputAxis: 0)
        }

        return weights
    }

    /// Redirect per-layer-quantization overrides keyed on the pre-fuse weight
    /// paths onto the fused `gate_up_proj` module. Gate and up inside the same
    /// MLP always ship with matching quantization (observed on Unsloth UD-MLX
    /// and Gemma 4 26B A4B mixed 4/8 variants), so merging both entries onto
    /// the fused key is a safe collapse.
    public func sanitize(perLayerQuantization: BaseConfiguration.PerLayerQuantization?)
        -> BaseConfiguration.PerLayerQuantization?
    {
        guard let plq = perLayerQuantization else { return nil }
        var remapped: [String: BaseConfiguration.QuantizationOption] = [:]
        for (key, value) in plq.perLayerQuantization {
            var rewritten = key
            for base in [".mlp", ".shared_expert"] {
                rewritten = rewritten.replacingOccurrences(
                    of: "\(base).gate_proj", with: "\(base).gate_up_proj")
                rewritten = rewritten.replacingOccurrences(
                    of: "\(base).up_proj", with: "\(base).gate_up_proj")
            }
            remapped[rewritten] = value
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: plq.quantization,
            perLayerQuantization: remapped)
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - BatchedHybridLLM (issue #8 — batched decode for Qwen3.5 hybrid)

extension Qwen35TextModel: BatchedHybridLLM {
    /// Fully batched decode: `[B, 1]` tokens → `[B, 1, vocab]` logits.
    /// The bridge dispatches here when it has a `BatchedHybridCache` in hand;
    /// the per-request `iterator.cache` path is left intact for fallback.
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        var out = model.fullyBatchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Build a fresh `BatchedHybridCache` sized for `maxBatch` requests.
    /// Per-layer cache type is decided by `Qwen35DecoderLayer.isLinear`. Both
    /// dense (Qwen3.5) and MoE (Qwen3.6) checkpoints share the same hybrid
    /// layer pattern, so this single layout serves both.
    public func newBatchedHybridCache(
        maxBatch: Int, parameters: GenerateParameters?
    ) -> BatchedHybridCache {
        // Same shape derivation as Qwen35GatedDeltaNet.init(args).
        let cfg = configuration
        let headDim = cfg.headDim ?? (cfg.hiddenSize / cfg.attentionHeads)
        let kernelMinusOne = cfg.linearConvKernelDim - 1
        let keyDim = cfg.linearKeyHeadDim * cfg.linearNumKeyHeads
        let valueDim = cfg.linearValueHeadDim * cfg.linearNumValueHeads
        let convDim = keyDim * 2 + valueDim

        // Sequence budget: prefer parameters.maxKVSize when set, else fall
        // back to 2048. This matches BatchedKVCache.init's default.
        let maxSeq = parameters?.maxKVSize ?? 2048

        let layers: [BatchedHybridCache.BatchedLayerCache] = model.layers.map { layer in
            if layer.isLinear {
                return .gdn(BatchedMambaCache(
                    maxBatch: maxBatch,
                    kernelMinusOne: kernelMinusOne,
                    convDim: convDim,
                    Hv: cfg.linearNumValueHeads,
                    Dv: cfg.linearValueHeadDim,
                    Dk: cfg.linearKeyHeadDim
                ))
            } else {
                return .attention(BatchedKVCache(
                    maxBatch: maxBatch,
                    kvHeads: cfg.kvHeads,
                    headDim: headDim,
                    maxSeq: maxSeq
                ))
            }
        }
        return BatchedHybridCache(layers: layers)
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public var defaultPrefillStepSize: Int { languageModel.defaultPrefillStepSize }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }

    public func sanitize(perLayerQuantization: BaseConfiguration.PerLayerQuantization?)
        -> BaseConfiguration.PerLayerQuantization?
    {
        guard let plq = perLayerQuantization else { return nil }
        // Keep both prefixed AND stripped keys. The quantize() loop uses Swift
        // module paths which include 'language_model.' from @ModuleInfo(key:),
        // so both forms must be present for per-layer lookup to match.
        let prefix = "language_model."
        var merged: [String: BaseConfiguration.QuantizationOption] = plq.perLayerQuantization
        for (key, value) in plq.perLayerQuantization {
            if key.hasPrefix(prefix) {
                merged[String(key.dropFirst(prefix.count))] = value
            }
        }
        return languageModel.sanitize(perLayerQuantization: BaseConfiguration.PerLayerQuantization(
            quantization: plq.quantization, perLayerQuantization: merged))
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - BatchedHybridLLM (top-level)

/// Forward batched-hybrid surface from `Qwen35Model` to its inner
/// `Qwen35TextModel`. `Qwen35MoEModel` inherits this conformance — see
/// `Qwen35MoE.swift` for the MoE-specific notes.
extension Qwen35Model: BatchedHybridLLM {
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: BatchedHybridCache
    ) -> MLXArray {
        languageModel.fullyBatchedDecode(inputs, caches: caches)
    }

    public func newBatchedHybridCache(
        maxBatch: Int, parameters: GenerateParameters?
    ) -> BatchedHybridCache {
        languageModel.newBatchedHybridCache(maxBatch: maxBatch, parameters: parameters)
    }
}
