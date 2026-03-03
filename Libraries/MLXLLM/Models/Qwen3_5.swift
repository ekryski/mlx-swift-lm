//
//  Qwen3_5.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//  Supports both dense (qwen3_5) and MoE (qwen3_5_moe) model types.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - GDN Kernel Diagnostics

/// One-time diagnostic logging for Gated DeltaNet Metal kernel dispatch.
/// Logs whether the optimized Metal kernel or sequential ops fallback is used.
private enum GDNKernelDiagnostics {
    private static var hasLoggedKernel = false
    private static var hasLoggedFallback = false

    static func logKernelUsed(dk: Int, dv: Int) {
        guard !hasLoggedKernel else { return }
        hasLoggedKernel = true
        print("[GDN] Metal kernel active: Dk=\(dk) Dv=\(dv) (Dk%32==0, grid=(32,Dv,B*Hv), threadGroup=(32,4,1))")
    }

    static func logFallback(dk: Int, gNdim: Int, t: Int) {
        guard !hasLoggedFallback else { return }
        hasLoggedFallback = true
        print("[GDN] WARNING: Metal kernel unavailable, using sequential ops fallback. Dk=\(dk) (Dk%32=\(dk%32)), g.ndim=\(gNdim), T=\(t)")
    }
}

// MARK: - Gated Delta Helpers

private func sigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    x * sigmoid(gate)
}

private func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay.asType(aLog.dtype)
}

// MARK: - Gated Delta Metal Kernels

/// Metal kernel for Gated Delta Net — processes all timesteps in a single GPU dispatch.
/// Matches the Python mlx-lm reference implementation's Metal kernel from gated_delta.py.
///
/// Grid: (32, Dv, B*Hv), ThreadGroup: (32, 4, 1)
/// Each SIMD group of 32 threads handles Dk/32 key-dim elements and collaborates via simd_sum
/// for dot products along the key dimension. State is held in per-thread float32 registers
/// for numerical stability (matching Python's approach).
///
/// Handles scalar gating (g shape [B, T, Hv]). Vectorized gating ([B, T, Hv, Dk]) falls back
/// to the ops-based sequential loop.
private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskCondition = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        if (dv_idx >= Dv) return;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // Load state into float32 registers for numerical stability
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv] — scalar gating (one decay value per head)
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
            if (\(maskCondition)) {
                // Step 1: Decay state and compute key-value memory (dot product along Dk)
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * static_cast<float>(g_[hv_idx]);
                    kv_mem += state[i] * static_cast<float>(k_[s_idx]);
                }
                kv_mem = simd_sum(kv_mem);

                // Step 2: Compute delta (residual between value and memory projection)
                auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                             * static_cast<float>(beta_[hv_idx]);

                // Step 3: Update state with rank-1 delta and compute output
                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
                    out += state[i] * static_cast<float>(q_[s_idx]);
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                    y[dv_idx] = static_cast<InT>(out);
                }
            }

            // Advance pointers to next timestep (regardless of mask)
            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            g_ += Hv;
            beta_ += Hv;
        }

        // Write final state back (always, even if all timesteps were masked)
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in"]
    if hasMask {
        inputNames.append("mask")
    }

    return MLXFast.metalKernel(
        name: hasMask ? "gated_delta_masked" : "gated_delta",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let maskedKernel: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        maskedKernel = makeGatedDeltaKernel(hasMask: true)
    }
}

/// Dispatch GDN computation to Metal kernel. Returns nil if kernel is unavailable
/// or dimensions are unsupported (falls back to ops-based loop).
private func gatedDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray, mask: MLXArray? = nil
) -> (MLXArray, MLXArray)? {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype

    // Kernel requires Dk divisible by 32 (SIMD group width on Apple Silicon)
    guard Dk % 32 == 0 else { return nil }

    // Only scalar gating supported (g shape [B, T, Hv], not vectorized [B, T, Hv, Dk])
    guard g.ndim == 3 else { return nil }

    // T=0 edge case: no timesteps to process
    guard T > 0 else { return nil }

    let manager = GatedDeltaKernelManager.shared

    if let mask {
        guard let kernel = manager.maskedKernel else { return nil }
        let maskArray = mask.asType(inputType)
        let outputs = kernel(
            [q, k, v, g, beta, state, maskArray],
            template: [
                ("InT", inputType),
                ("T", T),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape],
            outputDTypes: [inputType, inputType],
            initValue: 0.0
        )
        return (outputs[0], outputs[1])
    } else {
        guard let kernel = manager.kernel else { return nil }
        let outputs = kernel(
            [q, k, v, g, beta, state],
            template: [
                ("InT", inputType),
                ("T", T),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape],
            outputDTypes: [inputType, inputType]
        )
        return (outputs[0], outputs[1])
    }
}

// MARK: - Gated Delta Ops (Sequential Fallback)

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y, state)
}

private func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    // Qwen3.5 config specifies mamba_ssm_dtype: "float32" — use float32 for
    // GatedDeltaNet state to maintain numerical stability in recurrent updates.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT, k: kT, v: vT, g: gT, beta: betaT,
            state: state, mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

private func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    // Try Metal kernel for GPU-accelerated processing (all timesteps in one dispatch).
    // Falls back to ops-based sequential loop if kernel unavailable or dimensions unsupported.
    if let result = gatedDeltaKernel(
        q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask
    ) {
        GDNKernelDiagnostics.logKernelUsed(dk: q.dim(3), dv: v.dim(3))
        return result
    }

    GDNKernelDiagnostics.logFallback(dk: q.dim(3), gNdim: g.ndim, t: q.dim(1))
    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

// MARK: - Model Components

private final class Qwen3_5RMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        if let gate {
            x = x * silu(gate)
        }
        return x
    }
}

private final class Qwen3_5Attention: Module {
    let scale: Float
    let attentionHeads: Int
    let kvHeads: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: OffsetLayer

    init(_ args: Qwen3_5TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.scale = pow(Float(headDim), -0.5)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads

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

        return oProj(sigmoidMultiply(output, gate))
    }
}

private final class Qwen3_5MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Gated Delta Net with separate projections for Q/K/V, Z, B, A.
/// This is the key difference from Qwen3Next which uses fused projections.
private final class Qwen3_5GatedDeltaNet: Module {
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

    // Separate projections (unlike Qwen3Next's fused in_proj_qkvz / in_proj_ba)
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3_5RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen3_5TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(numVHeads % numKHeads == 0, "num_v_heads must be divisible by num_k_heads")

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

        _norm.wrappedValue = Qwen3_5RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        // Separate projections
        var mixedQKV = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let dtype = inputs.dtype
        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: dtype)
        }

        if let mask {
            mixedQKV = MLX.where(
                expandedDimensions(mask, axis: -1), mixedQKV, MLXArray.zeros(like: mixedQKV))
        }

        let convInput = concatenated([convState, mixedQKV], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (1 - convKernelSize)..., 0...]
        }

        let convOut = silu(conv1d(convInput))
        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)

        var qOut = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        var kOut = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let vOut = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let invScale = pow(Float(headKDim), -0.5)
        qOut =
            (invScale * invScale)
            * MLXFast.rmsNorm(qOut, weight: MLXArray.mlxNone, eps: 1e-6)
        kOut = invScale * MLXFast.rmsNorm(kOut, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newState) = gatedDeltaUpdate(
            q: qOut,
            k: kOut,
            v: vOut,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: cache?[1],
            mask: mask
        )

        if let cache {
            cache[1] = newState
        }

        let normalized = norm(out, gate: z)
        return outProj(normalized.reshaped(B, S, -1))
    }
}

private final class Qwen3_5SparseMoeBlock: Module {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3_5MLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen3_5TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3_5MLP(
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

private final class Qwen3_5DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3_5Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3_5GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen3_5TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen3_5GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen3_5Attention(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Use MoE if num_experts > 0, otherwise standard MLP
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen3_5SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3_5MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let h: MLXArray
        if isLinear {
            h = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            h = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let r = x + h
        let normed = postAttentionLayerNorm(r)
        if let moe = mlp as? Qwen3_5SparseMoeBlock {
            return r + moe(normed)
        }
        return r + (mlp as! Qwen3_5MLP)(normed)
    }
}

// MARK: - Model

public class Qwen3_5ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3_5DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen3_5TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen3_5DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask = layer.isLinear ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
        }

        return norm(hiddenStates)
    }
}

public class Qwen3_5Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3_5ModelInner
    let configuration: Qwen3_5TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3_5Configuration) {
        let textConfig = args.textConfig
        self.configuration = textConfig
        self.vocabularySize = textConfig.vocabularySize
        self.kvHeads = (0 ..< textConfig.hiddenLayers).map { _ in textConfig.kvHeads }
        self.model = Qwen3_5ModelInner(textConfig)

        if !textConfig.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(textConfig.hiddenSize, textConfig.vocabularySize, bias: false)
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

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func makeCache() -> [KVCache] {
        return newCache(parameters: nil)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        // Detect if weights are from original HF format (need norm shifting)
        let hasMTPWeights = weights.keys.contains(where: { $0.contains("mtp.") })
        let hasUnsanitizedConv1d = weights.contains(where: {
            $0.key.contains("conv1d.weight") && $0.value.dim(-1) != 1
        })
        let shouldShiftNorms = hasMTPWeights || hasUnsanitizedConv1d

        for (key, value) in weights {
            // Skip MTP weights
            if key.contains("mtp.") { continue }

            // Skip vision weights
            if key.hasPrefix("model.visual") || key.hasPrefix("vision_tower.") { continue }

            // Remap VLM key prefixes to match our model structure
            var newKey = key
            if key.hasPrefix("language_model.") {
                newKey = String(key.dropFirst("language_model.".count))
            }

            sanitizedWeights[newKey] = value
        }

        // Handle tie_word_embeddings
        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Norm weight shifting and conv1d transposition
        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for key in Array(sanitizedWeights.keys) {
            guard let value = sanitizedWeights[key] else { continue }

            // Transpose conv1d weights if needed
            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                sanitizedWeights[key] = value.movedAxis(source: 2, destination: 1)
                continue
            }

            // Shift norm weights (only if from original HF format)
            if shouldShiftNorms && normSuffixes.contains(where: { key.hasSuffix($0) })
                && value.ndim == 1
            {
                sanitizedWeights[key] = value + 1.0
            }
        }

        // Handle MoE expert weight format (for non-MLX-community weights)
        if configuration.numExperts > 0 {
            if sanitizedWeights["model.layers.0.mlp.experts.0.up_proj.weight"] != nil {
                // Individual expert format — stack into switch_mlp format
                for l in 0 ..< configuration.hiddenLayers {
                    let prefix = "model.layers.\(l).mlp"
                    for n in ["up_proj", "down_proj", "gate_proj"] {
                        let key = "\(prefix).experts.0.\(n).weight"
                        if sanitizedWeights[key] != nil {
                            let toJoin = (0 ..< configuration.numExperts).map { e in
                                sanitizedWeights.removeValue(
                                    forKey: "\(prefix).experts.\(e).\(n).weight")!
                            }
                            sanitizedWeights["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                        }
                    }
                }
            } else if sanitizedWeights.keys.contains(where: { $0.contains("experts.gate_up_proj") }) {
                // Fused gate_up_proj format — split into gate_proj + up_proj
                for l in 0 ..< configuration.hiddenLayers {
                    let prefix = "model.layers.\(l).mlp"
                    let gateUpKey = "\(prefix).experts.gate_up_proj"
                    if let gateUp = sanitizedWeights.removeValue(forKey: gateUpKey) {
                        let mid = gateUp.dim(-2) / 2
                        sanitizedWeights["\(prefix).switch_mlp.gate_proj.weight"] = gateUp[
                            .ellipsis, ..<mid, 0...]
                        sanitizedWeights["\(prefix).switch_mlp.up_proj.weight"] = gateUp[
                            .ellipsis, mid..., 0...]
                        if let downProj = sanitizedWeights.removeValue(
                            forKey: "\(prefix).experts.down_proj")
                        {
                            sanitizedWeights["\(prefix).switch_mlp.down_proj.weight"] = downProj
                        }
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension Qwen3_5Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Configuration

/// Top-level configuration for Qwen3.5 models.
/// Handles both VLM format (nested `text_config`) and flat text-only format.
public struct Qwen3_5Configuration: Codable, Sendable {
    public let textConfig: Qwen3_5TextConfiguration

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if container.contains(.textConfig) {
            // VLM format: text model config is nested inside text_config
            self.textConfig = try container.decode(
                Qwen3_5TextConfiguration.self, forKey: .textConfig)
        } else {
            // Flat format: the top-level IS the text config
            self.textConfig = try Qwen3_5TextConfiguration(from: decoder)
        }
    }
}

/// Text model configuration for Qwen3.5.
/// Supports both dense and MoE variants via optional MoE fields.
public struct Qwen3_5TextConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int

    // MoE fields (zero = dense model)
    var numExperts: Int
    var numExpertsPerTok: Int
    var sharedExpertIntermediateSize: Int
    var moeIntermediateSize: Int
    var normTopkProb: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
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
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    /// Decode-only keys for nested config fields that have no stored property.
    private enum NestedKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5"
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        // MoE models may not have intermediate_size (they use moe_intermediate_size instead)
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        self.linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        self.linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        self.linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        self.linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        // rope_parameters may be a nested dict containing rope_theta,
        // partial_rotary_factor, and rope_scaling (e.g. Qwen3.5 HF configs).
        // Try flat top-level keys first, then fall back to nested dict values.
        let nested = try decoder.container(keyedBy: NestedKeys.self)
        let ropeParams = try nested.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = theta
        } else if let theta = ropeParams?["rope_theta"]?.asFloat() {
            self.ropeTheta = theta
        } else {
            self.ropeTheta = 1_000_000
        }

        if let prf = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) {
            self.partialRotaryFactor = prf
        } else if let prf = ropeParams?["partial_rotary_factor"]?.asFloat() {
            self.partialRotaryFactor = prf
        } else {
            self.partialRotaryFactor = 1.0
        }

        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.fullAttentionInterval =
            try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields — default to 0 for dense models
        self.numExperts =
            try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb =
            try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
    }
}
