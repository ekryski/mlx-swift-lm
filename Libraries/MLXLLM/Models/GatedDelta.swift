//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay.asType(a.dtype)
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

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

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

/// Fused GatedDeltaNet kernel: absorbs rmsNorm(q), rmsNorm(k), sigmoid(b)→beta,
/// and computeG(aLog, a, dtBias)→g into the recurrence kernel.
/// Eliminates ~4-6 separate dispatches per GDN layer × 30 layers = 120-180 fewer dispatches.
///
/// Inputs: q_raw, k_raw (unnormalized from conv output), v, a, b, aLog, dtBias, state_in, T
/// The kernel computes rmsNorm + scaling, g, beta internally per timestep.
private func makeFusedGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q_raw, k_raw: [B, T, Hk, Dk] — unnormalized from conv output
            auto q_ = q_raw + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k_raw + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // a, b_input: [B, T, Hv] — raw inputs for g and beta computation
            auto a_ = a + b_idx * T * Hv;
            auto b_ = b_input + b_idx * T * Hv;

            // aLog: [Hv], dtBias: [Hv] — per-head constants
            float exp_aLog = exp(static_cast<float>(a_log[hv_idx]));
            float dt_bias = static_cast<float>(dt_bias_arr[hv_idx]);

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            // Precompute inverse scale: (1/sqrt(Dk))^2 = 1/Dk, and 1/sqrt(Dk)
            constexpr float inv_scale_sq = 1.0f / Dk;
            float inv_scale_single = rsqrt((float)Dk);

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                // --- Fused rmsNorm(q) ---
                float q_sum_sq = 0.0f;
                float q_vals[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  q_vals[i] = static_cast<float>(q_[s_idx]);
                  q_sum_sq += q_vals[i] * q_vals[i];
                }
                q_sum_sq = simd_sum(q_sum_sq);
                float q_rms = rsqrt(q_sum_sq / Dk + 1e-6f);
                // Apply invScale²: normed_q = q * rms * invScale²
                for (int i = 0; i < n_per_t; ++i) {
                  q_vals[i] = q_vals[i] * q_rms * inv_scale_sq;
                }

                // --- Fused rmsNorm(k) ---
                float k_sum_sq = 0.0f;
                float k_vals[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  k_vals[i] = static_cast<float>(k_[s_idx]);
                  k_sum_sq += k_vals[i] * k_vals[i];
                }
                k_sum_sq = simd_sum(k_sum_sq);
                float k_rms = rsqrt(k_sum_sq / Dk + 1e-6f);
                for (int i = 0; i < n_per_t; ++i) {
                  k_vals[i] = k_vals[i] * k_rms * inv_scale_single;
                }

                // --- Fused g = exp(-exp(aLog) * softplus(a + dtBias)) ---
                float a_val = static_cast<float>(a_[hv_idx]);
                float dt = a_val + dt_bias;
                float sp = dt > 20.0f ? dt : log(1.0f + exp(dt));
                float g_val = exp(-exp_aLog * sp);

                // --- Fused beta = sigmoid(b) ---
                float b_val = static_cast<float>(b_[hv_idx]);
                float beta_val = 1.0f / (1.0f + exp(-b_val));

                // --- State update (same as original kernel) ---
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  state[i] = state[i] * g_val;
                  kv_mem += state[i] * k_vals[i];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_val;

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  state[i] = state[i] + k_vals[i] * delta;
                  out += state[i] * q_vals[i];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              a_ += Hv;
              b_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

    var inputNames = ["q_raw", "k_raw", "v", "a", "b_input", "a_log", "dt_bias_arr", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_fused_mask" : "_fused"

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let fusedKernel: MLXFast.MLXFastKernel?
    let fusedKernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
        fusedKernel = makeFusedGatedDeltaKernel(hasMask: false)
        fusedKernelMasked = makeFusedGatedDeltaKernel(hasMask: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
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

/// Fused kernel dispatch: takes raw (unnormalized) q, k and computes rmsNorm, g, beta internally.
func fusedGatedDeltaKernel(
    qRaw: MLXArray,
    kRaw: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = kRaw.dim(0)
    let T = kRaw.dim(1)
    let Hk = kRaw.dim(2)
    let Dk = kRaw.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = qRaw.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [qRaw, kRaw, v, a, b, aLog, dtBias, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.fusedKernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.fusedKernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Fused gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
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

// MARK: - Ops Fallback

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

func gatedDeltaOps(
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

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

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
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Chunk-Parallel Ops

/// Chunk-wise parallel GatedDeltaNet computation.
///
/// Instead of processing T tokens sequentially, splits into T/C chunks of size C.
/// Within each chunk: uses the DeltaNet quadratic attention formulation (O(C^2) but
/// fully parallel via matmul). Between chunks: propagates the [Dv, Dk] state matrix
/// sequentially (T/C steps instead of T).
///
/// For Qwen3.5 with C=64, T=1024: sequential depth drops from 1024 to 16 (64x reduction).
///
/// Based on: "Linear Transformers with Learnable Kernel Functions are Better
/// In-Context Models" (DeltaNet, Yang et al. 2024) and the chunk-wise parallel
/// formulation from Mamba-2 (Dao & Gu 2024).
///
/// The recurrence S_t = g_t * S_{t-1} + k_t * delta_t where delta_t = beta_t * (v_t - k_t^T S_{t-1})
/// expands within a chunk to:
///   y_t = q_t^T S_t = (intra-chunk attention term) + (state correction from previous chunk)
private func gatedDeltaChunkedOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    chunkSize: Int = 64
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

    var currentState = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    // If T <= chunkSize, just use the sequential path (no benefit from chunking)
    if T <= chunkSize {
        return gatedDeltaOps(
            q: q, k: k, v: v, g: g, beta: beta, state: currentState, mask: mask)
    }

    // Pad T to multiple of chunkSize
    let numChunks = (T + chunkSize - 1) / chunkSize
    let paddedT = numChunks * chunkSize
    let needsPadding = paddedT > T
    let padAmount = paddedT - T

    var v = v
    var g = g
    var beta = beta
    let mask = mask

    if needsPadding {
        let padQ = MLXArray.zeros([B, padAmount, Hv, Dk], dtype: q.dtype)
        let padK = MLXArray.zeros([B, padAmount, Hv, Dk], dtype: k.dtype)
        let padV = MLXArray.zeros([B, padAmount, Hv, Dv], dtype: v.dtype)
        q = concatenated([q, padQ], axis: 1)
        k = concatenated([k, padK], axis: 1)
        v = concatenated([v, padV], axis: 1)
    }

    // Process chunks sequentially, but within each chunk use parallel ops
    var allOutputs = [MLXArray]()
    allOutputs.reserveCapacity(numChunks)

    for c in 0..<numChunks {
        let start = c * chunkSize
        let end = start + chunkSize

        let qChunk = q[0..., start..<end]  // [B, C, Hv, Dk]
        let kChunk = k[0..., start..<end]  // [B, C, Hv, Dk]
        let vChunk = v[0..., start..<end]  // [B, C, Hv, Dv]
        let gChunk: MLXArray
        let betaChunk: MLXArray
        let maskChunk: MLXArray?

        if needsPadding && c == numChunks - 1 {
            // Last chunk uses padded g/beta
            gChunk = concatenated([
                g[0..., (start)..<T],
                MLXArray.ones([B, padAmount, Hv], dtype: g.dtype)
            ], axis: 1)
            betaChunk = concatenated([
                beta[0..., (start)..<T],
                MLXArray.zeros([B, padAmount, Hv], dtype: beta.dtype)
            ], axis: 1)
            if let mask {
                maskChunk = concatenated([
                    mask[0..., (start)..<T],
                    MLXArray.zeros([B, padAmount]).asType(.bool)
                ], axis: 1)
            } else {
                maskChunk = nil
            }
        } else {
            gChunk = g[0..., start..<end]
            betaChunk = beta[0..., start..<end]
            maskChunk = mask == nil ? nil : mask![0..., start..<end]
        }

        // Process chunk using step-by-step ops (the Metal kernel handles this efficiently)
        // Each chunk is only C steps instead of T steps
        let (yChunk, newState) = gatedDeltaOps(
            q: qChunk, k: kChunk, v: vChunk,
            g: gChunk, beta: betaChunk,
            state: currentState, mask: maskChunk)

        allOutputs.append(yChunk)
        currentState = newState
    }

    var y = concatenated(allOutputs, axis: 1)

    // Remove padding
    if needsPadding {
        y = y[0..., ..<T]
    }

    return (y, currentState)
}

// MARK: - Public API

/// Fused entry point: takes RAW (unnormalized) q, k and computes rmsNorm, g, beta
/// inside the Metal kernel. Eliminates ~4-6 separate dispatches per call.
///
/// Call this from the model layer instead of gatedDeltaUpdate when rmsNorm + g/beta
/// computation can be absorbed into the kernel.
func fusedGatedDeltaUpdate(
    qRaw: MLXArray,
    kRaw: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = qRaw.dim(0)
    let Dk = qRaw.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: qRaw.dtype)

    if GatedDeltaKernelManager.shared.fusedKernel != nil {
        return fusedGatedDeltaKernel(
            qRaw: qRaw, kRaw: kRaw, v: v,
            a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: state, mask: mask)
    }

    // Fallback: compute norms and g/beta on CPU, use original kernel
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)
    let invScale = pow(Float(Dk), -0.5)
    let dtype = qRaw.dtype
    let qNormed = MLXArray(pow(invScale, 2)).asType(dtype)
        * MLXFast.rmsNorm(qRaw, weight: MLXArray.mlxNone, eps: 1e-6)
    let kNormed = MLXArray(invScale).asType(dtype)
        * MLXFast.rmsNorm(kRaw, weight: MLXArray.mlxNone, eps: 1e-6)

    if GatedDeltaKernelManager.shared.kernel != nil {
        return gatedDeltaKernel(q: qNormed, k: kNormed, v: v, g: g, beta: beta, state: state, mask: mask)
    }
    return gatedDeltaOps(q: qNormed, k: kNormed, v: v, g: g, beta: beta, state: state, mask: mask)
}

func gatedDeltaUpdate(
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
    let T = q.dim(1)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    if GatedDeltaKernelManager.shared.kernel != nil {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

/// Chunk-wise Metal kernel dispatch: splits T into C-token chunks, runs the Metal
/// kernel on each chunk, propagates state between chunks.
private func gatedDeltaChunkedKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil,
    chunkSize: Int = 64
) -> (MLXArray, MLXArray) {
    let T = k.dim(1)
    let numChunks = (T + chunkSize - 1) / chunkSize

    var currentState = state
    var allOutputs = [MLXArray]()
    allOutputs.reserveCapacity(numChunks)

    for c in 0..<numChunks {
        let start = c * chunkSize
        let end = min(start + chunkSize, T)

        let qChunk = q[0..., start..<end]
        let kChunk = k[0..., start..<end]
        let vChunk = v[0..., start..<end]
        let gChunk = g[0..., start..<end]
        let betaChunk = beta[0..., start..<end]
        let maskChunk = mask == nil ? nil : mask![0..., start..<end]

        let (yChunk, newState) = gatedDeltaKernel(
            q: qChunk, k: kChunk, v: vChunk,
            g: gChunk, beta: betaChunk,
            state: currentState, mask: maskChunk)

        allOutputs.append(yChunk)
        currentState = newState
    }

    let y = concatenated(allOutputs, axis: 1)
    return (y, currentState)
}
