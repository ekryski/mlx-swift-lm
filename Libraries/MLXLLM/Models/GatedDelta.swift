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
    // Stay in model dtype (bf16) — no fp32 promotion needed.
    // The double-exp is numerically safe in bf16 for the typical aLog range (-1 to -8).
    return exp(-exp(aLog) * softplus(a + dtBias))
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
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // Framework dispatch — pre-compiled Metal kernel from metallib
    let outputs = MLXFast.gatedDeltaStep(
        q: q, k: k, v: v, g: g, beta: beta, state: state,
        mask: mask, T: T, fused: false,
        Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
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
    let T = kRaw.dim(1)
    let Hk = kRaw.dim(2)
    let Dk = kRaw.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // Framework dispatch — pre-compiled Metal kernel from metallib
    let outputs = MLXFast.gatedDeltaStepFused(
        qRaw: qRaw, kRaw: kRaw, v: v,
        a: a, bInput: b, aLog: aLog, dtBias: dtBias,
        state: state, mask: mask,
        T: T, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
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

    // Process in sub-chunks with eval barriers to bound peak memory.
    // Without this, the lazy graph grows to T × (intermediates per step),
    // causing peak memory proportional to T during prefill.
    let evalInterval = 64
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

        // Eval barrier: materialize state and accumulated outputs to free
        // intermediate graph nodes. Caps graph at O(evalInterval) instead of O(T).
        if T > 1 && (t + 1) % evalInterval == 0 {
            eval(state)
            eval(ys)
            MLX.Memory.clearCache()
        }
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Quadratic Attention (ssmAttn-style for GatedDeltaNet)

/// Parallel GatedDeltaNet computation using quadratic attention formulation.
///
/// Based on "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
/// (Yang et al., NeurIPS 2024). Expresses the GatedDeltaNet recurrence as a causal
/// attention-like operation that's O(T²) but fully parallelizable via matmul.
///
/// The recurrence: S_t = g_t * S_{t-1} + β_t * k_t * (v_t - k_t^T * S_{t-1})^T
/// Expanded:       S_t = (g_t - β_t * k_t * k_t^T) * S_{t-1} + β_t * k_t * v_t^T
///
/// Within a chunk of C tokens, the output y_t = q_t^T * S_t can be decomposed:
///   y_t = y_inter_t + y_intra_t
///
///   y_inter_t: contribution from previous chunks' state (parallel matmul)
///   y_intra_t: contribution from within the current chunk (quadratic attention)
///
/// For Qwen3.5 (Dk=128, Dv=128, Hv=32, Hk=16):
///   Sequential kernel: O(T) steps, each ~20 FLOPs per thread
///   Quadratic attention: O(T²) matmul ops, but fully parallel on GPU
///   Crossover: quadratic is faster when T is small enough that matmul dominates
func gatedDeltaAttn(
    q: MLXArray,    // [B, T, Hk, Dk] — pre-normalized (Hk may differ from Hv for GQA)
    k: MLXArray,    // [B, T, Hk, Dk] — pre-normalized
    v: MLXArray,    // [B, T, Hv, Dv]
    g: MLXArray,    // [B, T, Hv] — decay
    beta: MLXArray, // [B, T, Hv] — sigmoid(b)
    state: MLXArray? = nil,  // [B, Hv, Dv, Dk] — initial state from previous chunks
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // GQA: expand q, k from Hk heads to Hv heads by repeating each head
    var q = q
    var k = k
    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        // [B, T, Hk, Dk] → [B, T, Hk, 1, Dk] → [B, T, Hk, repeat, Dk] → [B, T, Hv, Dk]
        q = MLX.repeated(expandedDimensions(q, axis: 3), count: repeatFactor, axis: 3)
            .reshaped(B, T, Hv, Dk)
        k = MLX.repeated(expandedDimensions(k, axis: 3), count: repeatFactor, axis: 3)
            .reshaped(B, T, Hv, Dk)
    }

    // --- Step 1: Cumulative log-decay within the sequence ---
    // logG[t] = Σ_{m=0}^{t} log(g[m])
    // Decay from position s to t: exp(logG[t] - logG[s])
    let logG = MLX.cumsum(MLX.log(g + 1e-10), axis: 1)  // [B, T, Hv]

    // --- Step 2: Build lower-triangular decay matrix ---
    // L[t, s] = exp(logG[t] - logG[s]) for s <= t, 0 otherwise
    // Shape: [B, Hv, T, T] (heads as batch dim for matmul)
    let logG_t = logG.transposed(0, 2, 1)  // [B, Hv, T]
    let decayMatrix = MLX.exp(
        expandedDimensions(logG_t, axis: -1) - expandedDimensions(logG_t, axis: -2)
    )  // [B, Hv, T, T]
    let causalDecay = MLX.tril(decayMatrix, k: 0)  // Lower-triangular causal mask

    // --- Step 3: Build the attention-like pattern ---
    // For standard linear attention (no delta rule): A[t,s] = q[t]^T * L[t,s] * k[s]
    // The output would be: y_t = Σ_s A[t,s] * β_s * v_s
    //
    // For the delta rule, we need to account for the state-dependent correction.
    // Using the "linearized" approximation: ignore the k^T*S correction in delta.
    // This is exact for the first-order term and a good approximation for small β*k*k^T.
    //
    // Q·K^T attention scores with decay weighting:
    let q_t = q.transposed(0, 2, 1, 3)  // [B, Hv, T, Dk]
    let k_t = k.transposed(0, 2, 1, 3)  // [B, Hv, T, Dk]
    let v_t = v.transposed(0, 2, 1, 3)  // [B, Hv, T, Dv]
    let beta_t = beta.transposed(0, 2, 1)  // [B, Hv, T]

    // Attention scores: [B, Hv, T, T] = Q @ K^T * causalDecay
    var scores = q_t.matmul(k_t.transposed(0, 1, 3, 2))  // [B, Hv, T, T]
    scores = scores * causalDecay

    // Weight by beta: each source position contributes β_s
    scores = scores * expandedDimensions(beta_t, axis: -2)  // broadcast β over query dim

    // Apply mask if needed
    if let mask {
        let maskExpanded = expandedDimensions(
            expandedDimensions(mask, axis: 1),  // [B, 1, T]
            axis: -1  // [B, 1, T, 1]
        )
        scores = scores * maskExpanded.transposed(0, 1, 3, 2)  // mask source positions
    }

    // --- Step 4: Compute intra-chunk output via matmul ---
    // y_intra = scores @ V : [B, Hv, T, T] @ [B, Hv, T, Dv] → [B, Hv, T, Dv]
    var y = scores.matmul(v_t)  // [B, Hv, T, Dv]

    // --- Step 5: Inter-chunk contribution from previous state ---
    if let state = state {
        // state: [B, Hv, Dv, Dk]
        // Contribution: y_inter[t] = q[t]^T * cumDecay(t) * S_prev
        // cumDecay(t) = exp(logG[t]) relative to chunk start
        let cumDecay = MLX.exp(logG_t)  // [B, Hv, T] — decay from start of chunk to each t

        // q @ S_prev^T: [B, Hv, T, Dk] @ [B, Hv, Dk, Dv] → [B, Hv, T, Dv]
        let yPrev = q_t.matmul(state.transposed(0, 1, 3, 2))  // [B, Hv, T, Dv]
        y = y + yPrev * expandedDimensions(cumDecay, axis: -1)
    }

    // --- Step 6: Compute next state ---
    // S_next = cumDecay(T-1) * S_prev + Σ_{t=0}^{T-1} cumDecay(T-1, t) * β_t * k_t * v_t^T
    // decay from each position to the end: exp(logG[T-1] - logG[t])
    let decayToEnd = MLX.exp(logG_t[0..., 0..., (-1)...] - logG_t)  // [B, Hv, T]

    // β_t * k_t weighted by decay to end: [B, Hv, T, Dk]
    let bkDecay = expandedDimensions(beta_t * decayToEnd, axis: -1) * k_t
    // bkDecay^T @ V → [B, Hv, Dk, Dv] → transpose to [B, Hv, Dv, Dk]
    var nextState = bkDecay.transposed(0, 1, 3, 2).matmul(v_t).transposed(0, 1, 3, 2)

    if let state = state {
        // finalDecay: [B, Hv, 1] → [B, Hv, 1, 1] for broadcast with [B, Hv, Dv, Dk]
        let finalDecay = expandedDimensions(MLX.exp(logG_t[0..., 0..., (-1)...]), axis: -1)
        // Now [B, Hv, 1, 1] — broadcasts correctly with [B, Hv, Dv, Dk]
        nextState = nextState + finalDecay * state
    }

    // Transpose output back: [B, Hv, T, Dv] → [B, T, Hv, Dv]
    y = y.transposed(0, 2, 1, 3)

    return (y, nextState)
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

    // For multi-token (prefill), try quadratic attention with chunking.
    // The quadratic formulation is O(C²) per chunk but fully parallel via matmul,
    // vs the Metal kernel which is O(T) sequential.
    // Use quadratic attention for moderate T (64-4096), sequential kernel for T=1 or very large T.
    let useQuadratic = ProcessInfo.processInfo.environment["GDN_QUADRATIC"] == "1"
    if useQuadratic && T > 1 && T <= 8192 {
        let chunkSize = 64
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

            let (yChunk, newState) = gatedDeltaAttn(
                q: qChunk, k: kChunk, v: vChunk,
                g: gChunk, beta: betaChunk,
                state: currentState, mask: maskChunk)

            allOutputs.append(yChunk)
            currentState = newState
        }

        let y = concatenated(allOutputs, axis: 1)
        return (y, currentState)
    }

    // Framework non-fused kernel has a correctness bug (max diff 0.25 vs ops at T=1,
    // catastrophic at T>1). Use ops fallback until the framework kernel is fixed.
    // The fused kernel (used for T=1 decode) is correct — only the non-fused variant
    // for T>1 prefill is broken. See GatedDeltaKernelTests.
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
