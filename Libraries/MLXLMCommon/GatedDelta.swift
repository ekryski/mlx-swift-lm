//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Shared Gated Delta Net operations used by Qwen3.5, Qwen3Next, and their VLM variants.
//  Includes a Metal kernel for GPU-accelerated processing with ops-based fallback.
//
//  Port of the Python mlx-lm gated_delta.py kernel:
//  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Diagnostics

/// One-time diagnostic logging for Gated DeltaNet Metal kernel dispatch.
private enum GDNKernelDiagnostics {
    nonisolated(unsafe) private static var hasLoggedKernel = false
    nonisolated(unsafe) private static var hasLoggedFallback = false

    static func logKernelUsed(dk: Int, dv: Int) {
        guard !hasLoggedKernel else { return }
        hasLoggedKernel = true
        print(
            "[GDN] Metal kernel active: Dk=\(dk) Dv=\(dv) (Dk%32==0, grid=(32,Dv,B*Hv), threadGroup=(32,4,1))"
        )
    }

    static func logFallback(dk: Int, gNdim: Int, t: Int) {
        guard !hasLoggedFallback else { return }
        hasLoggedFallback = true
        print(
            "[GDN] WARNING: Metal kernel unavailable, using sequential ops fallback. Dk=\(dk) (Dk%32=\(dk%32)), g.ndim=\(gNdim), T=\(t)"
        )
    }
}

// MARK: - Metal Kernel

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

// MARK: - Public API

/// Shared Gated Delta Net operations used by Qwen3.5, Qwen3Next, and their VLM variants.
public enum GatedDelta {

    /// Element-wise sigmoid gating: `x * sigmoid(gate)`.
    public static func sigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
        x * sigmoid(gate)
    }

    /// Compute gating decay: `exp(-exp(aLog) * softplus(a + dtBias))`.
    public static func computeG(
        _ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray
    ) -> MLXArray {
        let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        return decay.asType(a.dtype)
    }

    /// Single timestep state update using MLX ops.
    public static func stepOps(
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

    /// Sequential ops-based loop over all timesteps.
    public static func ops(
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

        for t in 0..<T {
            let qT = q[0..., t]
            let kT = k[0..., t]
            let vT = v[0..., t]
            let gT = g[0..., t]
            let betaT = beta[0..., t]
            let maskT = mask == nil ? nil : mask![0..., t]

            let (y, newState) = stepOps(
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

    /// High-level update using ops only (no Metal kernel). Computes beta and g internally.
    public static func update(
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
        let g = computeG(aLog, a, dtBias)

        let B = q.dim(0)
        let Dk = q.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)

        let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

        return ops(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    /// High-level update with Metal kernel acceleration. Tries GPU kernel first, falls back to ops.
    /// Uses float32 state for numerical stability (matching Qwen3.5 config: mamba_ssm_dtype: "float32").
    public static func updateWithKernel(
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
        let g = computeG(aLog, a, dtBias)

        let B = q.dim(0)
        let Dk = q.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)

        // Use float32 for state to maintain numerical stability in recurrent updates.
        let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

        // Try Metal kernel for GPU-accelerated processing (all timesteps in one dispatch).
        // Falls back to ops-based sequential loop if kernel unavailable or dimensions unsupported.
        if let result = gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask
        ) {
            GDNKernelDiagnostics.logKernelUsed(dk: q.dim(3), dv: v.dim(3))
            return result
        }

        GDNKernelDiagnostics.logFallback(dk: q.dim(3), gNdim: g.ndim, t: q.dim(1))
        return ops(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }
}
