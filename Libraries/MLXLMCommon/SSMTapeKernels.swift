// Copyright © 2026 Apple Inc.
//
// GDN tape kernels for `SSMStateCache` tape-replay rollback (spec 020 phase 2).
//
// Two paired Metal kernels:
//
//   1. `gated_delta_step_with_tape` — forward kernel variant that ALSO
//      writes per-step `delta_t` to a tape output buffer. Used by
//      speculative-decoder verify forwards on hybrid GDN+Attention models
//      (Qwen 3.5 / 3.6, Nemotron-H, Jamba).
//
//   2. `tape_replay` — re-folds the accepted prefix of an innovation tape
//      from a pre-record snapshot. Used by
//      `SSMStateCache.rollback(acceptedPrefix:)` on partial accept of a
//      verify round.
//
// Both adopt upstream dflash-mlx's correctness patterns from day 1:
//
//   - Masked-timestep correctness fix (commit `3217e15`): save `old_state`
//     before each step, restore via `metal::select` on masked positions.
//     Without this, masked positions silently corrupt state.
//   - Branchless pattern (commit `c9f992e`): `metal::select(old, new,
//     do_step)` instead of `if(do_step)` guards — better SIMD occupancy
//     when timestep masks are non-uniform within a SIMD group.
//
// Both kernels live in `MLXLMCommon` (and not next to the regular forward
// kernel in `MLXLLM/Models/GatedDelta.swift`) because `SSMStateCache.rollback`
// is in this module and the matched pair belongs together. The high-level
// layer dispatcher `gatedDeltaUpdateWithTape` stays in MLXLLM because it
// falls back to the regular `gatedDeltaUpdate` on the no-recording fast path.

import Foundation
import MLX

// MARK: - Forward-with-tape JIT kernel
//
// Variant of `gated_delta_step` that ALSO writes per-step `delta_t` to a
// tape output buffer. The body is byte-identical to `gated_delta_step`
// (see `MLXLLM/Models/GatedDelta.swift:makeGatedDeltaKernel`) except for
// the extra `tape_delta` write inside the per-step loop. Masked timesteps
// don't write the tape — the replay kernel reads its own mask buffer and
// skips those positions via `metal::select`.

private func makeGatedDeltaKernelWithTape(hasMask: Bool) -> MLXFast.MLXFastKernel? {
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

            // v, y, tape_delta: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;
            auto td_ = tape_delta + b_idx * T * Hv * Dv + hv_idx * Dv;

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

                // Tape write: one lane per SIMD group writes delta (all
                // SIMD lanes hold the same value post-simd_sum). Same
                // discipline as the y write below.
                if (thread_index_in_simdgroup == 0) {
                  td_[dv_idx] = static_cast<InT>(delta);
                }

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
              y  += Hv * Dv;
              td_ += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step_with_tape\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out", "tape_delta"],
        source: source
    )
}

private final class GatedDeltaWithTapeKernelManager: Sendable {
    static let shared = GatedDeltaWithTapeKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernelWithTape(hasMask: false)
        kernelMasked = makeGatedDeltaKernelWithTape(hasMask: true)
    }
}

/// Forward GDN kernel that also captures per-step `delta_t` into a tape
/// output buffer. Returns `(y, state_out, tape_delta)` — the tape_delta is
/// `[B, T, Hv, Dv]`, sliced and passed to `SSMStateCache.recordStep` by
/// the layer wrapper `gatedDeltaUpdateWithTape`.
public func gatedDeltaKernelWithTape(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaWithTapeKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaWithTapeKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta with-tape kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv]],
        outputDTypes: [inputType, stateType, inputType]
    )

    return (outputs[0], outputs[1], outputs[2])
}

// MARK: - Tape-replay JIT kernel

private func makeTapeReplayKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T_tape + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            constexpr int n_per_t = Dk / 32;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // delta_tape: [B, T_tape, Hv, Dv]
            // k_tape:     [B, T_tape, Hv, Dk]  (already GQA-expanded)
            // g_tape:     [B, T_tape, Hv]
            auto delta_ = delta_tape + b_idx * T_tape * Hv * Dv + hv_idx * Dv;
            auto k_     = k_tape     + b_idx * T_tape * Hv * Dk + hv_idx * Dk;
            auto g_     = g_tape     + b_idx * T_tape * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            }

            for (int t = 0; t < T_tape; ++t) {
              bool within_accepted = (t < accepted);
              bool mask_passes     = (\(maskSource));
              bool do_step         = within_accepted && mask_passes;

              // Save old state for masked / out-of-range restoration.
              float old_state[n_per_t];
              for (int i = 0; i < n_per_t; ++i) {
                old_state[i] = state[i];
              }

              float g_val = static_cast<float>(g_[hv_idx]);
              float d_val = static_cast<float>(delta_[dv_idx]);

              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                float new_val = state[i] * g_val
                              + static_cast<float>(k_[s_idx]) * d_val;
                state[i] = metal::select(old_state[i], new_val, do_step);
              }

              // Advance pointers regardless of do_step so indexing stays
              // in lockstep with the tape layout.
              delta_ += Hv * Dv;
              k_     += Hv * Dk;
              g_     += Hv;
            }

            for (int i = 0; i < n_per_t; ++i) {
              o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["delta_tape", "k_tape", "g_tape", "state_in", "T_tape", "accepted"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "tape_replay\(suffix)",
        inputNames: inputNames,
        outputNames: ["state_out"],
        source: source
    )
}

private final class TapeReplayKernelManager: Sendable {
    static let shared = TapeReplayKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeTapeReplayKernel(hasMask: false)
        kernelMasked = makeTapeReplayKernel(hasMask: true)
    }
}

// MARK: - Replay dispatcher

/// Re-fold the accepted prefix of a tape onto a pre-record state snapshot.
/// Used by `SSMStateCache.rollback(acceptedPrefix:)`. Adopts the masked-
/// timestep correctness fix + branchless `metal::select` pattern from
/// upstream dflash-mlx commits `3217e15` and `c9f992e`.
///
/// `tape` is the cache's tape buffer: an array of `[delta_t, k_t, g_t]`
/// triples per step. The Swift wrapper stacks the per-step tensors into
/// `[B, T_tape, Hv, *]` buffers before kernel dispatch.
public func gatedDeltaReplayUpdate(
    state: MLXArray,
    tape: [[MLXArray]],
    acceptedPrefix k: Int,
    mask: MLXArray? = nil
) -> MLXArray {
    precondition(!tape.isEmpty, "tape must be non-empty")
    precondition(
        k >= 0 && k <= tape.count,
        "acceptedPrefix (\(k)) out of range [0, \(tape.count)]")

    // Stack per-step tensors into [B, T_tape, ...] buffers.
    // Each tape entry is [delta, k, g] (the SSMStateCache contract).
    let deltaPerStep = tape.map { $0[0][.newAxis, .ellipsis] }
    let kPerStep     = tape.map { $0[1][.newAxis, .ellipsis] }
    let gPerStep     = tape.map { $0[2][.newAxis, .ellipsis] }
    // Concatenate along the new axis (becomes T_tape) and move it to
    // position 1: [B, T_tape, ...].
    let deltaTape = concatenated(deltaPerStep, axis: 0).transposed(1, 0, 2, 3)
    let kTape     = concatenated(kPerStep,     axis: 0).transposed(1, 0, 2, 3)
    let gTape     = concatenated(gPerStep,     axis: 0).transposed(1, 0, 2)

    let B  = state.dim(0)
    let Hv = state.dim(1)
    let Dv = state.dim(2)
    let Dk = state.dim(3)
    let T_tape = tape.count
    let Hk = Hv  // tape stores GQA-expanded k, so Hk == Hv in this kernel
    let inputType = deltaTape.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [deltaTape, kTape, gTape, state, MLXArray(T_tape), MLXArray(k)]
    if let mask {
        selectedKernel = TapeReplayKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = TapeReplayKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Tape replay kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [state.shape],
        outputDTypes: [stateType]
    )

    return outputs[0]
}
