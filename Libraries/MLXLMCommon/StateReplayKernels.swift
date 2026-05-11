// Copyright © 2026 Apple Inc.
//
// GDN tape kernels for `SSMStateCache` tape-replay rollback (spec 020 phase 2).
//
// Native-C Metal kernels via the `MLXFast` framework — the heavy lifting
// lives in `mlx-swift`'s `Source/Cmlx/mlx-generated/metal/gated_delta_replay.metal`
// (precompiled into `mlx.metallib`) and `mlx`'s C++ Primitive classes
// `GatedDeltaStepRecord` / `StateReplay`. This module just hosts the
// dispatch wrappers used by `SSMStateCache.rollback(acceptedPrefix:)` and
// (indirectly via `gatedDeltaUpdateRecord` in MLXLLM) by GDN layer
// forwards under a recording session.
//
// Two paired primitives:
//
//   1. `MLXFast.gatedDeltaStepRecord` — forward kernel variant that ALSO
//      writes per-step `delta_t` to a tape output buffer. Used by
//      speculative-decoder verify forwards on hybrid GDN+Attention models
//      (Qwen 3.5 / 3.6).
//
//   2. `MLXFast.stateReplay` — re-folds the accepted prefix of an
//      innovation tape from a pre-record snapshot. Used by
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
// The forward-with-tape dispatcher `gatedDeltaUpdateRecord` stays in
// MLXLLM because it falls back to the regular `gatedDeltaUpdate` on the
// no-recording fast path. The replay dispatcher (`stateReplayUpdate`
// below) is here so `SSMStateCache.rollback` can call it directly.

import Foundation
import MLX

// MARK: - Replay dispatcher

/// Re-fold the accepted prefix of a delta log onto a pre-record state
/// snapshot. Used by `SSMStateCache.rollback(acceptedPrefix:)`. Routes
/// through the native `MLXFast.stateReplay` kernel (precompiled in
/// `mlx.metallib`).
///
/// `deltaLog` is the cache's recorded log: an array of `[delta_t, k_t, g_t]`
/// triples per step. The wrapper stacks the per-step tensors into
/// `[B, T_log, Hv, *]` buffers before dispatching the native kernel.
public func stateReplayUpdate(
    state: MLXArray,
    deltaLog: [[MLXArray]],
    acceptedPrefix k: Int,
    mask: MLXArray? = nil
) -> MLXArray {
    precondition(!deltaLog.isEmpty, "delta log must be non-empty")
    precondition(
        k >= 0 && k <= deltaLog.count,
        "acceptedPrefix (\(k)) out of range [0, \(deltaLog.count)]")

    // Stack per-step tensors into [B, T_log, ...] buffers.
    // Each entry is [delta, k, g] (the SSMStateCache contract).
    let deltaPerStep = deltaLog.map { $0[0][.newAxis, .ellipsis] }
    let kPerStep     = deltaLog.map { $0[1][.newAxis, .ellipsis] }
    let gPerStep     = deltaLog.map { $0[2][.newAxis, .ellipsis] }
    // Concatenate along the new axis (becomes T_log) and move it to
    // position 1: [B, T_log, ...].
    let deltaBuf = concatenated(deltaPerStep, axis: 0).transposed(1, 0, 2, 3)
    let kBuf     = concatenated(kPerStep,     axis: 0).transposed(1, 0, 2, 3)
    let gBuf     = concatenated(gPerStep,     axis: 0).transposed(1, 0, 2)

    let Hv = state.dim(1)
    let Dv = state.dim(2)
    let Dk = state.dim(3)
    let T_log = deltaLog.count
    let Hk = Hv  // delta log stores GQA-expanded k, so Hk == Hv in this kernel

    let outputs = MLXFast.stateReplay(
        deltaLog: deltaBuf,
        kLog: kBuf,
        gLog: gBuf,
        state: state,
        mask: mask,
        T_log: T_log,
        accepted: k,
        Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv
    )
    return outputs[0]
}

// MARK: - Forward-with-record dispatcher (low-level)

/// Forward GDN kernel that also records per-step `delta_t` into the
/// `deltaLog` output buffer. Returns `(y, state_out, deltaLog)` — the
/// deltaLog is `[B, T, Hv, Dv]`, sliced and passed to
/// `SSMStateCache.recordStep` by
/// the layer wrapper `gatedDeltaUpdateRecord` (in MLXLLM).
public func gatedDeltaKernelRecord(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray, MLXArray) {
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let outputs = MLXFast.gatedDeltaStepRecord(
        q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask,
        T: T, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv
    )
    // outputs = [y, state_out, tape_delta]
    return (outputs[0], outputs[1], outputs[2])
}
