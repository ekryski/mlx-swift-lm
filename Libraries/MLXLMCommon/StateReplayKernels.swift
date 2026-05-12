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
/// `deltaLog` is the cache's recorded log. The protocol contract changed
/// in spec 020's perf pass to **one entry per verify round** rather than
/// one entry per step — the entry's tensors carry the T-axis intact, so
/// no per-step slicing happens at record time. Two shapes are supported:
///
/// - **T-axis (per-round, preferred)**: each entry is `[delta, k, g]` with
///   shapes `[B, T, Hv, Dv]` / `[B, T, Hv, Dk]` / `[B, T, Hv]`. Typical
///   iterator usage: one `recordStep` per verify forward.
///
/// - **Per-step (legacy)**: each entry is `[delta, k, g]` with shapes
///   `[B, Hv, Dv]` / `[B, Hv, Dk]` / `[B, Hv]` (no T-axis). Used by some
///   unit tests that record steps one at a time. The dispatcher stacks
///   them into the same `[B, T_log, ...]` shape before kernel dispatch.
///
/// The two shapes are detected by `ndim` on the first entry's delta
/// tensor (3 → per-step, 4 → T-axis).
public func stateReplayUpdate(
    state: MLXArray,
    deltaLog: [[MLXArray]],
    acceptedPrefix k: Int,
    mask: MLXArray? = nil
) -> MLXArray {
    precondition(!deltaLog.isEmpty, "delta log must be non-empty")

    let Hv = state.dim(1)
    let Dv = state.dim(2)
    let Dk = state.dim(3)
    let Hk = Hv  // delta log stores GQA-expanded k, so Hk == Hv in this kernel

    // Detect record format: T-axis (4D delta) vs per-step (3D delta).
    let isBatched = deltaLog[0][0].ndim == 4

    let deltaBuf: MLXArray
    let kBuf: MLXArray
    let gBuf: MLXArray
    let T_log: Int

    if isBatched && deltaLog.count == 1 {
        // Fast path: a single per-round entry with T-axis already
        // present. No stacking / transposing needed.
        deltaBuf = deltaLog[0][0]  // [B, T, Hv, Dv]
        kBuf     = deltaLog[0][1]  // [B, T, Hv, Dk]
        gBuf     = deltaLog[0][2]  // [B, T, Hv]
        T_log = deltaBuf.dim(1)
    } else {
        // Legacy path: per-step entries — stack along a new axis, move
        // to position 1 so layout is `[B, T_log, Hv, *]`. Also handles
        // the multi-batched case (multiple per-round entries) by
        // concatenating along the existing T-axis.
        let axis = isBatched ? 1 : 0
        let deltaParts = isBatched
            ? deltaLog.map { $0[0] }
            : deltaLog.map { $0[0][.newAxis, .ellipsis] }
        let kParts = isBatched
            ? deltaLog.map { $0[1] }
            : deltaLog.map { $0[1][.newAxis, .ellipsis] }
        let gParts = isBatched
            ? deltaLog.map { $0[2] }
            : deltaLog.map { $0[2][.newAxis, .ellipsis] }
        if isBatched {
            deltaBuf = concatenated(deltaParts, axis: 1)
            kBuf     = concatenated(kParts,     axis: 1)
            gBuf     = concatenated(gParts,     axis: 1)
        } else {
            deltaBuf = concatenated(deltaParts, axis: 0).transposed(1, 0, 2, 3)
            kBuf     = concatenated(kParts,     axis: 0).transposed(1, 0, 2, 3)
            gBuf     = concatenated(gParts,     axis: 0).transposed(1, 0, 2)
        }
        T_log = deltaBuf.dim(1)
        _ = axis
    }

    precondition(
        k >= 0 && k <= T_log,
        "acceptedPrefix (\(k)) out of range [0, \(T_log)]")

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
    // outputs = [y, state_out, delta_log]
    return (outputs[0], outputs[1], outputs[2])
}
