// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// TriAttention V3 cell scoring — Swift / MLX port of
// vllm-turboquant/vllm/v1/attention/triattention/scoring.py.
//
// Math (per cell, per kv-head block):
//   c_r, c_i, c_abs = calibration centers for this (layer, kv_head).
//   k_r = K[..., :fc], k_i = K[..., fc:n_rot]   (RoPE half-layout)
//   A_f = c_r[f] * k_r[f] + c_i[f] * k_i[f]
//   B_f = c_i[f] * k_r[f] - c_r[f] * k_i[f]
//   k_abs[f] = sqrt(k_r[f]^2 + k_i[f]^2 + 1e-8)
//   c_mag[f] = sqrt(c_r[f]^2 + c_i[f]^2 + 1e-8)
//   cb_delta[f] = c_abs[f] - c_mag[f]
//   cos_sum[f] = mean over offsets o of cos((max_pos + o) * omega[f])
//   sin_sum[f] = mean over offsets o of sin((max_pos + o) * omega[f])
//   acc_cell = sum_f (A_f * cos_sum[f] - B_f * sin_sum[f])
//   ext_cell = sum_f cb_delta[f] * k_abs[f]
//   score[cell] = sum over heads of (acc_cell + ext_cell)
//
// HIGHER score = evict first. The trig formula measures K's
// orthogonality to the calibration center (not alignment) — orthogonal
// cells contribute least to attention, so evicting them is cheapest.
import Foundation
import MLX

public enum TriAttentionScoring {

    /// Precompute (cos_sum, sin_sum) of shape `[fc]`, averaged over offsets.
    fileprivate static func precomputeOffsetSums(
        omega: MLXArray, offsets: MLXArray, maxPos: Int
    ) -> (MLXArray, MLXArray) {
        // t_vals[o] = max_pos + offsets[o]   shape [n_off]
        let tVals = (offsets + Float(maxPos)).asType(.float32)
        let omegaF = omega.asType(.float32)
        // phase[f, o] = t_vals[o] * omega[f]   shape [fc, n_off]
        let phase = MLX.expandedDimensions(tVals, axis: 0)
            * MLX.expandedDimensions(omegaF, axis: 1)
        let cosSum = MLX.cos(phase).mean(axis: 1)  // [fc]
        let sinSum = MLX.sin(phase).mean(axis: 1)  // [fc]
        return (cosSum, sinSum)
    }

    /// Score every cell in `K`, returning a `[seqLen]` fp32 tensor.
    /// Evicted positions (where `validMask` is false) and recent-window
    /// positions (`>= windowThr`) get score 0 — they're filtered at the
    /// policy stage anyway, but the explicit zero keeps things tidy.
    public static func scoreCells(
        K: MLXArray,           // [seq_len, n_kv_heads, head_dim] f32
        centerReal: MLXArray,  // [n_kv_heads, fc] f32
        centerImag: MLXArray,  // [n_kv_heads, fc] f32
        centerAbs: MLXArray,   // [n_kv_heads, fc] f32
        omega: MLXArray,       // [fc]
        offsets: MLXArray,     // [n_off]
        maxPos: Int,
        validMask: MLXArray,   // [seq_len] bool
        windowThr: Int,
        nRot: Int
    ) -> MLXArray {
        let seqLen = K.dim(0)
        let fc = nRot / 2

        // Split K into real/imag halves on the rotated dims.
        let kR = K[.ellipsis, ..<fc]
        let kI = K[.ellipsis, fc..<nRot]
        let kAbs = MLX.sqrt(kR * kR + kI * kI + 1e-8)

        // Broadcast center [H, fc] -> [1, H, fc].
        let cR = MLX.expandedDimensions(centerReal, axis: 0)
        let cI = MLX.expandedDimensions(centerImag, axis: 0)
        // (cr + i ci)(kr - i ki) = (cr*kr + ci*ki) + i (ci*kr - cr*ki)
        let A = cR * kR + cI * kI
        let B = cI * kR - cR * kI

        let cMag = MLX.sqrt(centerReal * centerReal + centerImag * centerImag + 1e-8)
        let cbDelta = MLX.expandedDimensions(centerAbs - cMag, axis: 0)  // [1, H, fc]

        let (cosSum, sinSum) = precomputeOffsetSums(
            omega: omega, offsets: offsets, maxPos: maxPos)
        let cosSumB = MLX.expandedDimensions(
            MLX.expandedDimensions(cosSum, axis: 0), axis: 0)  // [1, 1, fc]
        let sinSumB = MLX.expandedDimensions(
            MLX.expandedDimensions(sinSum, axis: 0), axis: 0)

        let acc = (A * cosSumB - B * sinSumB).sum(axis: -1)  // [T, H]
        let ext = (cbDelta * kAbs).sum(axis: -1)             // [T, H]
        let perHead = acc + ext

        // Sum across kv-heads.
        var score = perHead.sum(axis: -1)  // [T]

        // Mask invalid + window-protected.
        let zero = MLXArray.zeros(score.shape, dtype: .float32)
        score = MLX.where(validMask, score, zero)
        if windowThr > 0 {
            let positions = MLXArray(0..<Int32(seqLen))
            score = MLX.where(positions .< Int32(windowThr), score, zero)
        }
        return score
    }
}
