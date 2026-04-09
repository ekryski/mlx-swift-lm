import Foundation
import MLX

/// Fused RMSNorm + RoPE kernel for transformer attention.
///
/// Combines two Metal dispatches (RMSNorm + RoPE) into a single kernel launch.
/// Each threadgroup processes one row: one (batch, position, head) combination.
///
/// Algorithm:
/// 1. Compute sum(x²) via SIMD reduction for RMSNorm denominator
/// 2. For each rotation pair (i, i+halfDim):
///    - Apply weight scaling: `normed = x[i] * w[i] * rsqrt(sum_sq/dim + eps)`
///    - Compute theta from invFreqs: `theta = position * invFreqs[i]`
///    - Apply rotation: `out[i] = normed_a * cos(theta) - normed_b * sin(theta)`
///
/// Operates on pre-transpose layout `[B, L, nHeads, headDim]`.
/// For unrotated dimensions (invFreqs[i] = 0), rotation is identity (cos=1, sin=0).
public class FusedNormRoPEKernel {

    /// Inverse frequencies for RoPE rotation. For unrotated dims, value is 0 (identity rotation).
    public let invFreqs: MLXArray

    public init(invFreqs: MLXArray) {
        self.invFreqs = invFreqs
    }

    private var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private func kernel(headDim: Int, eps: Float, dtype: DType) -> MLXFast.MLXFastKernel {
        let key = "\(headDim)_\(dtype)"
        if let cached = kernels[key] { return cached }

        let halfDim = headDim / 2

        let source = """
            constexpr float eps = \(eps)f;
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;

            int offset_val = offset_buf[0];
            int seqLen_val = seqLen_buf[0];

            // Compute sequence position from row index: [B, L, nHeads, D]
            // row = b * L * nHeads + l * nHeads + h
            uint l = (row / nHeads) % uint(seqLen_val);
            float pos = float(offset_val + int(l));

            // Step 1: Load pair elements and compute sum(x²) for RMSNorm
            const device T* row_ptr = x + row * headDim;
            uint idx1 = tid;           // first half [0, halfDim)
            uint idx2 = tid + halfDim; // second half [halfDim, headDim)

            float v1 = float(row_ptr[idx1]);
            float v2 = float(row_ptr[idx2]);
            float sum_sq = v1 * v1 + v2 * v2;

            // SIMD reduction for sum of squares
            sum_sq = simd_sum(sum_sq);

            // Cross-SIMD-group reduction via shared memory
            threadgroup float shared_sums[32];
            uint sg_id = tid / 32;
            uint sg_lane = tid % 32;
            if (sg_lane == 0) shared_sums[sg_id] = sum_sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint num_simd_groups = (halfDim + 31) / 32;
            if (sg_id == 0) {
                float s = (sg_lane < num_simd_groups) ? shared_sums[sg_lane] : 0.0f;
                s = simd_sum(s);
                if (sg_lane == 0) shared_sums[0] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float inv_norm = metal::precise::rsqrt(shared_sums[0] / float(headDim) + eps);

            // Step 2: Apply RMSNorm weight + RoPE rotation
            float n1 = v1 * float(w[idx1]) * inv_norm;
            float n2 = v2 * float(w[idx2]) * inv_norm;

            // RoPE: theta = position * invFreq
            float inv_freq = float(inv_freqs[tid]);
            float theta = pos * inv_freq;
            float cos_t = metal::fast::cos(theta);
            float sin_t = metal::fast::sin(theta);

            device T* out_ptr = out + row * headDim;
            out_ptr[idx1] = T(n1 * cos_t - n2 * sin_t);
            out_ptr[idx2] = T(n1 * sin_t + n2 * cos_t);
            """

        let k = MLXFast.metalKernel(
            name: "fused_norm_rope_\(key)",
            inputNames: ["x", "w", "inv_freqs", "offset_buf", "seqLen_buf"],
            outputNames: ["out"],
            source: source
        )
        kernels[key] = k
        return k
    }

    /// Apply fused RMSNorm + RoPE.
    ///
    /// - Parameters:
    ///   - x: Input tensor `[B, L, nHeads, headDim]` (pre-transpose layout)
    ///   - weight: RMSNorm weight `[headDim]`
    ///   - invFreqs: Inverse frequencies `[headDim/2]`. For unrotated dims, use 0.
    ///   - eps: RMSNorm epsilon
    ///   - offset: Token position offset (cache.offset)
    ///   - nHeads: Number of attention heads (for extracting `l` from row index)
    /// - Returns: Normed + rotated tensor `[B, L, nHeads, headDim]`
    public func callAsFunction(
        _ x: MLXArray,
        weight: MLXArray,
        eps: Float,
        offset: Int,
        nHeads: Int
    ) -> MLXArray {
        let shape = x.shape
        let headDim = shape.last!
        let halfDim = headDim / 2
        let seqLen = shape[shape.count - 3]  // L from [B, L, nHeads, headDim]
        let totalRows = shape.dropLast().reduce(1, *)

        let k = kernel(headDim: headDim, eps: eps, dtype: x.dtype)

        let results = k(
            [x.reshaped(-1, headDim), weight, invFreqs,
             MLXArray([Int32(offset)]), MLXArray([Int32(seqLen)])],
            template: [
                ("T", x.dtype),
                ("headDim", headDim),
                ("halfDim", halfDim),
                ("nHeads", nHeads),
            ],
            grid: (totalRows * halfDim, 1, 1),
            threadGroup: (halfDim, 1, 1),
            outputShapes: [[totalRows, headDim]],
            outputDTypes: [x.dtype],
            verbose: ProcessInfo.processInfo.environment["FUSED_KERNEL_VERBOSE"] == "1"
        )
        return results[0].reshaped(shape)
    }
}
