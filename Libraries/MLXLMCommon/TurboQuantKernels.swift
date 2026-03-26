// Copyright © 2026 Eric Kryski. TurboQuant Metal kernels for compressed-domain attention.
//
// Ported from mlx-vlm PR #858 (Blaizzy/mlx-vlm, branch pc/turbo-quant).
// Uses MLXFast.metalKernel() JIT compilation API.
//
// Key insight: compute attention scores directly from packed indices + codebook,
// skipping full dequantization. Pre-rotate queries once before scoring all keys.

import Foundation
import MLX

// MARK: - Metal Kernel Source Strings

/// Metal kernel source code for TurboQuant compressed-domain operations.
///
/// Template parameters (injected by MLXFast JIT):
///   Bits: bit-width per codebook index (1-4)
///   Dim: vector dimension (e.g., 128)
///   PackedWidth: number of uint32 words per packed vector
///   token_count: number of cached KV tokens
///   repeat_count: GQA repeat factor (nQHeads / nKVHeads)
enum TurboQuantMetalKernels {

    /// MSE score kernel — computes Q×K attention scores from packed codebook indices.
    ///
    /// For each (query, cached_key) pair:
    ///   score = sum_d(rotated_query[d] * codebook[indices[d]]) * norm
    ///
    /// Grid: (32, totalQueries, tokenCount)
    /// Threadgroup: (32, 1, 1) — one SIMD group per (query, key) pair
    static let mseScoreSource = """
    constexpr uint MASK = (1u << Bits) - 1u;

    uint lane = thread_position_in_grid.x;
    uint q_idx = thread_position_in_grid.y;
    uint k_idx = thread_position_in_grid.z;

    const device float* q_ptr = q_rot + q_idx * Dim;
    uint kv_idx = q_idx / repeat_count;
    const device uint32_t* packed_ptr = packed + kv_idx * token_count * PackedWidth + k_idx * PackedWidth;
    float norm_val = norms[kv_idx * token_count + k_idx];

    float acc = 0.0f;
    for (uint d = lane; d < Dim; d += 32) {
        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint offset = bit_offset % 32;
        uint value = (packed_ptr[word_idx] >> offset);

        int spill = (int)offset + (int)Bits - 32;
        if (spill > 0) {
            value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
        }
        value &= MASK;

        acc += q_ptr[d] * codebook[value];
    }

    acc = simd_sum(acc);

    if (thread_index_in_simdgroup == 0) {
        scores[q_idx * token_count + k_idx] = acc * norm_val;
    }
    """

    /// MSE weighted sum kernel — value aggregation in compressed domain.
    ///
    /// Computes: output[d] = sum_t(weights[t] * norm[t] * codebook[indices[t,d]])
    /// Result is still in rotated space — caller applies inverse rotation.
    ///
    /// Grid: (32, totalHeads, ceil(Dim/32))
    /// Threadgroup: (32, 1, 1)
    static let mseWeightedSumSource = """
    constexpr uint MASK = (1u << Bits) - 1u;

    uint lane = thread_position_in_grid.x;
    uint head_idx = thread_position_in_grid.y;
    uint dim_block = thread_position_in_grid.z;

    uint d = dim_block * 32 + lane;
    if (d >= Dim) return;

    uint kv_head = head_idx / repeat_count;

    float acc = 0.0f;
    for (uint t = 0; t < (uint)token_count; t++) {
        float w = weights[head_idx * token_count + t];
        if (w == 0.0f) continue;

        float norm_val = norms[kv_head * token_count + t];
        const device uint32_t* packed_ptr = packed + kv_head * token_count * PackedWidth + t * PackedWidth;

        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint offset = bit_offset % 32;
        uint value = (packed_ptr[word_idx] >> offset);

        int spill = (int)offset + (int)Bits - 32;
        if (spill > 0) {
            value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
        }
        value &= MASK;

        acc += w * norm_val * codebook[value];
    }

    output[head_idx * Dim + d] = acc;
    """
}

// MARK: - Kernel Wrappers

/// Compiled Metal kernel instances for TurboQuant operations.
/// Kernels are JIT-compiled on first use and cached per (bits, dim) combination.
public enum TurboQuantKernelOps {

    /// Cache of compiled score kernels keyed by (bits, dim)
    nonisolated(unsafe) private static var scoreKernelCache: [String: MLXFast.MLXFastKernel] = [:]
    /// Cache of compiled weighted sum kernels keyed by (bits, dim)
    nonisolated(unsafe) private static var wsumKernelCache: [String: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    /// Compile or retrieve the MSE score kernel.
    private static func getScoreKernel(bits: Int, dim: Int) -> MLXFast.MLXFastKernel {
        let key = "\(bits)_\(dim)"
        lock.lock()
        if let cached = scoreKernelCache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let kernel = MLXFast.metalKernel(
            name: "turbo_mse_score_\(bits)_\(dim)",
            inputNames: ["q_rot", "packed", "norms", "codebook"],
            outputNames: ["scores"],
            source: TurboQuantMetalKernels.mseScoreSource,
            ensureRowContiguous: true
        )

        lock.lock()
        scoreKernelCache[key] = kernel
        lock.unlock()
        return kernel
    }

    /// Compile or retrieve the MSE weighted sum kernel.
    private static func getWSumKernel(bits: Int, dim: Int) -> MLXFast.MLXFastKernel {
        let key = "\(bits)_\(dim)"
        lock.lock()
        if let cached = wsumKernelCache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let kernel = MLXFast.metalKernel(
            name: "turbo_mse_wsum_\(bits)_\(dim)",
            inputNames: ["weights", "packed", "norms", "codebook"],
            outputNames: ["output"],
            source: TurboQuantMetalKernels.mseWeightedSumSource,
            ensureRowContiguous: true
        )

        lock.lock()
        wsumKernelCache[key] = kernel
        lock.unlock()
        return kernel
    }

    /// Compute attention scores between pre-rotated queries and MSE-encoded keys.
    ///
    /// - Parameters:
    ///   - rotatedQueries: Pre-rotated queries [totalQ, D] float32
    ///   - packedIndices: Packed key indices [totalKVHeads, T_kv, PackedWidth] uint32
    ///   - norms: Key norms [totalKVHeads, T_kv] float32
    ///   - codebook: Centroid values [2^bits] float32
    ///   - tokenCount: Number of cached tokens
    ///   - repeatCount: GQA repeat factor
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension
    /// - Returns: Scores [totalQ, T_kv] float32
    public static func mseScore(
        rotatedQueries: MLXArray,
        packedIndices: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let kernel = getScoreKernel(bits: bits, dim: dim)

        let totalQ = rotatedQueries.dim(0)

        let result = kernel(
            [rotatedQueries, packedIndices, norms, codebook],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
                ("token_count", tokenCount), ("repeat_count", repeatCount),
            ],
            grid: (32, totalQ, tokenCount),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, tokenCount]],
            outputDTypes: [.float32]
        )

        return result[0]
    }

    /// Compute weighted sum of MSE-encoded values using attention weights.
    ///
    /// - Parameters:
    ///   - weights: Attention weights [totalHeads, T_kv] float32
    ///   - packedIndices: Packed value indices [totalKVHeads, T_kv, PackedWidth] uint32
    ///   - norms: Value norms [totalKVHeads, T_kv] float32
    ///   - codebook: Centroid values [2^bits] float32
    ///   - tokenCount: Number of cached tokens
    ///   - repeatCount: GQA repeat factor
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension
    /// - Returns: Weighted sum [totalHeads, D] float32 (in rotated space)
    public static func mseWeightedSum(
        weights: MLXArray,
        packedIndices: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let kernel = getWSumKernel(bits: bits, dim: dim)

        let totalHeads = weights.dim(0)
        let dimBlocks = (dim + 31) / 32

        let result = kernel(
            [weights, packedIndices, norms, codebook],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
                ("token_count", tokenCount), ("repeat_count", repeatCount),
            ],
            grid: (32, totalHeads, dimBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalHeads, dim]],
            outputDTypes: [.float32]
        )

        return result[0]
    }
}
