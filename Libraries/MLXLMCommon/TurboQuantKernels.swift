// Copyright © 2026 Eric Kryski. TurboQuant Metal kernels for compressed-domain attention.
//
// Single Metal kernel computes Q×K attention scores directly from packed
// codebook indices, skipping full dequantization. Pre-rotated queries
// eliminate the per-token inverse rotation.
//
// Key formula: score[t] = norm[t] * Σ_j q_rot[j] * codebook[idx[t,j]]
// where q_rot = Π · q (pre-rotated once) and idx are b-bit packed indices.

import Foundation
import MLX

// MARK: - Metal Kernel Source

enum TurboQuantMetalKernels {

    /// Scoring kernel: computes attention scores from packed codebook indices.
    ///
    /// Each SIMD group (32 threads) handles one (query, key_token) pair.
    /// Codebook (8-16 entries) is loaded into thread-local registers.
    /// Bit unpacking + codebook lookup + dot product all happen in-register.
    ///
    /// Grid: (32, totalQueries, tokenCount)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: Bits, Dim, PackedWidth, token_count, repeat_count
    static let scoreKernelSource = """
    // Template constants injected by MLXFast JIT
    constexpr uint MASK = (1u << Bits) - 1u;
    constexpr uint LEVELS = 1u << Bits;

    uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
    uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
    uint k_idx = thread_position_in_grid.z;     // key token index

    // Map query head to KV head (GQA)
    uint kv_idx = q_idx / repeat_count;

    // Pointers
    const device float* q_ptr = q_rot + q_idx * Dim;
    const device uint32_t* packed_ptr = packed + kv_idx * token_count * PackedWidth + k_idx * PackedWidth;
    float norm_val = norms[kv_idx * token_count + k_idx];

    // Load codebook into registers (small: 4-16 entries)
    float cb[LEVELS];
    for (uint i = 0; i < LEVELS; i++) {
        cb[i] = codebook[i];
    }

    // Parallel dot product: each lane handles dims [lane, lane+32, lane+64, ...]
    float acc = 0.0f;
    for (uint d = lane; d < Dim; d += 32) {
        // Unpack b-bit index for dimension d
        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint value = (packed_ptr[word_idx] >> shift);

        // Handle bits that spill across uint32 word boundary
        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
        }
        value &= MASK;

        // Codebook lookup + accumulate dot product
        acc += q_ptr[d] * cb[value];
    }

    // SIMD reduction across 32 lanes
    acc = simd_sum(acc);

    // Lane 0 writes final score (scaled by stored norm)
    if (thread_index_in_simdgroup == 0) {
        scores[q_idx * token_count + k_idx] = acc * norm_val;
    }
    """

    /// Value aggregation kernel: weighted sum of codebook-quantized values.
    ///
    /// output[d] = Σ_t weights[t] * norm[t] * codebook[val_idx[t,d]]
    /// Result is in rotated space — caller applies inverse rotation.
    ///
    /// Grid: (32, totalHeads, ceil(Dim/32))
    /// Threadgroup: (32, 1, 1)
    static let valueKernelSource = """
    constexpr uint MASK = (1u << Bits) - 1u;
    constexpr uint LEVELS = 1u << Bits;

    uint lane = thread_position_in_grid.x;
    uint head_idx = thread_position_in_grid.y;
    uint dim_block = thread_position_in_grid.z;

    uint d = dim_block * 32 + lane;
    if (d >= Dim) return;

    uint kv_head = head_idx / repeat_count;

    // Load codebook
    float cb[LEVELS];
    for (uint i = 0; i < LEVELS; i++) {
        cb[i] = codebook[i];
    }

    float acc = 0.0f;
    for (uint t = 0; t < (uint)token_count; t++) {
        float w = weights[head_idx * token_count + t];
        if (w == 0.0f) continue;

        float norm_val = norms[kv_head * token_count + t];
        const device uint32_t* packed_ptr = packed + kv_head * token_count * PackedWidth + t * PackedWidth;

        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint value = (packed_ptr[word_idx] >> shift);

        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
        }
        value &= MASK;

        acc += w * norm_val * cb[value];
    }

    output[head_idx * Dim + d] = acc;
    """
}

// MARK: - Kernel Dispatch Wrappers

public enum TurboQuantKernelOps {

    // Kernel caches
    nonisolated(unsafe) private static var scoreKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var valueKernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    /// Compute Q×K attention scores from packed codebook indices.
    ///
    /// - Parameters:
    ///   - rotatedQueries: Pre-rotated queries [totalQ, D] (already scaled)
    ///   - packed: Packed key indices [totalKVHeads, T, PackedWidth] uint32
    ///   - norms: Key norms [totalKVHeads, T] float32
    ///   - codebook: Centroids [2^bits] float32
    ///   - tokenCount: Number of cached tokens
    ///   - repeatCount: GQA repeat factor (nQHeads / nKVHeads)
    ///   - bits: MSE bit-width
    ///   - dim: Vector dimension
    /// - Returns: Scores [totalQ, T] float32
    public static func mseScore(
        rotatedQueries: MLXArray,
        packed: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "\(bits)_\(dim)"

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = scoreKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_score_\(bits)_\(dim)",
                inputNames: ["q_rot", "packed", "norms", "codebook"],
                outputNames: ["scores"],
                source: TurboQuantMetalKernels.scoreKernelSource,
                ensureRowContiguous: true
            )
            lock.lock()
            scoreKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let totalQ = rotatedQueries.dim(0)

        return kernel(
            [rotatedQueries, packed, norms, codebook],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
                ("token_count", tokenCount), ("repeat_count", repeatCount),
            ],
            grid: (32, totalQ, tokenCount),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, tokenCount]],
            outputDTypes: [.float32]
        )[0]
    }

    /// Compute weighted sum of values from packed codebook indices.
    ///
    /// Result is in ROTATED space — caller must apply inverse rotation.
    ///
    /// - Returns: [totalHeads, D] float32 (rotated space)
    public static func mseWeightedSum(
        weights: MLXArray,
        packed: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "\(bits)_\(dim)"

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = valueKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_value_\(bits)_\(dim)",
                inputNames: ["weights", "packed", "norms", "codebook"],
                outputNames: ["output"],
                source: TurboQuantMetalKernels.valueKernelSource,
                ensureRowContiguous: true
            )
            lock.lock()
            valueKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let totalHeads = weights.dim(0)
        let dimBlocks = (dim + 31) / 32

        return kernel(
            [weights, packed, norms, codebook],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
                ("token_count", tokenCount), ("repeat_count", repeatCount),
            ],
            grid: (32, totalHeads, dimBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalHeads, dim]],
            outputDTypes: [.float32]
        )[0]
    }
}
