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
    /// Fused quantize+pack kernel — finds nearest codebook centroid and packs indices.
    ///
    /// For each (row, dimension):
    ///   1. Find nearest codebook centroid: argmin_c |rotated[d] - codebook[c]|
    ///   2. Pack the index into the output uint32 word at the correct bit offset
    ///
    /// Uses atomic_fetch_or for thread-safe bit packing within shared uint32 words.
    ///
    /// Grid: (Dim, totalRows, 1) — one thread per (row, dimension)
    /// Threadgroup: (min(Dim, 256), 1, 1)
    ///
    /// Inputs:
    ///   rotated: pre-rotated unit vectors [totalRows, Dim] float32
    ///   codebook: centroid values [2^Bits] float32
    ///
    /// Output:
    ///   packed: bit-packed indices [totalRows, PackedWidth] uint32
    static let fusedQuantizePackSource = """
    constexpr uint MASK = (1u << Bits) - 1u;
    constexpr uint LEVELS = 1u << Bits;

    uint d = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    if (d >= Dim) return;

    float val = rotated[row * Dim + d];

    // Find nearest codebook centroid
    uint best_idx = 0;
    float best_dist = fabs(val - codebook[0]);
    for (uint c = 1; c < LEVELS; c++) {
        float dist = fabs(val - codebook[c]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = c;
        }
    }

    // Pack into output: atomic OR into the correct uint32 word
    uint bit_offset = d * Bits;
    uint word_idx = bit_offset / 32;
    uint offset = bit_offset % 32;

    // Primary word
    uint bits_to_write = (best_idx & MASK) << offset;
    atomic_fetch_or_explicit(
        (device atomic_uint*)(packed + row * PackedWidth + word_idx),
        bits_to_write,
        memory_order_relaxed);

    // Handle spill to next word (when bits cross a 32-bit boundary)
    int spill = (int)offset + (int)Bits - 32;
    if (spill > 0) {
        uint high_bits = (best_idx & MASK) >> ((uint)Bits - (uint)spill);
        atomic_fetch_or_explicit(
            (device atomic_uint*)(packed + row * PackedWidth + word_idx + 1),
            high_bits,
            memory_order_relaxed);
    }
    """

    /// Flash attention kernel with in-kernel TurboQuant dequantization.
    ///
    /// Based on MLX's sdpa_vector pattern: one threadgroup per (batch×head, query_pos).
    /// Reads packed codebook indices directly — never materializes full float32 KV.
    ///
    /// For each query position:
    ///   1. Load query (already pre-rotated for keys)
    ///   2. Tile over cached tokens in blocks of BN=32:
    ///      a. Load packed key indices, dequant via codebook, compute Q·K dot product
    ///      b. Online softmax: update running max and sum_exp
    ///   3. Second pass (or fused): accumulate weighted values
    ///
    /// Grid: (totalQHeads, 1, 1)
    /// Threadgroup: (32, 1, 1) — one SIMD group per query head
    ///
    /// Inputs:
    ///   q_rot: pre-rotated queries [totalQHeads, Dim] float32
    ///   k_packed: packed key indices [totalKVHeads, T_kv, PackedWidth] uint32
    ///   k_norms: key norms [totalKVHeads, T_kv] float32
    ///   k_codebook: key centroids [2^KeyBits] float32
    ///   v_packed: packed value indices [totalKVHeads, T_kv, ValPackedWidth] uint32
    ///   v_norms: value norms [totalKVHeads, T_kv] float32
    ///   v_codebook: value centroids [2^ValBits] float32
    ///
    /// Output:
    ///   output: [totalQHeads, Dim] float32 (in rotated value space — caller inverse-rotates)
    static let turboFlashAttentionSource = """
    constexpr uint K_MASK = (1u << KeyBits) - 1u;
    constexpr uint V_MASK = (1u << ValBits) - 1u;
    constexpr uint BN = 32u;  // tile size for KV tokens

    uint q_head = thread_position_in_grid.x;  // which query head
    uint lane = thread_index_in_simdgroup;     // 0-31

    uint kv_head = q_head / repeat_count;

    // Load query vector into registers (each lane loads every 32nd element)
    float q_reg[8];  // max Dim/32 = 256/32 = 8 elements per lane
    for (uint i = 0; i < (Dim + 31) / 32; i++) {
        uint d = lane + i * 32;
        q_reg[i] = (d < Dim) ? q_rot[q_head * Dim + d] : 0.0f;
    }

    // Online softmax state
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // Output accumulator (in rotated value space)
    float out_reg[8];
    for (uint i = 0; i < (Dim + 31) / 32; i++) {
        out_reg[i] = 0.0f;
    }

    // Tile over all cached tokens
    for (uint t_base = 0; t_base < (uint)token_count; t_base += BN) {
        uint t_end = min(t_base + BN, (uint)token_count);

        // For each token in this tile
        for (uint t = t_base; t < t_end; t++) {
            // --- Compute Q·K score ---
            float dot = 0.0f;
            const device uint32_t* k_ptr = k_packed + kv_head * token_count * KeyPackedWidth + t * KeyPackedWidth;
            float k_norm = k_norms[kv_head * token_count + t];

            for (uint i = 0; i < (Dim + 31) / 32; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                // Unpack key index for dimension d
                uint bit_off = d * KeyBits;
                uint word = bit_off / 32;
                uint off = bit_off % 32;
                uint idx = (k_ptr[word] >> off);
                int spill = (int)off + (int)KeyBits - 32;
                if (spill > 0) {
                    idx |= (k_ptr[word + 1] << ((uint)KeyBits - (uint)spill));
                }
                idx &= K_MASK;

                float k_val = k_codebook[idx] * k_norm;
                dot += q_reg[i] * k_val;
            }

            // SIMD reduce the dot product
            dot = simd_sum(dot);

            // Queries are pre-scaled by caller (q *= scale)
            float score = dot;

            // --- Online softmax update ---
            float old_max = max_score;
            max_score = max(max_score, score);
            float factor = exp(old_max - max_score);
            sum_exp = sum_exp * factor + exp(score - max_score);

            // Rescale existing output accumulator
            for (uint i = 0; i < (Dim + 31) / 32; i++) {
                out_reg[i] *= factor;
            }

            // --- Accumulate weighted value ---
            float weight = exp(score - max_score);
            const device uint32_t* v_ptr = v_packed + kv_head * token_count * ValPackedWidth + t * ValPackedWidth;
            float v_norm = v_norms[kv_head * token_count + t];

            for (uint i = 0; i < (Dim + 31) / 32; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                // Unpack value index
                uint bit_off = d * ValBits;
                uint word = bit_off / 32;
                uint off = bit_off % 32;
                uint idx = (v_ptr[word] >> off);
                int spill2 = (int)off + (int)ValBits - 32;
                if (spill2 > 0) {
                    idx |= (v_ptr[word + 1] << ((uint)ValBits - (uint)spill2));
                }
                idx &= V_MASK;

                float v_val = v_codebook[idx] * v_norm;
                out_reg[i] += weight * v_val;
            }
        }
    }

    // Normalize by sum_exp and write output
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint i = 0; i < (Dim + 31) / 32; i++) {
        uint d = lane + i * 32;
        if (d < Dim) {
            output[q_head * Dim + d] = out_reg[i] * inv_sum;
        }
    }
    """

    /// Fused norm+rotate kernel using Walsh-Hadamard Transform butterfly.
    ///
    /// Computes: norms = ||v||, unit = v / ||v||, rotated = WHT(unit)
    /// The WHT butterfly is done in shared memory within each threadgroup.
    ///
    /// Grid: (Dim, totalRows, 1)
    /// Threadgroup: (Dim, 1, 1) — full dim must fit in one threadgroup
    ///
    /// Inputs:
    ///   vectors: raw input vectors [totalRows, Dim] float32
    ///   signs: random ±1 signs [Dim] float32
    ///
    /// Outputs:
    ///   rotated: WHT-rotated unit vectors [totalRows, Dim] float32
    ///   norms: L2 norms [totalRows] float32
    static let fusedNormWHTSource = """
    uint d = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;

    if (d >= Dim) return;

    // Load value and compute partial norm (squared)
    float val = vectors[row * Dim + d];
    float sq = val * val;

    // Use SIMD reduction for norm computation
    // First, sum within SIMD group
    float simd_sq = simd_sum(sq);

    // Store partial sums and reduce across SIMD groups via threadgroup memory
    threadgroup float shared[256];  // max 256 threads
    shared[d] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes full norm
    float norm = 0.0f;
    if (d == 0) {
        for (uint i = 0; i < Dim; i++) {
            norm += shared[i];
        }
        norm = sqrt(norm);
        if (norm < 1e-8f) norm = 1e-8f;
        norms[row] = norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    norm = norms[row];

    // Normalize to unit vector and apply random signs
    float unit_val = (val / norm) * signs[d];
    shared[d] = unit_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Walsh-Hadamard butterfly in shared memory
    for (uint halfLen = 1; halfLen < Dim; halfLen <<= 1) {
        uint pair_idx = d / (halfLen * 2);
        uint within = d % (halfLen * 2);
        uint base = pair_idx * halfLen * 2;

        float a_val, b_val;
        if (within < halfLen) {
            a_val = shared[base + within];
            b_val = shared[base + within + halfLen];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (within < halfLen) {
            shared[base + within] = a_val + b_val;
            shared[base + within + halfLen] = a_val - b_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write normalized result
    rotated[row * Dim + d] = shared[d] * (1.0f / sqrt((float)Dim));
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
    /// Cache of compiled quantize+pack kernels
    nonisolated(unsafe) private static var qpKernelCache: [String: MLXFast.MLXFastKernel] = [:]
    /// Cache of compiled norm+WHT kernels
    nonisolated(unsafe) private static var nwKernelCache: [String: MLXFast.MLXFastKernel] = [:]
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

    // MARK: - Flash Attention Kernel

    /// Cache of compiled flash attention kernels
    nonisolated(unsafe) private static var flashKernelCache: [String: MLXFast.MLXFastKernel] = [:]

    /// Flash attention with in-kernel TurboQuant dequantization.
    ///
    /// Reads packed codebook indices directly — never materializes full float32 KV.
    /// Based on MLX's sdpa_vector pattern with online softmax.
    ///
    /// - Parameters:
    ///   - rotatedQueries: Pre-rotated queries [totalQHeads, D] float32
    ///   - keyPacked: Packed key indices [totalKVHeads, T_kv, KeyPackedWidth] uint32
    ///   - keyNorms: Key norms [totalKVHeads, T_kv] float32
    ///   - keyCodebook: Key centroids [2^keyBits] float32
    ///   - valPacked: Packed value indices [totalKVHeads, T_kv, ValPackedWidth] uint32
    ///   - valNorms: Value norms [totalKVHeads, T_kv] float32
    ///   - valCodebook: Value centroids [2^valBits] float32
    ///   - scale: Attention scale (1/√d)
    ///   - tokenCount: Number of cached tokens
    ///   - repeatCount: GQA repeat factor
    ///   - keyBits: Key quantization bits
    ///   - valBits: Value quantization bits
    ///   - dim: Head dimension
    /// - Returns: Output [totalQHeads, D] float32 (in rotated value space)
    public static func turboFlashAttention(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray,
        keyNorms: MLXArray,
        keyCodebook: MLXArray,
        valPacked: MLXArray,
        valNorms: MLXArray,
        valCodebook: MLXArray,
        scale: Float,
        tokenCount: Int,
        repeatCount: Int,
        keyBits: Int,
        valBits: Int,
        dim: Int
    ) -> MLXArray {
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valBits)
        let key = "flash_\(keyBits)_\(valBits)_\(dim)"

        lock.lock()
        let kernel: MLXFast.MLXFastKernel
        if let cached = flashKernelCache[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let newKernel = MLXFast.metalKernel(
                name: "turbo_flash_attn_\(keyBits)_\(valBits)_\(dim)",
                inputNames: [
                    "q_rot", "k_packed", "k_norms", "k_codebook",
                    "v_packed", "v_norms", "v_codebook",
                ],
                outputNames: ["output"],
                source: TurboQuantMetalKernels.turboFlashAttentionSource,
                ensureRowContiguous: true
            )
            lock.lock()
            flashKernelCache[key] = newKernel
            lock.unlock()
            kernel = newKernel
        }

        let totalQHeads = rotatedQueries.dim(0)

        let result = kernel(
            [
                rotatedQueries, keyPacked, keyNorms, keyCodebook,
                valPacked, valNorms, valCodebook,
            ],
            template: [
                ("KeyBits", keyBits), ("ValBits", valBits),
                ("Dim", dim), ("KeyPackedWidth", kpw), ("ValPackedWidth", vpw),
                ("token_count", tokenCount), ("repeat_count", repeatCount),
            ],
            grid: (totalQHeads, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQHeads, dim]],
            outputDTypes: [.float32]
        )

        return result[0]
    }

    // MARK: - Fused Encode Kernels

    /// Fused norm + WHT rotation in a single Metal dispatch.
    ///
    /// Computes: norms = ||v||, rotated = WHT(v / ||v||)
    /// WHT butterfly runs entirely in threadgroup shared memory.
    ///
    /// - Parameters:
    ///   - vectors: Input vectors [totalRows, dim] float32
    ///   - signs: Random ±1 signs [dim] float32
    ///   - dim: Vector dimension (must be power of 2, ≤ 256 for threadgroup limit)
    /// - Returns: (rotated [totalRows, dim], norms [totalRows])
    public static func fusedNormWHT(
        vectors: MLXArray,
        signs: MLXArray,
        dim: Int
    ) -> (MLXArray, MLXArray) {
        let key = "nw_\(dim)"
        lock.lock()
        let kernel: MLXFast.MLXFastKernel
        if let cached = nwKernelCache[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let newKernel = MLXFast.metalKernel(
                name: "turbo_norm_wht_\(dim)",
                inputNames: ["vectors", "signs"],
                outputNames: ["rotated", "norms"],
                source: TurboQuantMetalKernels.fusedNormWHTSource,
                atomicOutputs: false
            )
            lock.lock()
            nwKernelCache[key] = newKernel
            lock.unlock()
            kernel = newKernel
        }

        let totalRows = vectors.dim(0)
        let flatSigns = signs.reshaped([dim])

        let result = kernel(
            [vectors, flatSigns],
            template: [("Dim", dim)],
            grid: (dim, totalRows, 1),
            threadGroup: (dim, 1, 1),  // Full dim in one threadgroup for shared memory WHT
            outputShapes: [[totalRows, dim], [totalRows]],
            outputDTypes: [.float32, .float32]
        )

        return (result[0], result[1])
    }

    /// Fused quantize + pack in a single Metal dispatch.
    ///
    /// For each dimension: finds nearest codebook centroid and atomically
    /// packs the index into uint32 output words.
    ///
    /// - Parameters:
    ///   - rotated: Pre-rotated unit vectors [totalRows, dim] float32
    ///   - codebook: Centroid values [2^bits] float32
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension
    /// - Returns: Packed indices [totalRows, packedWidth] uint32
    public static func fusedQuantizePack(
        rotated: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "qp_\(bits)_\(dim)"

        lock.lock()
        let kernel: MLXFast.MLXFastKernel
        if let cached = qpKernelCache[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let newKernel = MLXFast.metalKernel(
                name: "turbo_quantize_pack_\(bits)_\(dim)",
                inputNames: ["rotated", "codebook"],
                outputNames: ["packed"],
                source: TurboQuantMetalKernels.fusedQuantizePackSource,
                atomicOutputs: true  // packed uses atomic_fetch_or
            )
            lock.lock()
            qpKernelCache[key] = newKernel
            lock.unlock()
            kernel = newKernel
        }

        let totalRows = rotated.dim(0)
        let tgSize = min(dim, 256)

        let result = kernel(
            [rotated, codebook],
            template: [("Bits", bits), ("Dim", dim), ("PackedWidth", pw)],
            grid: (dim, totalRows, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [[totalRows, pw]],
            outputDTypes: [.uint32],
            initValue: 0  // zero-init packed output for atomic OR
        )

        return result[0]
    }
}
