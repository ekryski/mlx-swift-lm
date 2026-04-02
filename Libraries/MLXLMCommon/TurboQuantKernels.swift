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

    /// Fused encode kernel: norm + rotate + quantize + pack + norm correction in ONE dispatch.
    ///
    /// For each input vector [D]:
    ///   1. Compute L2 norm (SIMD reduction)
    ///   2. Normalize to unit vector
    ///   3. Rotate: y = Π · x_unit (shared memory matmul)
    ///   4. Quantize: find codebook index via boundary comparison
    ///   5. Pack bits into uint32 words (atomic OR)
    ///   6. Norm correction: compute reconstruction norm, store original_norm / recon_norm
    ///
    /// Norm correction compensates for quantization error so that
    /// centroid[idx] * corrected_norm more accurately reconstructs the original vector.
    /// This is why TurboQuant beats q8_0 on perplexity.
    ///
    /// Grid: (Dim, numRows, 1) — one threadgroup per vector
    /// Threadgroup: (Dim, 1, 1) — all D threads cooperate
    ///
    /// Template params: Bits, Dim, PackedWidth, NumBoundaries (= 2^Bits - 1)
    static let fusedEncodeSource = """
    constexpr uint LEVELS = 1u << Bits;

    uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
    uint row = thread_position_in_grid.y;         // vector index (B*H*T)

    // --- Step 1: Load input value ---
    float val = input[row * Dim + d];

    // --- Step 2: Compute L2 norm (SIMD reduction) ---
    float sq = val * val;
    float norm_sq = simd_sum(sq);
    // For Dim > 32, need threadgroup reduction
    threadgroup float shared_norm[4];  // up to 4 SIMD groups
    uint sg_id = d / 32;
    if (d % 32 == 0) {
        shared_norm[sg_id] = norm_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_norm_sq = 0;
    uint num_groups = (Dim + 31) / 32;
    for (uint i = 0; i < num_groups; i++) {
        total_norm_sq += shared_norm[i];
    }
    float norm_val = sqrt(total_norm_sq);
    float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

    // --- Step 3: Normalize ---
    float unit_val = val * inv_norm;

    // --- Step 4: Rotate (y = Π · x_unit) via shared memory matmul ---
    // Each thread d computes: y[d] = Σ_j rotation[d * Dim + j] * x_unit[j]
    threadgroup float shared_unit[1024];  // max Dim = 1024
    shared_unit[d] = unit_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rotated = 0.0f;
    for (uint j = 0; j < Dim; j++) {
        rotated += rotation[d * Dim + j] * shared_unit[j];
    }

    // --- Step 5: Quantize via boundary comparison ---
    // Count how many boundaries this value exceeds
    uint idx = 0;
    for (uint b = 0; b < LEVELS - 1; b++) {
        if (rotated > boundaries[b]) idx++;
    }

    // --- Step 6: Pack bits into uint32 word (atomic OR) ---
    uint bit_offset = d * Bits;
    uint word_idx = bit_offset / 32;
    uint shift = bit_offset % 32;
    uint masked = idx & ((1u << Bits) - 1u);

    // Pack bits — use threadgroup shared memory to avoid atomic contention
    // Each thread writes its index bits to shared, then thread 0 per word writes output
    threadgroup uint shared_packed[64];  // max PackedWidth = 64 words
    if (d < PackedWidth) shared_packed[d] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each dimension contributes its bits via atomic OR on threadgroup memory
    uint primary_val = masked << shift;
    atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

    int spill = (int)shift + (int)Bits - 32;
    if (spill > 0) {
        uint spill_val = masked >> ((uint)Bits - (uint)spill);
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write packed words to output (one thread per word)
    if (d < PackedWidth) {
        packed_out[row * PackedWidth + d] = shared_packed[d];
    }

    // --- Step 7: Norm correction ---
    // Compute reconstruction norm: ||codebook[idx]||₂ for the quantized unit vector.
    // Store corrected_norm = original_norm / recon_norm so that
    // decode(centroid[idx] * corrected_norm) better approximates the original vector.
    float centroid_val = codebook[idx];
    float recon_sq = centroid_val * centroid_val;
    float recon_norm_sq = simd_sum(recon_sq);
    // Threadgroup reduction for Dim > 32
    if (d % 32 == 0) {
        shared_norm[sg_id] = recon_norm_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_recon_sq = 0;
    for (uint i = 0; i < num_groups; i++) {
        total_recon_sq += shared_norm[i];
    }
    float recon_norm = sqrt(total_recon_sq);
    float corrected_norm = (recon_norm > 1e-8f) ? (norm_val / recon_norm) : norm_val;

    if (d == 0) {
        norms_out[row] = corrected_norm;
    }
    """

    /// Fused WHT encode kernel: norm + WHT rotation + quantize + pack + norm correction.
    ///
    /// Same as fusedEncodeSource but replaces dense O(d²) matmul with Fast Walsh-Hadamard
    /// Transform O(d log d) butterfly + random sign flip. 18× fewer ops for dim=128.
    ///
    /// WHT forward rotation: y = WHT(signs * x_unit) / sqrt(Dim)
    /// The butterfly pattern: for each stage s in 0..<log2(Dim), pairs at distance 2^s
    /// are combined: (a, b) → (a+b, a-b).
    ///
    /// Template params: Bits, Dim, PackedWidth, LogDim (= log2(Dim))
    static let fusedEncodeWHTSource = """
    constexpr uint LEVELS = 1u << Bits;

    uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
    uint row = thread_position_in_grid.y;         // vector index (B*H*T)

    // --- Step 1: Load input value ---
    float val = input[row * Dim + d];

    // --- Step 2: Compute L2 norm (SIMD reduction) ---
    float sq = val * val;
    float norm_sq = simd_sum(sq);
    threadgroup float shared_norm[4];
    uint sg_id = d / 32;
    if (d % 32 == 0) {
        shared_norm[sg_id] = norm_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_norm_sq = 0;
    uint num_groups = (Dim + 31) / 32;
    for (uint i = 0; i < num_groups; i++) {
        total_norm_sq += shared_norm[i];
    }
    float norm_val = sqrt(total_norm_sq);
    float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

    // --- Step 3: Normalize ---
    float unit_val = val * inv_norm;

    // --- Step 4: WHT rotation via butterfly + sign flip ---
    // Apply random sign: x_signed = signs[d] * x_unit
    threadgroup float shared_buf[1024];  // max Dim = 1024
    shared_buf[d] = wht_signs[d] * unit_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Walsh-Hadamard butterfly: log2(Dim) stages
    // Each stage s: pairs at distance 2^s do (a+b, a-b)
    for (uint s = 0; s < LogDim; s++) {
        uint half_block = 1u << s;
        uint block_size = half_block << 1;
        uint block_id = d / block_size;
        uint pos_in_block = d % block_size;

        float a, b;
        if (pos_in_block < half_block) {
            a = shared_buf[block_id * block_size + pos_in_block];
            b = shared_buf[block_id * block_size + pos_in_block + half_block];
            shared_buf[d] = a + b;
        } else {
            a = shared_buf[block_id * block_size + pos_in_block - half_block];
            b = shared_buf[block_id * block_size + pos_in_block];
            shared_buf[d] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize: WHT has scale factor sqrt(Dim)
    float inv_sqrt_dim = 1.0f / sqrt((float)Dim);
    float rotated = shared_buf[d] * inv_sqrt_dim;

    // --- Step 5: Quantize via boundary comparison ---
    uint idx = 0;
    for (uint b = 0; b < LEVELS - 1; b++) {
        if (rotated > boundaries[b]) idx++;
    }

    // --- Step 6: Pack bits into uint32 word (atomic OR) ---
    uint bit_offset = d * Bits;
    uint word_idx = bit_offset / 32;
    uint shift = bit_offset % 32;
    uint masked = idx & ((1u << Bits) - 1u);

    threadgroup uint shared_packed[64];
    if (d < PackedWidth) shared_packed[d] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint primary_val = masked << shift;
    atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

    int spill = (int)shift + (int)Bits - 32;
    if (spill > 0) {
        uint spill_val = masked >> ((uint)Bits - (uint)spill);
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d < PackedWidth) {
        packed_out[row * PackedWidth + d] = shared_packed[d];
    }

    // --- Step 7: Norm correction ---
    float centroid_val = codebook[idx];
    float recon_sq = centroid_val * centroid_val;
    float recon_norm_sq = simd_sum(recon_sq);
    if (d % 32 == 0) {
        shared_norm[sg_id] = recon_norm_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_recon_sq = 0;
    for (uint i = 0; i < num_groups; i++) {
        total_recon_sq += shared_norm[i];
    }
    float recon_norm = sqrt(total_recon_sq);
    float corrected_norm = (recon_norm > 1e-8f) ? (norm_val / recon_norm) : norm_val;

    if (d == 0) {
        norms_out[row] = corrected_norm;
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
        if (w < 1e-6f) continue;  // Sparse V: skip negligible attention weights

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
    nonisolated(unsafe) private static var encodeKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var scoreKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var valueKernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    /// Fused encode: norm + rotate + quantize + pack + norm correction in single GPU dispatch.
    ///
    /// - Parameters:
    ///   - input: Raw vectors [numRows, D] float32
    ///   - rotation: Rotation matrix Π [D, D] float32
    ///   - boundaries: Codebook boundaries [2^bits - 1] float32
    ///   - codebook: Centroids [2^bits] float32 (needed for norm correction)
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension
    /// - Returns: (packed: [numRows, PackedWidth] uint32, norms: [numRows] float32)
    ///            norms are norm-corrected: original_norm / reconstruction_norm
    public static func fusedEncode(
        input: MLXArray,
        rotation: MLXArray,
        boundaries: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "encode_nc_\(bits)_\(dim)"  // nc = norm-corrected

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = encodeKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_fused_encode_\(bits)_\(dim)",
                inputNames: ["input", "rotation", "boundaries", "codebook"],
                outputNames: ["packed_out", "norms_out"],
                source: TurboQuantMetalKernels.fusedEncodeSource,
                ensureRowContiguous: true
            )
            lock.lock()
            encodeKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let numRows = input.dim(0)

        let results = kernel(
            [input, rotation, boundaries, codebook],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
            ],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, pw], [numRows]],
            outputDTypes: [.uint32, .float32],
            initValue: 0  // zero-init packed output for atomic OR
        )

        return (packed: results[0], norms: results[1])
    }

    /// Fused WHT encode: norm + WHT rotation + quantize + pack + norm correction.
    ///
    /// Same as fusedEncode but uses O(d log d) Walsh-Hadamard butterfly instead of
    /// O(d²) dense matmul. Only works for power-of-2 dimensions.
    ///
    /// - Parameters:
    ///   - input: Raw vectors [numRows, D] float32
    ///   - whtSigns: Random ±1 signs [D] float32
    ///   - boundaries: Codebook boundaries [2^bits - 1] float32
    ///   - codebook: Centroids [2^bits] float32 (needed for norm correction)
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension (must be power of 2)
    /// - Returns: (packed: [numRows, PackedWidth] uint32, norms: [numRows] float32)
    public static func fusedEncodeWHT(
        input: MLXArray,
        whtSigns: MLXArray,
        boundaries: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let logDim = Int(log2(Double(dim)))
        let key = "encode_wht_\(bits)_\(dim)"

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = encodeKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_fused_encode_wht_\(bits)_\(dim)",
                inputNames: ["input", "wht_signs", "boundaries", "codebook"],
                outputNames: ["packed_out", "norms_out"],
                source: TurboQuantMetalKernels.fusedEncodeWHTSource,
                ensureRowContiguous: true
            )
            lock.lock()
            encodeKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let numRows = input.dim(0)

        let results = kernel(
            [input, whtSigns, boundaries, codebook],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw), ("LogDim", logDim),
            ],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, pw], [numRows]],
            outputDTypes: [.uint32, .float32],
            initValue: 0
        )

        return (packed: results[0], norms: results[1])
    }

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
