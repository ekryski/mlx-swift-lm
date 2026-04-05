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

    // --- Step 5: Quantize via branchless boundary comparison ---
    // V2.1 optimization: use arithmetic sum of comparisons instead of branching.
    // Metal compiles (rotated > boundaries[b]) to a predicated 0/1 — summing these
    // is branchless and avoids SIMD lane divergence.
    uint idx = 0;
    for (uint b = 0; b < LEVELS - 1; b++) {
        idx += (uint)(rotated > boundaries[b]);
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

    // --- Step 3: Normalize + sign flip (fused) ---
    // V2.1 optimization: pre-compute inv_norm * sign to eliminate one multiply per element.
    // Instead of: unit_val = val * inv_norm; wht_val = sign * unit_val (2 muls)
    // We do:      wht_val = val * (inv_norm * sign) (1 mul + 1 FMA-friendly product)
    float inv_norm_sign = inv_norm * wht_signs[d];
    float wht_val = val * inv_norm_sign;

    // --- Step 4: WHT rotation via cooperative SIMD shuffle ---
    // V2.1 optimization: use simd_shuffle_xor for intra-SIMD butterfly stages
    // (register-to-register, no shared memory or barriers needed for first 5 stages)

    // Phase 1: Intra-SIMD butterfly via simd_shuffle_xor (stages 0..min(LogDim,5)-1)
    // Each stage s XORs lane indices at distance 2^s — effectively free on Apple GPU
    uint simd_stages = min(LogDim, 5u);  // 5 stages covers 32 lanes (2^5 = 32)
    uint lane_in_simd = d % 32;
    for (uint s = 0; s < simd_stages; s++) {
        uint step = 1u << s;
        float other = simd_shuffle_xor(wht_val, step);
        wht_val = (lane_in_simd & step) ? (other - wht_val) : (other + wht_val);
    }

    // Phase 2: Cross-SIMD-group butterfly via shared memory (stages 5..LogDim-1)
    // Only needed when Dim > 32 — these stages cross SIMD group boundaries
    threadgroup float shared_buf[1024];  // max Dim = 1024
    if (LogDim > 5u) {
        shared_buf[d] = wht_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = simd_stages; s < LogDim; s++) {
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
        wht_val = shared_buf[d];
    }

    // Normalize: WHT has scale factor sqrt(Dim)
    float inv_sqrt_dim = 1.0f / sqrt((float)Dim);
    float rotated = wht_val * inv_sqrt_dim;

    // --- Step 5: Quantize via branchless boundary comparison ---
    // V2.1 optimization: arithmetic sum avoids SIMD lane divergence
    uint idx = 0;
    for (uint b = 0; b < LEVELS - 1; b++) {
        idx += (uint)(rotated > boundaries[b]);
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

    /// TurboFlashAttention Pass 1: Per-block partial attention with online softmax.
    ///
    /// Parallelizes across both query heads AND token blocks. Each SIMD group (32 lanes)
    /// handles one (query, block) pair, producing partial online softmax state (m, l, o[D]).
    /// Pass 2 merges partials across blocks.
    ///
    /// Grid: (32, totalQueries, numBlocks)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: KeyBits, ValueBits, Dim, KeyPackedWidth, ValuePackedWidth,
    ///                  BlockSize, token_count, repeat_count, num_blocks
    static let turboFlashPass1Source = """
    constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
    constexpr uint KEY_LEVELS = 1u << KeyBits;
    constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
    constexpr uint VAL_LEVELS = 1u << ValueBits;
    constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

    uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
    uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
    uint block_idx = thread_position_in_grid.z; // token block index
    uint kv_idx = q_idx / repeat_count;         // map to KV head (GQA)

    // Token range for this block
    uint t_start = block_idx * BlockSize;
    uint t_end = t_start + BlockSize;
    if (t_end > (uint)token_count) t_end = (uint)token_count;

    // Load key codebook into registers
    float key_cb[KEY_LEVELS];
    for (uint i = 0; i < KEY_LEVELS; i++) {
        key_cb[i] = key_codebook[i];
    }

    // Load value codebook into registers
    float val_cb[VAL_LEVELS];
    for (uint i = 0; i < VAL_LEVELS; i++) {
        val_cb[i] = val_codebook[i];
    }

    // Load query values for this lane's dimensions
    float q_vals[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
    }

    // Online softmax state for this block
    float m = -INFINITY;
    float l = 0.0f;
    float o[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

    // Process tokens in this block
    for (uint t = t_start; t < t_end; t++) {
        // --- Score: Q×K dot product ---
        const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
        float k_norm = key_norms[kv_idx * token_count + t];

        float dot_partial = 0.0f;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d >= Dim) break;

            uint k_bit_offset = d * KeyBits;
            uint k_word_idx = k_bit_offset / 32;
            uint k_shift = k_bit_offset % 32;
            uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
            int k_spill = (int)k_shift + (int)KeyBits - 32;
            if (k_spill > 0) {
                k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
            }
            k_value &= KEY_MASK;

            dot_partial += q_vals[i] * key_cb[k_value];
        }

        float score = simd_sum(dot_partial) * k_norm;

        // --- Online softmax update + V accumulation ---
        float new_m = max(m, score);
        float exp_diff = exp(m - new_m);
        float exp_score = exp(score - new_m);

        const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
        float v_norm = val_norms[kv_idx * token_count + t];

        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d >= Dim) break;

            uint v_bit_offset = d * ValueBits;
            uint v_word_idx = v_bit_offset / 32;
            uint v_shift = v_bit_offset % 32;
            uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
            int v_spill = (int)v_shift + (int)ValueBits - 32;
            if (v_spill > 0) {
                v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
            }
            v_value &= VAL_MASK;

            o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
        }

        l = l * exp_diff + exp_score;
        m = new_m;
    }

    // Write partial results: o[D], m, l
    uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        if (d < Dim) {
            o_partials[partial_base + d] = o[i];
        }
    }
    if (lane == 0) {
        uint ml_idx = q_idx * num_blocks + block_idx;
        m_partials[ml_idx] = m;
        l_partials[ml_idx] = l;
    }
    """

    /// TurboFlashAttention Pass 1 (Causal): Per-block partial attention with causal masking.
    ///
    /// Same as turboFlashPass1Source but supports L>1 query chunks with causal masking.
    /// Each query position q_within_L only attends to tokens where t <= q_offset + q_within_L.
    /// Blocks that are entirely future-masked exit early.
    ///
    /// Grid: (32, totalQueries, numBlocks) where totalQueries = B * nQHeads * L
    /// Threadgroup: (32, 1, 1)
    ///
    /// Additional template params: L (query chunk length), q_offset (absolute offset of first query)
    static let turboFlashPass1CausalSource = """
    constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
    constexpr uint KEY_LEVELS = 1u << KeyBits;
    constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
    constexpr uint VAL_LEVELS = 1u << ValueBits;
    constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

    uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
    uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
    uint block_idx = thread_position_in_grid.z; // token block index

    // For L>1, queries are laid out as [B * nQHeads * L, D] from reshape of [B, nQHeads, L, D].
    // q_idx = b * (nQHeads * L) + h * L + l
    // We need: l (position within chunk) and kv_head (for GQA mapping).
    uint q_within_L = q_idx % L;
    uint q_head_idx = q_idx / L;               // index into [B * nQHeads]
    uint kv_idx = q_head_idx / repeat_count;   // map to KV head (GQA)

    // Causal boundary: this query can attend to tokens 0..q_abs (inclusive)
    uint q_abs = q_offset + q_within_L;

    // Token range for this block
    uint t_start = block_idx * BlockSize;
    uint t_end = t_start + BlockSize;
    if (t_end > (uint)token_count) t_end = (uint)token_count;

    // Early exit: entire block is future-masked
    if (t_start > q_abs) {
        uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) o_partials[partial_base + d] = 0.0f;
        }
        if (lane == 0) {
            uint ml_idx = q_idx * num_blocks + block_idx;
            m_partials[ml_idx] = -INFINITY;
            l_partials[ml_idx] = 0.0f;
        }
        return;
    }

    // Clamp t_end to causal boundary
    if (t_end > q_abs + 1) t_end = q_abs + 1;

    // Load key codebook into registers
    float key_cb[KEY_LEVELS];
    for (uint i = 0; i < KEY_LEVELS; i++) {
        key_cb[i] = key_codebook[i];
    }

    // Load value codebook into registers
    float val_cb[VAL_LEVELS];
    for (uint i = 0; i < VAL_LEVELS; i++) {
        val_cb[i] = val_codebook[i];
    }

    // Load query values for this lane's dimensions
    float q_vals[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
    }

    // Online softmax state for this block
    float m = -INFINITY;
    float l = 0.0f;
    float o[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

    // Process tokens in this block (up to causal boundary)
    for (uint t = t_start; t < t_end; t++) {
        // --- Score: Q×K dot product ---
        const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
        float k_norm = key_norms[kv_idx * token_count + t];

        float dot_partial = 0.0f;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d >= Dim) break;

            uint k_bit_offset = d * KeyBits;
            uint k_word_idx = k_bit_offset / 32;
            uint k_shift = k_bit_offset % 32;
            uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
            int k_spill = (int)k_shift + (int)KeyBits - 32;
            if (k_spill > 0) {
                k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
            }
            k_value &= KEY_MASK;

            dot_partial += q_vals[i] * key_cb[k_value];
        }

        float score = simd_sum(dot_partial) * k_norm;

        // --- Online softmax update + V accumulation ---
        float new_m = max(m, score);
        float exp_diff = exp(m - new_m);
        float exp_score = exp(score - new_m);

        const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
        float v_norm = val_norms[kv_idx * token_count + t];

        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d >= Dim) break;

            uint v_bit_offset = d * ValueBits;
            uint v_word_idx = v_bit_offset / 32;
            uint v_shift = v_bit_offset % 32;
            uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
            int v_spill = (int)v_shift + (int)ValueBits - 32;
            if (v_spill > 0) {
                v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
            }
            v_value &= VAL_MASK;

            o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
        }

        l = l * exp_diff + exp_score;
        m = new_m;
    }

    // Write partial results: o[D], m, l
    uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        if (d < Dim) {
            o_partials[partial_base + d] = o[i];
        }
    }
    if (lane == 0) {
        uint ml_idx = q_idx * num_blocks + block_idx;
        m_partials[ml_idx] = m;
        l_partials[ml_idx] = l;
    }
    """

    /// TurboFlashAttention Pass 2: Cross-block reduction.
    ///
    /// Merges partial online softmax states from pass 1 across token blocks.
    /// Each SIMD group handles one query, iterating over all blocks to produce
    /// the final normalized output.
    ///
    /// Grid: (32, totalQueries, 1)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: Dim, num_blocks
    static let turboFlashPass2Source = """
    constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

    uint lane = thread_position_in_grid.x;
    uint q_idx = thread_position_in_grid.y;

    float m = -INFINITY;
    float l = 0.0f;
    float o[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

    for (uint b = 0; b < (uint)num_blocks; b++) {
        uint ml_idx = q_idx * num_blocks + b;

        // All lanes read the same m/l (broadcast read from device memory)
        float block_m = m_partials[ml_idx];
        float block_l = l_partials[ml_idx];

        // Skip empty blocks
        if (block_l == 0.0f) continue;

        float new_m = max(m, block_m);
        float exp_old = exp(m - new_m);
        float exp_block = exp(block_m - new_m);

        uint partial_base = (q_idx * num_blocks + b) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
            }
        }

        l = l * exp_old + block_l * exp_block;
        m = new_m;
    }

    // Write normalized output
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        if (d < Dim) {
            output[q_idx * Dim + d] = o[i] * inv_l;
        }
    }
    """

    /// TurboFlashAttention Pass 2 with fused output rotation.
    ///
    /// Same as turboFlashPass2Source but applies inverse value rotation (Π_val) in-kernel
    /// after merging partials, eliminating a separate MLX matmul dispatch.
    /// Uses threadgroup shared memory to gather the full output vector across SIMD lanes,
    /// then each lane computes rotated output as dot product with rotation matrix rows.
    ///
    /// Grid: (32, totalQueries, 1)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: Dim, num_blocks
    static let turboFlashPass2FusedRotSource = """
    constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

    uint lane = thread_position_in_grid.x;
    uint q_idx = thread_position_in_grid.y;

    float m = -INFINITY;
    float l = 0.0f;
    float o[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

    for (uint b = 0; b < (uint)num_blocks; b++) {
        uint ml_idx = q_idx * num_blocks + b;

        float block_m = m_partials[ml_idx];
        float block_l = l_partials[ml_idx];

        if (block_l == 0.0f) continue;

        float new_m = max(m, block_m);
        float exp_old = exp(m - new_m);
        float exp_block = exp(block_m - new_m);

        uint partial_base = (q_idx * num_blocks + b) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
            }
        }

        l = l * exp_old + block_l * exp_block;
        m = new_m;
    }

    // Normalize
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

    // Gather normalized output into threadgroup shared memory for rotation
    threadgroup float shared_out[Dim];
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        if (d < Dim) {
            shared_out[d] = o[i] * inv_l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply inverse value rotation: output[d] = Σ_j shared_out[j] * Π_val[j][d]
    // matmul(x, Π_val) reads column d of Π_val for output dimension d.
    // Π_val is stored row-major [Dim, Dim], so column d = val_rotation[j * Dim + d]
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        if (d < Dim) {
            float acc = 0.0f;
            for (uint j = 0; j < Dim; j++) {
                acc += shared_out[j] * val_rotation[j * Dim + d];
            }
            output[q_idx * Dim + d] = acc;
        }
    }
    """

    /// Sparse V skip threshold for the value aggregation kernel.
    /// Attention weights below this threshold are skipped entirely, saving memory bandwidth
    /// on the V codebook lookup. Default 1e-6 works well for most models.
    ///
    /// Override via environment variable `TURBO_SPARSE_V_THRESHOLD` (e.g. "1e-5", "0.0").
    /// Set to "0.0" to disable sparse V skipping entirely.
    ///
    /// From llama.cpp TurboQuant+ research: this threshold matters for quality/perf tradeoff.
    /// Too aggressive (e.g. 1e-4) can clip meaningful attention weights in long contexts.
    /// Too conservative (e.g. 0) loses the 15-30% decode speedup from sparse V.
    static let sparseVThreshold: Float = {
        if let envValue = ProcessInfo.processInfo.environment["TURBO_SPARSE_V_THRESHOLD"],
           let parsed = Float(envValue) {
            return parsed
        }
        return 1e-6  // default, validated in llama.cpp benchmarks
    }()

    /// Value aggregation kernel: weighted sum of codebook-quantized values.
    ///
    /// output[d] = Σ_t weights[t] * norm[t] * codebook[val_idx[t,d]]
    /// Result is in rotated space — caller applies inverse rotation.
    /// Sparse V: tokens with attention weight < sparseVThreshold are skipped.
    ///
    /// Grid: (32, totalHeads, ceil(Dim/32))
    /// Threadgroup: (32, 1, 1)
    static var valueKernelSource: String {
        let threshold = String(format: "%e", sparseVThreshold)
        return """
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
        if (w < \(threshold)f) continue;  // Sparse V: skip negligible attention weights

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

    // Flash attention kernel caches
    nonisolated(unsafe) private static var flashPass1Kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var flashPass2Kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// Default block size for TurboFlashAttention two-pass approach.
    /// Each SIMD group processes this many tokens per block.
    /// Tuned for M1 Max via sweep: B=64 wins or ties at all token counts (512-8192+).
    /// Smaller blocks = more parallelism but more pass-2 merge work.
    ///
    /// Override via environment variable `TURBO_FLASH_BLOCK_SIZE` to tune for different
    /// GPU configurations. M5 Max (more GPU cores, higher bandwidth) may benefit from
    /// larger block sizes (e.g. 128) to reduce pass-2 merge overhead, while older chips
    /// with fewer cores may prefer smaller blocks (e.g. 32) for better parallelism.
    ///
    /// Example: `TURBO_FLASH_BLOCK_SIZE=128 ./my_app`
    public static let flashBlockSize: Int = {
        if let envValue = ProcessInfo.processInfo.environment["TURBO_FLASH_BLOCK_SIZE"],
           let parsed = Int(envValue), parsed > 0 {
            return parsed
        }
        return 64  // default, tuned for M1 Max
    }()

    /// Current sparse V skip threshold. Reads from `TurboQuantMetalKernels.sparseVThreshold`.
    /// Attention weights below this value are skipped in the value aggregation kernel.
    /// Configurable via `TURBO_SPARSE_V_THRESHOLD` environment variable.
    public static var sparseVThreshold: Float { TurboQuantMetalKernels.sparseVThreshold }

    /// Shared pass 1 dispatch — used by both causal and non-causal variants.
    private static func dispatchFlashPass1(
        source: String, cachePrefix: String,
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        blockSize: Int, extraTemplateParams: [(String, Int)] = []
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)

        let pass1Key = "\(cachePrefix)_\(keyBits)_\(valueBits)_\(dim)"
        let pass1Kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass1Kernels[pass1Key] {
            pass1Kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_\(cachePrefix)_\(keyBits)_\(valueBits)_\(dim)",
                inputNames: ["q_rot", "key_packed", "key_norms", "key_codebook",
                             "val_packed", "val_norms", "val_codebook"],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: source,
                ensureRowContiguous: true
            )
            lock.lock()
            flashPass1Kernels[pass1Key] = k
            lock.unlock()
            pass1Kernel = k
        }

        var template: [(String, Int)] = [
            ("KeyBits", keyBits), ("ValueBits", valueBits),
            ("Dim", dim), ("KeyPackedWidth", kpw), ("ValuePackedWidth", vpw),
            ("BlockSize", blockSize), ("token_count", tokenCount),
            ("repeat_count", repeatCount), ("num_blocks", numBlocks),
        ]
        template.append(contentsOf: extraTemplateParams)

        let partials = pass1Kernel(
            [rotatedQueries, keyPacked, keyNorms, keyCodebook,
             valPacked, valNorms, valCodebook],
            template: template,
            grid: (32, totalQ, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )

        return (oPartials: partials[0], mPartials: partials[1], lPartials: partials[2])
    }

    /// Shared pass 2 dispatch — with optional fused output rotation.
    ///
    /// When `valRotation` is provided, the inverse value rotation (Π_val) is applied
    /// in-kernel using threadgroup shared memory, eliminating a separate MLX matmul dispatch.
    /// Output is in original (non-rotated) space.
    ///
    /// When `valRotation` is nil, output is in rotated V space (caller must apply inverse rotation).
    private static func dispatchFlashPass2(
        oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray,
        dim: Int, numBlocks: Int, totalQ: Int,
        valRotation: MLXArray? = nil
    ) -> MLXArray {
        let fused = valRotation != nil
        let pass2Key = fused ? "flash_p2_fused_\(dim)" : "flash_p2_\(dim)"
        let pass2Kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass2Kernels[pass2Key] {
            pass2Kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k: MLXFast.MLXFastKernel
            if fused {
                k = MLXFast.metalKernel(
                    name: "turbo_flash_p2_fused_\(dim)",
                    inputNames: ["o_partials", "m_partials", "l_partials", "val_rotation"],
                    outputNames: ["output"],
                    source: TurboQuantMetalKernels.turboFlashPass2FusedRotSource,
                    ensureRowContiguous: true
                )
            } else {
                k = MLXFast.metalKernel(
                    name: "turbo_flash_p2_\(dim)",
                    inputNames: ["o_partials", "m_partials", "l_partials"],
                    outputNames: ["output"],
                    source: TurboQuantMetalKernels.turboFlashPass2Source,
                    ensureRowContiguous: true
                )
            }
            lock.lock()
            flashPass2Kernels[pass2Key] = k
            lock.unlock()
            pass2Kernel = k
        }

        let inputs: [MLXArray] = fused
            ? [oPartials, mPartials, lPartials, valRotation!]
            : [oPartials, mPartials, lPartials]

        return pass2Kernel(
            inputs,
            template: [
                ("Dim", dim), ("num_blocks", numBlocks),
            ],
            grid: (32, totalQ, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, dim]],
            outputDTypes: [.float32]
        )[0]
    }

    /// TurboFlashAttention: two-pass fused Score + Online Softmax + Value.
    ///
    /// Pass 1: Parallelizes across (query × token_block) pairs. Each SIMD group processes
    ///         BlockSize tokens, producing partial online softmax state (m, l, o[D]).
    /// Pass 2: Merges partial states across blocks to produce final normalized output.
    ///
    /// Eliminates intermediate score and attention weight arrays entirely.
    ///
    /// - Parameter valRotation: Optional [D, D] inverse value rotation matrix. When provided,
    ///   rotation is fused into pass 2, eliminating a separate MLX matmul dispatch.
    ///   Output is in original space. When nil, output is in rotated V space.
    /// - Parameter blockSize: Tokens per block (default: flashBlockSize). Smaller = more parallelism
    ///   but more pass-2 merge work. Must be > 0.
    /// - Returns: Output [totalQ, D] float32
    public static func turboFlashAttention(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray,
        keyNorms: MLXArray,
        keyCodebook: MLXArray,
        valPacked: MLXArray,
        valNorms: MLXArray,
        valCodebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        keyBits: Int,
        valueBits: Int,
        dim: Int,
        valRotation: MLXArray? = nil,
        blockSize: Int? = nil
    ) -> MLXArray {
        let blockSize = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)

        let (oPartials, mPartials, lPartials) = dispatchFlashPass1(
            source: TurboQuantMetalKernels.turboFlashPass1Source,
            cachePrefix: "flash_p1",
            rotatedQueries: rotatedQueries,
            keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
            valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            blockSize: blockSize
        )

        return dispatchFlashPass2(
            oPartials: oPartials, mPartials: mPartials, lPartials: lPartials,
            dim: dim, numBlocks: numBlocks, totalQ: totalQ,
            valRotation: valRotation
        )
    }

    /// TurboFlashAttention with causal masking for L>1 prefill chunks.
    ///
    /// Same as turboFlashAttention but each query position only attends to tokens
    /// where t <= queryOffset + q_within_L. Eliminates the need to materialize
    /// the full [nQHeads, L, T] score matrix for causal masking.
    ///
    /// - Parameter queryChunkLength: Number of query positions in the chunk (L)
    /// - Parameter queryOffset: Absolute position of the first query in the chunk
    /// - Returns: Output [totalQ, D] float32
    public static func turboFlashAttentionCausal(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray,
        keyNorms: MLXArray,
        keyCodebook: MLXArray,
        valPacked: MLXArray,
        valNorms: MLXArray,
        valCodebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        keyBits: Int,
        valueBits: Int,
        dim: Int,
        queryChunkLength: Int,
        queryOffset: Int,
        valRotation: MLXArray? = nil,
        blockSize: Int? = nil
    ) -> MLXArray {
        let blockSize = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)

        let (oPartials, mPartials, lPartials) = dispatchFlashPass1(
            source: TurboQuantMetalKernels.turboFlashPass1CausalSource,
            cachePrefix: "flash_p1_causal",
            rotatedQueries: rotatedQueries,
            keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
            valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            blockSize: blockSize,
            extraTemplateParams: [("L", queryChunkLength), ("q_offset", queryOffset)]
        )

        return dispatchFlashPass2(
            oPartials: oPartials, mPartials: mPartials, lPartials: lPartials,
            dim: dim, numBlocks: numBlocks, totalQ: totalQ,
            valRotation: valRotation
        )
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
