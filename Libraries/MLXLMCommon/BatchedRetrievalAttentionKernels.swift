// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Custom Metal kernels for the batched retrieval-attention decode path.
//
// Three kernels live here:
//   1. ra_build_mask          — single-slot fp32 mask builder (per-position
//                               membership check vs static / sliding /
//                               top-K-fine / top-K-coarse selections).
//   2. ra_build_mask_b        — batched sibling; one TG per (slot, T-tile).
//   3. ra_group_sdpa_decode   — NSA-style fused sparse SDPA decode kernel
//                               with per-(B, KV-head) gather list. Each
//                               threadgroup serves one (B, Q-head); group
//                               sharing of K/V loads across the GQA group
//                               is done at the threadgroup level (1024
//                               threads = 32 simdgroups x 32 lanes).
//
// These are the minimal subset required by the BATCHED decode stack here.

import Foundation
import MLX

/// Cached one-time-constructed kernel handles. Built lazily on first call.
private final class _BatchedRAKernelCache: @unchecked Sendable {
    static let shared = _BatchedRAKernelCache()
    private let lock = NSLock()
    private var groupSDPAKernel: MLXFast.MLXFastKernel?
    private var buildMaskKernel: MLXFast.MLXFastKernel?
    private var buildMaskBatchedKernel: MLXFast.MLXFastKernel?

    /// NSA-style group-centric fused sparse SDPA. One threadgroup per
    /// (B, Q head); BN=32 simdgroups process BN gather positions in
    /// parallel via simd_sum.
    func getGroupSDPA() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = groupSDPAKernel { return k }
        let header = """
            // Group-centric fused sparse SDPA decode kernel.

            """
        let source = """
            // Template constants: HEAD_DIM, GROUP_SIZE
            // Inputs:
            //   q:       [B, NQH, 1, HEAD_DIM]
            //   k:       [B, NKVH, T, HEAD_DIM]
            //   v:       [B, NKVH, T, HEAD_DIM]
            //   gather:  [B * NKVH * K_padded] int32 (sorted, may carry -1 sentinels)
            //   params:  [scale, NQH_f, NKVH_f, T_f, K_padded_f]
            // Output:
            //   out:     [B, NQH, 1, HEAD_DIM]  (fp32 inside; Swift casts to Q dtype)
            //
            // Layout matches mlx-swift sdpa_vector:
            //   threadgroup = BN=32 simdgroups x BD=32 lanes = 1024 threads
            //   qk_per_thread = HEAD_DIM / BD
            //   inner gather loop: for i = simd_gid; i < K_padded; i += BN
            //   simd_sum for QK reduction (no threadgroup barriers)
            //
            // Grid: (B * NQH * 1024, 1, 1). One threadgroup per (B, Q head).

            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int qk_per_thread = HEAD_DIM / BD;
            constexpr int v_per_thread = HEAD_DIM / BD;

            const float scale = params[0];
            const uint NQH = (uint)params[1];
            const uint NKVH = (uint)params[2];
            const uint T = (uint)params[3];
            const uint K_padded = (uint)params[4];

            const uint qh_idx = threadgroup_position_in_grid.x;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint b = qh_idx / NQH;
            const uint qh = qh_idx % NQH;
            const uint kvh = qh / GROUP_SIZE;

            // Threadgroup state for cross-simd reduce at the end.
            threadgroup float outputs[BN * BD];
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];

            // Per-thread Q register file: qk_per_thread elements pre-scaled.
            float q_local[qk_per_thread];
            float k_local[qk_per_thread];
            float o_local[v_per_thread];

            // Load Q (this thread owns lanes [simd_lid * qk_per_thread, ...]).
            const uint q_base = (b * NQH + qh) * HEAD_DIM
                                 + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; i++) {
                q_local[i] = scale * (float)q[q_base + i];
            }
            for (int i = 0; i < v_per_thread; i++) {
                o_local[i] = 0.0f;
            }

            float max_score = -INFINITY;
            float sum_exp_score = 0.0f;

            const uint gather_base = (b * NKVH + kvh) * K_padded;

            // Each simdgroup processes a strided slice of K_padded.
            for (uint i = simd_gid; i < K_padded; i += BN) {
                const int pos = gather[gather_base + i];
                // Sentinel skip: -1 marks adjacent-dup positions deduped upstream.
                if (pos < 0) continue;

                const uint kv_base = (b * NKVH + kvh) * T * HEAD_DIM
                                     + (uint)pos * HEAD_DIM
                                     + simd_lid * qk_per_thread;

                // Load qk_per_thread K elements
                for (int j = 0; j < qk_per_thread; j++) {
                    k_local[j] = (float)k[kv_base + j];
                }

                // Compute partial QK dot
                float score = 0.0f;
                for (int j = 0; j < qk_per_thread; j++) {
                    score += q_local[j] * k_local[j];
                }
                // simd-level reduction across BD=32 lanes - no barriers.
                score = simd_sum(score);

                // Online softmax update (every lane sees same score).
                const float new_max = fmax(max_score, score);
                const float factor = (max_score == -INFINITY)
                    ? 0.0f : exp(max_score - new_max);
                const float exp_score = exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;

                // V accumulate
                for (int j = 0; j < v_per_thread; j++) {
                    const float v_local = (float)v[kv_base + j];
                    o_local[j] = o_local[j] * factor + exp_score * v_local;
                }
            }

            // Cross-simdgroup reduce (mirrors mlx-swift sdpa_vector).
            if (simd_lid == 0) {
                max_scores[simd_gid] = max_score;
                sum_exp_scores[simd_gid] = sum_exp_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score = max_scores[simd_lid];
            const float new_max = simd_max(max_score);
            const float factor = exp(max_score - new_max);
            sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

            // Aggregate per-output-dim partials across simdgroups.
            const uint out_base = (b * NQH + qh) * HEAD_DIM
                                   + simd_gid * v_per_thread;
            for (int i = 0; i < v_per_thread; i++) {
                outputs[simd_lid * BD + simd_gid] = o_local[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float reduced = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                if (simd_lid == 0) {
                    out[out_base + i] = sum_exp_score == 0.0f
                        ? reduced
                        : (reduced / sum_exp_score);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """
        let k = MLXFast.metalKernel(
            name: "ra_group_sdpa_decode",
            inputNames: ["q", "k", "v", "gather", "params"],
            outputNames: ["out"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        groupSDPAKernel = k
        return k
    }

    /// Single-slot mask builder. Takes per-KV-head top-K block start
    /// positions plus static/sliding window params and writes the
    /// [1, 1, 1, T] additive attention mask. One thread per mask position.
    func getBuildMask() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = buildMaskKernel { return k }
        let header = """
            // Fused build-mask kernel (single slot).

            """
        let source = """
            // Template constants: FINE_BS, COARSE_BS, NKVH, K_FINE, K_COARSE
            // Inputs:
            //   fine_starts:   [NKVH * K_FINE]   int32 block start positions
            //   coarse_starts: [NKVH * K_COARSE] int32 block start positions
            //   params:        [T_f, staticInit_f, slidingWindow_f]
            // Output:
            //   mask:          [T] float — additive (0 valid, -inf masked)

            const uint p = thread_position_in_grid.x;
            const uint T = (uint)params[0];
            if (p >= T) return;
            const uint staticInit = (uint)params[1];
            const uint slidingWindow = (uint)params[2];
            const uint sliding_start = (T > slidingWindow) ? (T - slidingWindow) : 0u;

            // Cache the topK start arrays in threadgroup memory.
            threadgroup int fine_tg[NKVH * K_FINE];
            threadgroup int coarse_tg[NKVH * K_COARSE];

            const uint tid = thread_position_in_threadgroup.x;
            const uint tg_size = threads_per_threadgroup.x;
            for (uint i = tid; i < NKVH * K_FINE; i += tg_size) {
                fine_tg[i] = fine_starts[i];
            }
            for (uint i = tid; i < NKVH * K_COARSE; i += tg_size) {
                coarse_tg[i] = coarse_starts[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool valid = (p < staticInit) || (p >= sliding_start);

            if (!valid) {
                for (uint i = 0; i < NKVH * K_FINE && !valid; i++) {
                    const int start = fine_tg[i];
                    if ((int)p >= start && (int)p < start + (int)FINE_BS) {
                        valid = true;
                    }
                }
            }
            if (!valid) {
                for (uint i = 0; i < NKVH * K_COARSE && !valid; i++) {
                    const int start = coarse_tg[i];
                    if ((int)p >= start && (int)p < start + (int)COARSE_BS) {
                        valid = true;
                    }
                }
            }

            mask[p] = valid ? 0.0f : -INFINITY;
            """
        let k = MLXFast.metalKernel(
            name: "ra_build_mask",
            inputNames: ["fine_starts", "coarse_starts", "params"],
            outputNames: ["mask"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        buildMaskKernel = k
        return k
    }

    /// Batched mask builder. Takes per-slot top-K block starts and writes
    /// the `[B, 1, 1, T]` additive attention mask in one launch. Grid Y
    /// axis indexes the batch slot; X axis the mask position.
    func getBuildMaskBatched() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = buildMaskBatchedKernel { return k }
        let header = """
            // Batched build-mask kernel.

            """
        let source = """
            // Template constants: FINE_BS, COARSE_BS, NKVH, K_FINE, K_COARSE
            // Inputs:
            //   fine_starts:   [B * NKVH * K_FINE]   int32 block start positions
            //   coarse_starts: [B * NKVH * K_COARSE] int32 block start positions
            //   params:        [T_f, staticInit_f, slidingWindow_f]
            // Output:
            //   mask:          [B * T] float — additive (0 valid, -inf masked)

            const uint p = thread_position_in_grid.x;
            const uint b = threadgroup_position_in_grid.y;
            const uint T = (uint)params[0];
            if (p >= T) return;
            const uint staticInit = (uint)params[1];
            const uint slidingWindow = (uint)params[2];
            const uint sliding_start = (T > slidingWindow) ? (T - slidingWindow) : 0u;

            const uint fine_off = b * (uint)NKVH * (uint)K_FINE;
            const uint coarse_off = b * (uint)NKVH * (uint)K_COARSE;

            threadgroup int fine_tg[NKVH * K_FINE];
            threadgroup int coarse_tg[NKVH * K_COARSE];

            const uint tid = thread_position_in_threadgroup.x;
            const uint tg_size = threads_per_threadgroup.x;
            for (uint i = tid; i < NKVH * K_FINE; i += tg_size) {
                fine_tg[i] = fine_starts[fine_off + i];
            }
            for (uint i = tid; i < NKVH * K_COARSE; i += tg_size) {
                coarse_tg[i] = coarse_starts[coarse_off + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool valid = (p < staticInit) || (p >= sliding_start);

            if (!valid) {
                for (uint i = 0; i < NKVH * K_FINE && !valid; i++) {
                    const int start = fine_tg[i];
                    if ((int)p >= start && (int)p < start + (int)FINE_BS) {
                        valid = true;
                    }
                }
            }
            if (!valid) {
                for (uint i = 0; i < NKVH * K_COARSE && !valid; i++) {
                    const int start = coarse_tg[i];
                    if ((int)p >= start && (int)p < start + (int)COARSE_BS) {
                        valid = true;
                    }
                }
            }

            mask[b * T + p] = valid ? 0.0f : -INFINITY;
            """
        let k = MLXFast.metalKernel(
            name: "ra_build_mask_b",
            inputNames: ["fine_starts", "coarse_starts", "params"],
            outputNames: ["mask"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        buildMaskBatchedKernel = k
        return k
    }
}

// MARK: - Public wrappers

/// Run the group-centric fused sparse SDPA decode kernel.
///
/// - Parameters:
///   - queries: `[B, nQH, 1, D]` decode-step Q.
///   - keys: `[B, nKVH, T, D]` cached K (rectangular).
///   - values: `[B, nKVH, T, D]` cached V (rectangular).
///   - perKVHeadGather: `[B, nKVH, K_padded]` int32 sorted positions; -1
///     sentinel marks deduped slots.
///   - scale: SDPA scale (typically `1/sqrt(D)`).
/// - Returns: `[B, nQH, 1, D]` attention output, dtype = queries.dtype.
public func retrievalAttentionGroupSparseSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    perKVHeadGather: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(queries.shape.count == 4, "queries must be [B, nQH, 1, D]")
    precondition(keys.shape.count == 4, "keys must be [B, nKVH, T, D]")
    precondition(values.shape.count == 4, "values must be [B, nKVH, T, D]")
    precondition(perKVHeadGather.shape.count == 3,
        "gather must be [B, nKVH, K_padded] (got \(perKVHeadGather.shape))")
    let B = queries.dim(0)
    let nQH = queries.dim(1)
    precondition(queries.dim(2) == 1, "decode-step L=1 only")
    let D = queries.dim(3)
    let nKVH = keys.dim(1)
    let T = keys.dim(2)
    precondition(perKVHeadGather.dim(0) == B && perKVHeadGather.dim(1) == nKVH,
        "gather batch/heads mismatch")
    let kPadded = perKVHeadGather.dim(2)
    precondition(keys.dim(3) == D, "K head_dim must match Q")
    precondition(values.dim(3) == D, "V head_dim must equal K (Dv==D)")
    precondition(nQH % nKVH == 0, "Q heads must be a multiple of KV heads")
    let groupSize = nQH / nKVH
    precondition([32, 64, 96, 128, 256].contains(D),
        "head_dim \(D) outside the supported template list")

    let params = MLXArray([
        scale,
        Float(nQH),
        Float(nKVH),
        Float(T),
        Float(kPadded),
    ])
    let gatherFlat = perKVHeadGather.reshaped(B * nKVH * kPadded)

    let kernel = _BatchedRAKernelCache.shared.getGroupSDPA()
    // One threadgroup per (B, Q head). 1024 threads/TG (32 simdgroups
    // x 32 lanes), matching mlx-swift sdpa_vector layout.
    let TGSize = 1024
    let totalThreads = B * nQH * TGSize
    let outputs = kernel(
        [queries, keys, values, gatherFlat, params],
        template: [
            ("HEAD_DIM", D),
            ("GROUP_SIZE", groupSize),
        ],
        grid: (totalThreads, 1, 1),
        threadGroup: (TGSize, 1, 1),
        outputShapes: [[B, nQH, 1, D]],
        outputDTypes: [.float32]
    )
    return outputs[0].asType(queries.dtype)
}

/// Build a single-slot `[1, 1, 1, T]` additive attention mask via the
/// fused membership-check kernel.
///
/// - Parameters:
///   - fineStarts: `[nKVH, K_fine]` int32 block-start positions.
///   - coarseStarts: `[nKVH, K_coarse]` int32 block-start positions.
///   - T: total cache positions.
///   - staticInit / slidingWindow / fineBS / coarseBS: window params.
///   - outputDtype: cast for the returned mask (typically `K.dtype`).
/// - Returns: `[1, 1, 1, T]` additive mask (0 valid, -inf masked).
public func retrievalAttentionBuildMaskFused(
    fineStarts: MLXArray,
    coarseStarts: MLXArray,
    T: Int,
    staticInit: Int,
    slidingWindow: Int,
    fineBS: Int,
    coarseBS: Int,
    outputDtype: DType
) -> MLXArray {
    precondition(fineStarts.shape.count == 2, "fineStarts must be [nKVH, K_fine]")
    precondition(coarseStarts.shape.count == 2, "coarseStarts must be [nKVH, K_coarse]")
    let nKVH = fineStarts.dim(0)
    let kFine = fineStarts.dim(1)
    let kCoarse = coarseStarts.dim(1)
    precondition(coarseStarts.dim(0) == nKVH, "head count mismatch")

    let params = MLXArray([
        Float(T), Float(staticInit), Float(slidingWindow),
    ])
    let fineFlat = fineStarts.reshaped(nKVH * kFine)
    let coarseFlat = coarseStarts.reshaped(nKVH * kCoarse)

    let kernel = _BatchedRAKernelCache.shared.getBuildMask()
    let tgSize = 256
    // Round up grid to threadgroup multiple so the kernel only does
    // bounds-check on `p`.
    let gridX = ((T + tgSize - 1) / tgSize) * tgSize
    let outputs = kernel(
        [fineFlat, coarseFlat, params],
        template: [
            ("FINE_BS", fineBS),
            ("COARSE_BS", coarseBS),
            ("NKVH", nKVH),
            ("K_FINE", kFine),
            ("K_COARSE", kCoarse),
        ],
        grid: (gridX, 1, 1),
        threadGroup: (tgSize, 1, 1),
        outputShapes: [[1, 1, 1, T]],
        outputDTypes: [.float32]
    )
    return outputs[0].asType(outputDtype)
}

/// Build a batched `[B, 1, 1, T]` additive attention mask in one launch
/// from per-slot top-K block starts.
///
/// - Parameters:
///   - fineStarts: `[B, nKVH, K_fine]` int32 token positions.
///   - coarseStarts: `[B, nKVH, K_coarse]` int32 token positions.
///   - T: total cache positions (rectangular across slots).
///   - staticInit / slidingWindow / fineBS / coarseBS: window params.
///   - outputDtype: cast for the returned mask (typically `K.dtype`).
/// - Returns: `[B, 1, 1, T]` additive mask.
public func retrievalAttentionBuildMaskFusedBatched(
    fineStarts: MLXArray,
    coarseStarts: MLXArray,
    T: Int,
    staticInit: Int,
    slidingWindow: Int,
    fineBS: Int,
    coarseBS: Int,
    outputDtype: DType
) -> MLXArray {
    precondition(fineStarts.shape.count == 3,
        "fineStarts must be [B, nKVH, K_fine] (got \(fineStarts.shape))")
    precondition(coarseStarts.shape.count == 3,
        "coarseStarts must be [B, nKVH, K_coarse] (got \(coarseStarts.shape))")
    let B = fineStarts.dim(0)
    let nKVH = fineStarts.dim(1)
    let kFine = fineStarts.dim(2)
    let kCoarse = coarseStarts.dim(2)
    precondition(coarseStarts.dim(0) == B, "B mismatch")
    precondition(coarseStarts.dim(1) == nKVH, "head count mismatch")

    let params = MLXArray([
        Float(T), Float(staticInit), Float(slidingWindow),
    ])
    let fineFlat = fineStarts.reshaped(B * nKVH * kFine)
    let coarseFlat = coarseStarts.reshaped(B * nKVH * kCoarse)

    let kernel = _BatchedRAKernelCache.shared.getBuildMaskBatched()
    let tgSize = 256
    let gridX = ((T + tgSize - 1) / tgSize) * tgSize
    let outputs = kernel(
        [fineFlat, coarseFlat, params],
        template: [
            ("FINE_BS", fineBS),
            ("COARSE_BS", coarseBS),
            ("NKVH", nKVH),
            ("K_FINE", kFine),
            ("K_COARSE", kCoarse),
        ],
        // Grid: (T-tile X) x (B slots Y) x 1.
        grid: (gridX, B, 1),
        threadGroup: (tgSize, 1, 1),
        outputShapes: [[B, 1, 1, T]],
        outputDTypes: [.float32]
    )
    return outputs[0].asType(outputDtype)
}
