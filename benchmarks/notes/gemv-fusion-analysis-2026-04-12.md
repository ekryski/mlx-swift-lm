# GEMV Fusion Analysis: Why It Doesn't Help Decode on Apple Silicon

**Date**: 2026-04-12
**Branch**: `ek/tom-eric-moe-tuning`
**Model**: Gemma4 E2B 4-bit, M1 Max 64GB

## Summary

Three approaches to fusing Q/K/V (and gate/up MLP) GEMV projections were tested. All regressed decode performance by 3-5%. The dispatch overhead saved (~15-30μs per token) is negligible compared to the memory-bandwidth-bound GEMV compute (~10ms per token).

## Approaches Tested

### 1. Weight Concatenation (sanitize-time fusion)

Concatenate Q/K/V weights into a single `[N_q+N_k+N_v, K_packed]` QuantizedLinear. One `quantizedMM` call + output split replaces 3 separate calls.

**Result**: -5% decode across all contexts.
**Why**: The larger fused output dimension (8192 vs 4096/2048/2048) changes the qmv kernel's tiling behavior. MLX's qmv dispatches 8 output rows per threadgroup — doubling N doesn't improve parallelism but increases per-threadgroup work imbalance. The `split()` to separate Q/K/V adds overhead.

### 2. Custom Metal Kernel — Sequential (shared-memory x)

New `batched_qkv_qgemv.metal` kernel: loads x to shared memory once, then sequentially computes Q, K, V GEMVs within each threadgroup. Grid: `(1, ceil(max_N/8), 1)`.

**Result**: -4% decode.
**Why**: Each threadgroup runs 3x longer (Q then K then V). The GPU can't overlap the sequential work — it's the same threads doing 3 GEMVs back to back. Threadgroups beyond K/V's smaller output dim waste cycles on early-exit.

### 3. Custom Metal Kernel — Z-Parallel (tid.z selects matrix)

Same kernel but uses `tid.z ∈ {0,1,2}` to run Q/K/V in parallel across z-slices. Grid: `(1, ceil(max_N/8), 3)`. Each z-slice loads x independently.

**Result**: -1% to -3% decode (within noise at all contexts).
**Why**: Each z-slice loads x from global memory independently (threadgroups can't share threadgroup memory across z-slices). The 3x more threadgroups compete for the same memory bandwidth. Net: same bandwidth usage as 3 separate dispatches, minus ~15μs dispatch overhead, plus x-load overhead.

## Long Context A/B (8K, 16K, 32K)

| Context | Batched ON | Batched OFF | Delta |
|---------|:----------:|:-----------:|:-----:|
| 8192 | 90.0 | 91.3 | -1.4% |
| 16384 | 85.2 | 84.2 | +1.2% |
| 32768 | 71.3 | 73.3 | -2.7% |

All within noise. No positive impact at any context length.

## Root Cause: Memory Bandwidth Bound

At T=1 decode, each GEMV reads the full weight matrix from unified memory:
- Q proj: 4096 × 2816 × 0.5B (4-bit) = 2.8 MB
- K proj: 2048 × 2816 × 0.5B = 1.4 MB
- V proj: 2048 × 2816 × 0.5B = 1.4 MB
- Total per layer: 5.6 MB × 15 non-shared layers = 84 MB

At 400 GB/s bandwidth: 84 MB / 400 GB/s = 0.21ms for weight reads alone.
At ~420 dispatches/token × ~7μs/dispatch = 2.9ms dispatch overhead.
At ~10ms/token total: dispatch overhead is ~29%, weight reads ~2%.

Wait — that math suggests dispatch IS significant. But the benchmark disagrees. The likely explanation: MLX batches dispatches into Metal command buffers (100+ ops per buffer), amortizing the per-dispatch overhead. The effective overhead per dispatch is much less than 7μs when batched.

The real breakdown (from Metal System Trace):
- GPU compute (weight reads + matmul): ~4.5ms
- CPU graph encoding: ~5-6ms
- GPU idle gaps: ~1-2ms
- Dispatch overhead (within command buffer): <0.5ms

Fusing 3 dispatches into 1 saves ~10μs of in-buffer dispatch overhead — 0.1% of token time.

## Conclusion

For GEMV-dominated decode on Apple Silicon unified memory:
- **Dispatch reduction doesn't help** — in-buffer dispatch overhead is negligible
- **Weight concatenation hurts** — changes kernel tiling behavior
- **Shared-memory x reuse doesn't help** — x is tiny (5.6 KB) vs weights (MB)
- **The bottleneck is memory bandwidth** for weight reads, not dispatch scheduling

## Where Fusion DOES Help

Fusion helps when it eliminates **intermediate buffer traffic** or changes the **algorithmic approach**:
- **RMSNorm+RoPE** fusion: eliminates intermediate normalized tensor write/read (+1-3%)
- **RMSNorm+Residual** fusion: eliminates norm output buffer (+1.2%, 90 dispatches saved)
- **Warp Decode MoE**: changes parallelism from expert-centric to output-centric, eliminating 5 of 8 pipeline stages (est. 1.3-1.8x for MoE models)

The pattern: **fusion wins when it reduces memory traffic, not when it reduces dispatch count.**

## Artifacts

The kernel infrastructure remains in the mlx-swift fork for future use:
- `batched_qkv_qgemv.metal` — Metal kernel (z-parallel variant)
- `BatchedQKVQuantizedGEMV` — C++ primitive in `fast_primitives.h`
- `mlx_fast_batched_qkv_qgemv` — C bridge
- `MLXFast.batchedQKVQuantizedGEMV()` — Swift binding

These will be useful as a starting point for Warp Decode MoE kernels.
