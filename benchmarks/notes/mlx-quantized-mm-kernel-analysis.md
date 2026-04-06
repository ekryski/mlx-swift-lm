# MLX Quantized MatMul Kernel Analysis

**Date**: 2026-04-06
**Source**: `/Users/eric/Development/personal/ai/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`
**Dispatch**: `/Users/eric/Development/personal/ai/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp`

## Kernel Architecture Overview

MLX's quantized matmul has THREE kernel variants for different problem sizes:

### 1. `qmv_fast` / `gather_qmv_fast` — Vector (M=1, decode)

Used when M=1 (single token decode) and K % 512 == 0 and N % 8 == 0.

**Grid**: (M, ceil(N/8), B) — one threadgroup per 8 output elements per batch
**Threadgroup**: (32, 2, 1) — 2 SIMD groups × 32 lanes = 64 threads

**Algorithm**: Each SIMD group computes 4 output elements (`results_per_simdgroup=4`).
Each thread loads `values_per_thread` elements of x and performs `qdot` against
weight rows. The inner loop iterates over K in blocks of `block_size`.

**Key parameters for 4-bit**: `packs_per_thread=2`, `pack_factor=8` (8 values per uint32),
`values_per_thread=16`, `block_size=512` (16 × 32 lanes).

### 2. `qmm_t` / `gather_qmm_t` — Matrix (M≥vector_limit, prefill)

Used for batched matmul (multiple tokens). Uses Steel's BlockMMA with tiled
matrix multiply. Default tile: BM=32, BN=32, BK=32.

The `gather_qmm_rhs` variant groups consecutive tokens by expert index and
does batch matmuls per group — THIS is why sorting helps (longer consecutive
runs = larger efficient matmuls).

### 3. `gather_qmm_rhs` — Optimized sorted gather (M=1, B≥16, sorted)

Specialized path when `M==1 && B>=16 && right_sorted==true && B/E>=4`.
Batches by expert index for maximum weight reuse.

**NOT used for Qwen3.5 decode** because B=8 (topK=8) < 16 threshold.

## Dispatch Decision for Qwen3.5-35B Decode

```cpp
// GatherQMM::eval_gpu dispatch logic:

if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(...)   // NOT TAKEN: B=8 < 16
}
if (M >= vector_limit) {  // vector_limit=14 for M1 Max with D≤2048
    gather_qmm(...)        // NOT TAKEN: M=1 < 14
}
if (transpose_) {
    gather_qmv(...)        // TAKEN: M=1 with transpose
}
```

## Optimization Opportunities

### A. Lower the gather_qmm_rhs threshold (Quick, MLX C++ change)

The `B >= 16` threshold prevents the optimized sorted-gather path from being
used during decode with topK=8. Lowering to `B >= 4` or `B >= 8` would enable
it for Qwen3.5 (B=8) and most MoE models.

The `B / E >= 4` check would also need adjustment (8/256 = 0.03 < 4). This
check ensures enough tokens per expert to make batching worthwhile, but with
sorted indices and the fused gate_up, the access pattern is already good.

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1383`
**Change**: `if (M == 1 && B >= 8 && right_sorted_ == true)` (remove B/E check)
**Risk**: May be slower for very sparse expert selection — needs benchmarking

### B. Fuse multiple gather_qmv calls into single dispatch

Currently each Linear projection is a separate `gather_qmv` dispatch. For the
MoE block, we have 2 dispatches (fused gate_up + down_proj). Each dispatch
reads the weight tensor independently.

A fused kernel that does gate_up → activation → down_proj in a single dispatch
would avoid re-reading the input vector and keep intermediate values in registers.

**Effort**: Medium-High (new Metal kernel)
**Expected impact**: ~20-30% reduction in MoE GPU compute

### C. Tune qmv block sizes for Apple Silicon

The kernel uses `results_per_simdgroup=4` and `num_simdgroups=2`. For M1 Max:
- 32 GPU cores × 4 execution units = 128 SIMD groups capacity
- Current: each threadgroup uses 2 SIMD groups
- Grid for one MoE projection: M=1, N=1024, B=8 → (8, 128, 1) = 1024 threadgroups

With 1024 threadgroups and 128 SIMD group capacity, occupancy is ~100%.
The block sizes appear well-tuned already.

### D. Expert-aware weight prefetching

The gather kernel reads weight data for the selected expert index. On Apple
Silicon, the L2 cache is 48MB (M1 Max). Each expert's weight for one projection
is ~512KB at 4-bit. With 8 experts × 2 projections = ~8MB — fits in L2!

If we could prefetch the next expert's weights while processing the current
one, we'd hide memory latency. Metal doesn't have explicit prefetch instructions,
but ordering the expert processing to maximize L2 hits could help.

This is essentially what the sorting achieves at the token level — but could
be done at the expert level within a single kernel dispatch.

### E. Int8 dequantization path

The current 4-bit path does: unpack uint4 from uint32 → cast to float → scale + bias.
An Int8 intermediate path would: unpack uint4 → extend to int8 → int8 matmul → scale.
Apple's AMX (Apple Matrix Extension) has native int8 matmul support that could
be significantly faster than the float path.

**Status**: Research — would require new kernel variant
**Expected impact**: 2-3x compute improvement (but memory bandwidth is the limit)

## Implementation Status

### Option A: Lower threshold — TESTED, mixed results (commit 01a09b0 in mlx-swift)
Changed `B >= 16 && B/E >= 4` to `B >= 4`. Enables gather_qmm_rhs for MoE decode.
Result: +4.6% at 1024 no-quant, -7.9% at 32K no-quant. Quality intact.

### Option B: MLX compile() for activation — FAILED
`MLX.compile(shapeless: true)` for split + silu + multiply crashes with quantized
tensors. The compiled graph can't handle the quantized output from gatherQuantizedMM.
A custom Metal kernel is needed for the full fusion.

### Options C-E: Not yet tested

## Summary

The MLX quantized kernels are already well-optimized with proper tiling, SIMD
utilization, and specialized paths for different problem sizes. The main
opportunities for MoE decode are:

1. **Lower gather_qmm_rhs threshold** (A) — quick C++ change, enables optimized
   sorted-gather path for decode
2. **Fused MoE kernel** (B) — combines gate_up + activation + down_proj
3. **Expert prefetching** (D) — better L2 utilization within gather

The bandwidth ceiling (400 GB/s ÷ ~1 GB active weights = ~400 tok/s theoretical)
is still far from current performance (~52 tok/s), so there IS room for improvement
in the kernel dispatch and memory access patterns.
