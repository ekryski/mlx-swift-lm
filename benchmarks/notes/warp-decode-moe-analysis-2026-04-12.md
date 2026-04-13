# Warp Decode MoE Analysis: Why It Doesn't Help on Apple Silicon

**Date**: 2026-04-12
**Branch**: `ek/tom-eric-moe-tuning`
**Model**: Qwen3.5-35B-A3B 4-bit, M1 Max 64GB

## Summary

Implemented Cursor's Warp Decode approach (output-centric MoE parallelism) as two custom Metal kernels. Benchmarked against MLX's `gatherQuantizedMM`. Result: **-20-25% regression**. MLX's existing kernel is already well-tuned for Apple Silicon's unified memory architecture.

## What We Built

Two Metal kernels following the Warp Decode design:

1. **`warp_moe_gate_up.metal`** — Fused gate+up projection + activation (silu/gelu/swiglu)
   - Grid: (1, ceil(H/8), topK) — z-parallel across experts
   - Each threadgroup loads x to shared memory, computes 8 activated neurons for one expert
   - Template activation via compile-time constant (3 types)

2. **`warp_moe_down.metal`** — Fused down projection + routing weight folding
   - Grid: (1, ceil(outDims/8), 1)
   - Each threadgroup computes 8 final output neurons, looping over topK experts
   - Routing score folded into accumulator (eliminates separate weighted-sum dispatch)

Full C framework chain: Metal → C++ primitive → C bridge → Swift binding → `FusedGateUpSwitchGLU.warpDecode()`

## Benchmark Results

Qwen3.5-35B-A3B, 4-bit, no-quant KV, M1 Max:

| Context | Warp ON | Warp OFF | Delta |
|---------|:-------:|:--------:|:-----:|
| 128 | 47.3 | 59.0 | **-20%** |
| 1024 | 46.9 | 62.5 | **-25%** |

## Why It's Slower

### 1. Redundant x Loading (gate_up kernel)
The gate_up kernel uses `tid.z` to parallelize across topK experts. Each z-slice loads x to its own threadgroup shared memory independently — for topK=8, that's 8 redundant loads of the input vector. `gatherQuantizedMM` loads x once.

### 2. Sequential Expert Loop (down kernel)
The down kernel loops over topK experts within each threadgroup. Each iteration:
- Loads the expert's activated vector to shared memory
- Reads the expert's weight row
- Accumulates the weighted dot product

This serializes expert work within each threadgroup. `gatherQuantizedMM` distributes expert work across many threadgroups in parallel.

### 3. Less-Tuned GEMV Inner Loop
Our `qdot_4bit` inner loop is a direct port from `rms_norm_qgemv.metal` — correct but not as optimized as MLX's `qmv_fast_impl` which has:
- Vectorized load instructions (`simdgroup_matrix_load`)
- Better register allocation
- Prefetching hints
- Tuned block sizes per GPU architecture

### 4. Apple Silicon vs NVIDIA Architecture
The Cursor blog achieved 1.84x on B200 (6.8 TB/s, 208 SMs). Apple Silicon differs:
- **Unified memory**: No PCIe transfer overhead, but lower bandwidth (400 GB/s M1 Max)
- **Smaller GPU**: 32 cores vs 208 SMs — fewer concurrent threadgroups
- **Different SIMD width**: 32 lanes (same as CUDA warp) but different instruction mix
- **No cooperative groups**: Can't synchronize across threadgroups (Metal limitation)
- **Different memory hierarchy**: 48MB SLC vs NVIDIA's 60MB L2

The Warp Decode approach benefits from:
- Eliminating scatter-gather data management stages → less impactful on Apple Silicon where unified memory makes these cheaper
- Higher memory bandwidth utilization → already well-utilized by `gatherQuantizedMM`
- Warp-level shuffles → `simd_shuffle_xor` works but the GEMV inner loop dominates, not the reduction

## Comparison with GEMV Fusion Findings

This result is consistent with the QKV/MLP GEMV fusion analysis from the same session:

| Approach | Expected | Actual | Root Cause |
|----------|----------|--------|------------|
| QKV weight concat | -3-5% dispatch | -5% | Larger GEMV worse tiling |
| QKV custom kernel (sequential) | -3% dispatch | -4% | 3x longer per threadgroup |
| QKV custom kernel (z-parallel) | -1% dispatch | -3% | Redundant x loads |
| Warp MoE gate_up+down | -3 dispatch/layer | -20-25% | Less-tuned GEMV, serial expert loop |

**Pattern**: On Apple Silicon, MLX's existing kernels are already near-optimal for the memory-bandwidth-bound GEMV workload. Custom kernels that change the parallelism strategy don't overcome the implementation maturity gap.

## What Would Help Instead

Based on this session's findings, the most promising directions for MoE decode improvement are:

1. **Upstream MLX `gatherQuantizedMM` improvements** — profile and tune the existing kernel for Apple Silicon's cache hierarchy
2. **Expert weight memory layout optimization** — Morton order / co-selection clustering (confirmed -3% from random reorder, so layout matters)
3. **Batch decode** — amortize weight reads across B>1 tokens (Phase 4 in optimization plan)
4. **Profile-guided expert reordering** — calibration-based spectral clustering of frequently co-selected experts

## Artifacts

The kernel infrastructure is preserved for future iteration:
- `warp_moe_gate_up.metal`, `warp_moe_down.metal` — Metal kernels
- `WarpMoeGateUp`, `WarpMoeDown` — C++ primitives
- `MLXFast.warpMoeGateUp()`, `MLXFast.warpMoeDown()` — Swift bindings
- `FusedGateUpSwitchGLU.warpDecode()` — Swift caller (disabled by default, `WARP_MOE_DECODE=1` to enable)
