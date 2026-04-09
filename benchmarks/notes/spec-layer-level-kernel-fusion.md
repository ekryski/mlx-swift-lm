# Spec: Layer-Level Kernel Fusion

## Status: Planned (blocked on memory overhead investigation)

## Problem

Each Gemma4 E2B decode token dispatches ~420 Metal kernels (after existing fusions). At ~5-10μs encoding cost per dispatch, that's ~2-4ms of CPU-side command encoding overhead. Combined with command buffer commit overhead, `asyncEval` takes ~10ms per token when the theoretical GPU compute floor is ~2.5ms.

## Current State: All Ops Already Have C Kernels

For the standard decode path (no TurboQuant, no GDN), **every operation already has a framework-level C kernel**:

| Operation | Framework kernel | File |
|-----------|-----------------|------|
| quantizedMatmul (GEMV/GEMM) | ✅ | quantized.cpp |
| rmsNorm | ✅ | normalization.cpp |
| rmsNormRoPE (fused) | ✅ | normalization.cpp |
| RoPE | ✅ | rope.cpp |
| scaledDotProductAttention | ✅ | scaled_dot_product_attention.cpp |
| matmul (non-quantized) | ✅ | steel gemm |
| softmax, argmax, categorical | ✅ | standard Metal kernels |
| add, mul, silu, gelu | ✅ | binary.metal, unary.metal |
| gatherMM / gatherQuantizedMM | ✅ | quantized.cpp |

JIT kernels only fire for:
- TurboQuant KV compression (9 kernels, only with `--kv turbo*`)
- GatedDeltaNet recurrence (2 kernels, only for Qwen3.5 GDN layers)
- Fused gate activation (1 kernel, disabled by default)

**The bottleneck is dispatch COUNT, not kernel quality.**

## What's Fusable

### Already fused

| Fusion | Ops saved | Kernel |
|--------|----------|--------|
| RMSNorm + RoPE → 1 dispatch | 60/token (2/layer × 30) | `rms_norm_rope.metal` |
| RMSNorm + residual add → 1 dispatch | ~60/token | `compiledNormResidual` |
| gelu + multiply → 1 dispatch | ~30/token | `compiledGeglu` |

### Proposed fusions (no hard barriers)

#### 1. Batched QKV GEMV (highest impact)

Fuse Q, K, V projections into a single dispatch. All three read the same normed input `x` — perfect for shared memory reuse.

```
Current: [norm] [Q_gemv] [K_gemv] [V_gemv]  →  4 dispatches
Fused:   [norm+QKV_gemv]                     →  1 dispatch
```

**Saves**: 3 dispatches × 30 layers = **90 dispatches/token**

**Metal kernel design**: Load `x` into shared memory once, compute RMSNorm, then stream Q/K/V weight rows through. Each threadgroup block produces output rows for all 3 projections.

**Complexity**: Medium — extends the `rms_norm_qgemv` kernel pattern to 3 output matrices.

#### 2. Fused MLP (gate + up + activation + down)

Fuse the entire MLP block. Internal dependency: `down` reads the output of `gelu(gate) × up`.

```
Current: [norm] [gate_gemv] [up_gemv] [gelu×mul] [down_gemv] [norm+residual]  →  6 dispatches
Fused:   [fused_MLP]                                                          →  1 dispatch
```

**Saves**: 5 dispatches × 30 layers = **150 dispatches/token**

**Metal kernel design**: Two-phase kernel. Phase 1: shared memory load + norm + gate/up GEMVs + activation (all in registers). Phase 2: down GEMV from the activated intermediate. Challenge: intermediate dimension (2112 for Gemma4) must fit in registers or shared memory.

**Complexity**: High — the intermediate result between gate×up and down_proj is ~4 KB (2112 × 2 bytes), which fits in shared memory but requires careful orchestration.

#### 3. O projection + norm + residual

```
Current: [O_gemv] [norm+residual]  →  2 dispatches
Fused:   [Oproj_norm_residual]     →  1 dispatch
```

**Saves**: 1 dispatch × 30 layers = **30 dispatches/token**

**Complexity**: Medium — GEMV output feeds into norm+add, all fixed-shape.

### Hard barriers (cannot fuse across)

| Barrier | Why |
|---------|-----|
| KV cache update → SDPA | SDPA must read the updated cache (includes new K/V) |
| Attention output → MLP input | Data dependency |
| Layer N output → Layer N+1 input | Sequential dependency |
| Router → Expert dispatch (MoE) | Top-K selection determines which expert weights to read |

### Proposed per-layer dispatch structure

```
Current (~14 dispatches/layer for Gemma4 E2B):
  [inputNorm] [Q_gemv] [K_gemv] [V_gemv] [v_norm] [q_normrope] [k_normrope]
  [cache_update] [SDPA]
  [O_gemv] [norm+residual]
  [preFFNnorm] [gate_gemv] [up_gemv] [gelu×mul] [down_gemv] [norm+residual]

Proposed (~4 dispatches/layer):
  [fused_norm+QKV_gemv+v_norm+normrope]   — 1 dispatch, all projections + norms
  [cache_update + SDPA]                    — 1-2 dispatches (barrier between them)
  [fused_Oproj+norm+residual]              — 1 dispatch
  [fused_MLP]                              — 1 dispatch (norm+gate+up+gelu×mul+down+norm+residual)
```

**Result: ~4-5 dispatches/layer × 30 layers = ~120-150 total** (down from ~420)

### Projected impact

| Metric | Current | After fusion | Improvement |
|--------|---------|-------------|-------------|
| Dispatches/token | ~420 | ~120-150 | **-65-70%** |
| Encoding time | ~2-4ms | ~0.6-1.5ms | ~2ms saved |
| Command buffer commits | 4-5 | 1-2 | ~1ms saved |
| **Total asyncEval** | ~10ms | ~7-8ms | **~20-30% faster decode** |

Note: this does NOT reduce GPU compute time (still reading ~1 GB of weights). It reduces CPU-side encoding overhead and GPU kernel launch gaps.

## What about a single mega-kernel?

### Theoretically possible but practically limited

A single kernel for the entire forward pass would eliminate ALL dispatch overhead. But:

1. **Threadgroup memory limit (32 KB)**: Hidden state (5.6 KB) + attention intermediates + MLP intermediates won't all fit. Larger models (hidden=5376) are even worse.

2. **Different parallelism patterns per operation**:
   - GEMV: parallel over output rows (N threads)
   - SDPA: parallel over heads × key blocks
   - Norm: parallel over row elements with SIMD reduction
   - These need different thread layouts — a single dispatch wastes most of the GPU on each sub-operation.

3. **Weight reads dominate**: Even with zero dispatch overhead, ~1 GB weights ÷ 400 GB/s = 2.5ms floor. The ~2-3ms we'd save from zero-dispatch is meaningful (~20-30%) but not transformative.

4. **The real mega-kernel opportunity is Metal Indirect Command Buffers**: Pre-encode the entire ~120-dispatch sequence once, replay it each token with just buffer pointer updates. This achieves the "zero encoding overhead" of a mega-kernel without the parallelism constraints. See `spec-adaptive-command-buffer-management.md`.

## Implementation priority

| Fusion | Dispatches saved | Difficulty | Priority |
|--------|-----------------|------------|----------|
| **Batched QKV GEMV** | 90 | Medium | **1st** |
| **Fused MLP** | 150 | High | **2nd** |
| O proj + norm + residual | 30 | Medium | 3rd |
| Metal ICBs (replay all) | Eliminates encoding | High | Long-term |

## Prerequisites

- Resolve memory overhead investigation (why peak memory is higher than expected)
- Implement adaptive command buffer management (output size tracking)
- These fusions increase per-dispatch memory footprint (larger kernels, more shared memory), so memory management must be solid first.

## Models affected

| Model | Current ops | After fusion | Notes |
|-------|------------|-------------|-------|
| Gemma4 E2B | ~420 | ~120 | Standard fusion |
| Gemma4 26B-A4B | ~900 | ~350 | MoE routing adds unfusable ops |
| Gemma4 31B | ~1260 | ~300 | 60 layers, biggest absolute savings |
| Qwen3.5-27B | ~1120 | ~400 | GDN layers already use fused kernel |
| Qwen3.5-35B-A3B | ~1020 | ~400 | GDN+MoE limits fusion opportunities |
| GPT-OSS-20B | ~480 | ~150 | Only 24 layers, swiglu MoE |
