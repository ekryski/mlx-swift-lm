# Phase 1: MLX Kernel Options — Empirical Results

Date: 2026-04-08
Machine: Apple Silicon (M1 Max, tested via mlx-swift-lm benchmark suite)

## 1a. Option A — gather_qmm_rhs threshold B>=4 (Already Done)

+6-14% prefill, +5-11% decode. Keeping. No changes.

## 1b. Option B — Fused Activation via compile() and Custom Metal Kernel

### compile() Approach

Wrapped `split + activation + multiply` in `compile(shapeless:)` inside `FusedGateUpSwitchGLU`.

**Results:**
- `compile(shapeless: true)` — **crashes** with `[Primitive::output_shapes] Split cannot infer output shapes`. The `split` operation needs to know the input shape to determine split points; `shapeless: true` is incompatible.
- `compile(shapeless: false)` — **neutral to slightly negative**. Recompilation overhead on shape changes (prefill vs decode) negates any benefit. Benchmarks across Qwen3.5-35B and GPT-OSS showed no measurable improvement.

**Conclusion:** `compile()` is not effective for this use case. The operations are too simple for graph-level compilation to help.

### Custom Metal Kernel Approach

Wrote a fused Metal kernel (`FusedGateActivationKernel`) supporting silu, gelu_approximate, and swiglu activations. Single dispatch replaces split + activation + multiply (3 dispatches).

**Results — Gemma4 26B (gelu_approximate):**
| Config | Metric | Without kernel | With kernel | Delta |
|--------|--------|---------------|-------------|-------|
| 1K none | Decode | 31.1 | 31.1 | 0% |
| 4K none | Decode | 29.8 | 29.9 | 0% |

**Results — GPT-OSS (swiglu, always-on vs decode-only):**
| Config | Metric | Baseline | Always fused | Decode-only fused |
|--------|--------|----------|-------------|-------------------|
| 1K none | Prefill | 652 | 469 (-28%) | 659 (+1%) |
| 1K none | Decode | 68.5 | 73.9 (+8%) | 67.6 (-1%) |
| 4K none | Prefill | 679 | 488 (-28%) | 695 (+2%) |
| 4K none | Decode | 65.1 | 68.9 (+6%) | 64.7 (-1%) |

**Analysis:**
- **Prefill regression with always-on kernel:** The custom kernel's 1D grid is less parallel than MLX's native op pipeline for large token counts. Same pattern as the GDN fused kernel.
- **Decode-only kernel is neutral:** With a size threshold (totalElements <= 32768) to restrict to decode, the improvement vanishes. This means the "always-on" decode gain was likely a thermal/scheduling artifact from the slower prefill.
- **Root cause:** MLX's lazy evaluation already fuses these 3 simple element-wise ops effectively. The dispatch overhead for split + activation + multiply at decode sizes (2 experts x 2048 hidden = 4096 elements) is negligible. Custom kernels only help when fusing many ops (4-6+) with complex access patterns (like the GDN fused kernel).

**Decision:** Kernel code retained but disabled by default (`MOE_FUSED_ACTIVATION=1` to enable). Not worth the complexity for <1% benefit.

### Qwen3.5-35B with silu Fused Kernel

Tested fused silu kernel on Qwen3.5. **-3-5% decode regression.** Disabled.

## 1c. Option D — Sort Threshold A/B

Already env-var configurable (`MOE_SORT_THRESHOLD`, default 128). Benchmarks deferred to dedicated sort threshold sweep.

## 1d. Option E — Int8 vs FP16 vs Int4 Matmul Microbenchmark

### Expert Projections (K=2560, N=1536 — Qwen3.5 expert dimensions)

| Tokens (M) | FP16 ms | Int8 ms | Int4 ms | Int8/FP16 | Int4/FP16 |
|-----------|---------|---------|---------|-----------|-----------|
| 1 | 0.329 | 0.273 | 0.250 | 0.83x | 0.76x |
| 32 | 0.432 | 0.310 | 0.322 | 0.72x | 0.74x |
| 128 | 0.459 | 0.398 | 0.395 | 0.87x | 0.86x |
| 512 | 0.678 | 0.751 | 0.750 | **1.11x** | **1.11x** |
| 1024 | 1.102 | 1.262 | 1.279 | **1.15x** | **1.16x** |
| 2048 | 2.009 | 2.375 | 2.808 | **1.18x** | **1.40x** |
| 4096 | 3.996 | 4.805 | 5.191 | **1.20x** | **1.30x** |

### LM Head Projection (Gemma4: K=2816, N=262144)

| Tokens (M) | FP16 ms | Int8 ms | Int4 ms | Int8/FP16 | Int4/FP16 |
|-----------|---------|---------|---------|-----------|-----------|
| 1 | 30.15 | 2.39 | 1.45 | 0.08x | **0.05x** |
| 32 | 20.03 | 6.76 | 7.45 | 0.34x | 0.37x |
| 128 | 39.09 | 25.01 | 24.92 | 0.64x | 0.64x |

### LM Head Projection (Qwen3.5: K=2560, N=151936)

| Tokens (M) | FP16 ms | Int8 ms | Int4 ms | Int8/FP16 | Int4/FP16 |
|-----------|---------|---------|---------|-----------|-----------|
| 1 | 3.66 | 1.38 | 0.91 | 0.38x | **0.25x** |

### Key Findings

1. **Crossover point at ~256-512 tokens.** Below this, quantized matmul wins (memory-bound regime — dequant hidden by latency). Above this, FP16 wins (compute-bound — dequant overhead visible).

2. **Int8 degrades more gracefully than Int4 at large T.** At 4096 tokens: Int8 is 1.20x FP16 vs Int4 at 1.30x. The simpler Int8 byte unpacking costs less under compute pressure than Int4's sub-byte extraction.

3. **LM head quantization is massively beneficial.** Gemma4's 262K vocab projection: Int4 is 20.8x faster than FP16 at decode. Stays beneficial up to 128+ tokens because the huge N keeps the op memory-bound.

4. **Int4 wins for LM head decode, FP16 wins for prefill.** Clear regime separation.

### Actionable Items (added to Phase 2)

- **2e. Quantized LM Head for FP16/BF16 Models:** Quantize just `lm_head.weight` to Int4 in `sanitize()` even for non-quantized models. Expected: significant decode speedup for large-vocab models (Gemma4, Qwen3.5).
- **2f. Benchmark affine8 KV Cache:** Run `--kv affine8` to measure quality/speed tradeoff vs affine4. Int8 dequant is nearly free at decode batch sizes.
- **2g. TurboQuant Int8 Variant:** Investigate `turbo8v4` (8-bit keys, 4-bit values) for better quality at comparable speed.

## Phase 3 Tier 1 Results (Combined with Phase 1)

Changes: 3a (prefill 2048), 3b (FusedGateUpSwitchGLU for Gemma4), 3c (v_norm MLXFast.rmsNorm), 3e (peek cache).

### Gemma 4 E2B (dense 2B)

| Config | Metric | Baseline | Optimized | Delta |
|--------|--------|----------|-----------|-------|
| 1K none | Prefill | 1001 | 994 | -0.7% |
| 1K none | Decode | 76.7 | 80.4 | **+4.8%** |
| 1K none | TTFT | 1008ms | 1016ms | +0.8% |
| 4K none | Prefill | 1550 | 1632 | **+5.3%** |
| 4K none | Decode | 74.2 | 79.5 | **+7.1%** |
| 4K none | TTFT | 2711ms | 2574ms | **-5.1%** |
| 1K turbo | Prefill | 1003 | 993 | -1.0% |
| 1K turbo | Decode | 76.3 | 81.5 | **+6.8%** |
| 1K turbo | TTFT | 1006ms | 1017ms | +1.1% |
| 4K turbo | Prefill | 1564 | 1622 | +3.7% |
| 4K turbo | Decode | 73.0 | 78.4 | **+7.4%** |
| 4K turbo | TTFT | 2693ms | 2587ms | **-3.9%** |

### Gemma 4 26B-A4B (MoE)

| Config | Metric | Baseline | Optimized | Delta |
|--------|--------|----------|-----------|-------|
| 1K none | Prefill | 560 | 573 | +2.3% |
| 1K none | Decode | 29.3 | 31.1 | **+6.1%** |
| 1K none | TTFT | 2017ms | 1860ms | **-7.8%** |
| 4K none | Prefill | 612 | 617 | +0.8% |
| 4K none | Decode | 28.4 | 29.9 | **+5.3%** |
| 4K none | TTFT | 7065ms | 6956ms | -1.5% |
| 1K turbo | Prefill | 550 | 600 | **+9.1%** |
| 1K turbo | Decode | 29.4 | 31.2 | **+6.1%** |
| 1K turbo | TTFT | 2024ms | 1836ms | **-9.3%** |
| 4K turbo | Prefill | 608 | 612 | +0.7% |
| 4K turbo | Decode | 28.4 | 29.8 | **+4.9%** |
| 4K turbo | TTFT | 7035ms | 6947ms | -1.3% |

### Qwen3.5-35B-A3B and GPT-OSS-20B

No Gemma4-specific optimizations apply. Results within noise of baseline. GPT-OSS showed +3.4% prefill / -3.2% TTFT at 4K turbo from prefill chunk size increase.

### Prefill Chunk Size Sweep

Tested 512 (default), 1024, 2048, 4096 for pure attention models.

| Chunk Size | E2B 4K Prefill | 26B 4K Prefill | GPT-OSS 4K Prefill |
|-----------|---------------|---------------|-------------------|
| 512 | 1550 | 612 | 679 |
| 1024 | 1623 (+5%) | 617 (+1%) | 677 (0%) |
| 2048 | 1632 (+5%) | 610 (0%) | 687 (+1%) |
| 4096 | 947 (-39%) | 533 (-13%) | 607 (-11%) |

**Decision:** Use 2048 for pure attention models. 4096 causes massive regression from attention matrix size exceeding optimal GPU utilization.
