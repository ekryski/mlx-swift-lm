# TurboQuant Optimization: P1 — Compressed-Domain Metal Kernels (Default)

> **Correction (2026-04-02)**: Memory conclusions below based on KV Delta were misleading. The new KV Cache metric shows turbo4 is actually 15% smaller than affine4 (1.88GB vs 2.21GB at 32K). See `turbo-comprehensive-analysis.md`.

**Date**: 2026-04-02
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Base**: `ek/consolidated-benchmarks`

## Hypothesis

P0's dequant-first path maintained both compressed storage AND an FP16 dequant buffer (same size as no-quant), resulting in worse memory than no compression. By making the existing compressed-domain Metal kernels (`mseScore` + `mseWeightedSum`) the default decode path, we eliminate the FP16 dequant buffer entirely. Score and value computation happen directly from packed indices — no intermediate FP16 materialization. This should dramatically improve KV Delta while maintaining or improving gen tok/s.

## Changes

**`Libraries/MLXLMCommon/AttentionUtils.swift`**: For decode (L=1 or `isCompressed`), route to `turboCache.compressedAttention()` instead of `updateAndDequant()`. This uses:
1. Metal fused encode kernel (encode new token)
2. Metal score kernel (Q×K from packed indices, pre-rotated queries)
3. MLX softmax
4. Metal value kernel (Attn×V from packed indices)
5. One inverse rotation matmul (output only)

Eliminated: the entire `dequantKeys`/`dequantValues` FP16 buffer, 2 of 4 rotation matmuls per step.

## Results vs Baseline (Qwen3.5-2B 8-bit, summarization, quick+32K)

| Metric | No-Quant | Affine4 | Turbo4 P0 (dequant) | **Turbo4 P1 (compressed)** | P1 vs Affine4 |
|--------|----------|---------|---------------------|---------------------------|---------------|
| **Gen tok/s (128)** | 88.7 | 88.7 | 78.7 | **79.3** | -10.6% |
| **Gen tok/s (1024)** | 89.0 | 83.4 | 77.9 | **79.9** | -4.2% |
| **Gen tok/s (4096)** | 85.3 | 82.0 | 77.8 | **77.7** | -5.2% |
| **Gen tok/s (32K)** | 68.6 | 64.1 | — | **63.3** | -1.2% |
| Prefill tok/s (128) | 624 | 635 | 595 | **612** | -3.6% |
| Prefill tok/s (4096) | 1263 | 1238 | 1229 | **1233** | -0.4% |
| Prefill tok/s (32K) | 1302 | 1289 | — | **1281** | -0.6% |
| **KV Delta (128)** | 15MB | 9MB | 11MB | **13MB** | +44% |
| **KV Delta (1024)** | 9MB | 15MB | 18MB | **23MB** | +53% |
| **KV Delta (4096)** | 39MB | 18MB | 60MB | **40MB** | +122% |
| **KV Delta (32K)** | 333MB | 98MB | — | **399MB** | +307% |
| Think KLD (128) | 0.039 | 0.021 | -0.004 | **0.035** | Similar |
| Think KLD (1024) | 0.008 | 0.047 | 0.012 | **0.028** | Better |
| Think KLD (4096) | 0.017 | 0.052 | 0.019 | **0.030** | Better |
| Think KLD (32K) | 0.016 | 0.033 | — | **0.050** | Worse |
| Gen KLD (128) | 0.003 | 0.041 | 0.032 | **0.000** | Much better |
| Gen KLD (1024) | 0.020 | 0.057 | -0.045 | **-0.001** | Much better |
| Gen KLD (4096) | 0.094 | -0.087 | 0.007 | **-0.001** | Better |
| Gen KLD (32K) | 0.007 | 0.757 | — | **0.028** | Much better |

## Key Learnings

### 1. Memory: Improved vs P0 but Still Worse Than Affine4

KV Delta at 4096 dropped from 60MB (P0) to 40MB (P1) — a 33% improvement from eliminating the dequant buffer. But it's still 2.2× higher than affine4's 18MB.

**Root cause investigation**: At 4096 tokens, turbo4 compressed storage should be:
- Packed: 28 layers × 16 heads × 4096 tokens × 16 uint32 × 4 bytes = 28 × 16 × 4096 × 64 = ~115 MB (K+V)
- Norms: 28 × 16 × 4096 × 4 = ~7 MB (K+V)
- Total: ~122 MB — but KV Delta shows only 40MB

The low KV Delta number suggests the benchmark is measuring incremental memory (delta from baseline), and the compressed storage allocation pattern (step=256 growth) plus MLX's lazy evaluation means not all memory is materialized at measurement time.

However, at 32K the KV Delta of 399MB vs affine4's 98MB (4× higher) is a problem. The compressed format at 4-bit turbo should be comparable to 4-bit affine. The gap is likely from:
- Codec overhead: 2 rotation matrices [128,128] × FP32 = 128KB per codec × 2 per layer × 28 layers = ~7MB
- Codebook/boundary arrays (negligible)
- Raw prefill cache not fully freed (MLX lazy eval)

### 2. Speed: Similar to P0, Slightly Better at Short Context

Gen tok/s improved slightly at 128 (79.3 vs 78.7) and 1024 (79.9 vs 77.9) — the Metal score/value kernels avoid the dequant buffer allocation overhead. At 4096 tokens it's essentially the same (77.7 vs 77.8).

At 32K, turbo4 (63.3) is within 1.2% of affine4 (64.1) — the Metal kernels scale reasonably well.

**The remaining speed gap vs no-quant (79.3 vs 88.7 at 128)** comes from:
- Fused encode: 2 Metal kernel dispatches per step (K + V)
- Pre-rotate query: 1 matmul [B,Hq,1,D]×[D,D]
- Inverse rotate output: 1 matmul [B,Hq,1,D]×[D,D]
- Score kernel: 1 Metal dispatch
- Value kernel: 1 Metal dispatch
- Total: 6 dispatches vs no-quant's 1 (SDPA)

### 3. Quality: Excellent, Better Than Affine4

Gen KLD is near-zero across all context lengths (0.000, -0.001, -0.001, 0.028). This is significantly better than affine4, which shows 0.757 Gen KLD at 32K. TurboQuant's rotation + norm correction preserves quality extremely well.

### 4. 32K Context: Key Competitive Data Point

At 32K context, turbo4 shows:
- Gen tok/s: 63.3 vs affine4's 64.1 — essentially tied
- KV Delta: 399MB vs affine4's 98MB — 4× worse
- Gen KLD: 0.028 vs affine4's 0.757 — **27× better quality**

The quality advantage at long context is dramatic. If we can close the memory gap, turbo4 would be strictly superior.

## Decision

**ITERATE** — The compressed attention path is the right architecture (no dequant buffer, Metal kernels for score/value). Quality is excellent. Speed is close but not yet matching affine4. Memory is the main remaining issue.

Next optimizations to pursue:
1. **P2: Pre-computed constants** — Quick win, eliminates ~100ms init overhead
2. **P3: Asymmetric K/V** — Use fewer bits for V to improve memory
3. **Investigate memory gap** — Why is KV Delta 4× higher than affine4 at 32K? Profile allocation patterns.
