# MoE Sort Threshold A/B Test

**Date**: 2026-04-05
**Model**: Qwen3.5-35B-A3B 4-bit, no-quant KV
**Hardware**: Apple M1 Max, 64GB RAM
**Variable**: `MOE_SORT_THRESHOLD` env var controlling `SwitchGLU` expert reordering

## What the sort does

`SwitchGLU.callAsFunction` calls `gatherSort()` when `indices.size >= threshold` to
reorder tokens by expert index before calling `gatherQuantizedMM`. This groups tokens
going to the same expert together, improving memory access locality for the quantized
weight tensor. After computation, `scatterUnsort()` restores original token order.

- **Default threshold**: 64 (sort when indices.size >= 64)
- **Decode** (batch=1, topK=8): indices.size=8 < 64 → never sorts (unaffected)
- **Prefill** (T=4096, topK=8): indices.size=32,768 → always sorts

## Results

### A: Sort ENABLED (default, threshold=64)

| Context | Prefill tok/s | Gen tok/s | TTFT |
|---------|--------------|-----------|------|
| 128 | 155.5 | 47.5 | 754ms |
| 1024 | **472.7** | 47.4 | 2,462ms |
| 4096 | **502.4** | 45.7 | 8,642ms |
| 32768 | **399.4** | 35.1 | 89,884ms |

### B: Sort DISABLED (threshold=0)

| Context | Prefill tok/s | Gen tok/s | TTFT |
|---------|--------------|-----------|------|
| 128 | 193.7 | 47.3 | 606ms |
| 1024 | **268.5** | 47.1 | 4,206ms |
| 4096 | **262.6** | 45.3 | 16,042ms |
| 32768 | **246.5** | 35.1 | 137,332ms |

### Comparison

| Context | Sort ON | Sort OFF | Delta | Conclusion |
|---------|---------|----------|-------|------------|
| 128 | 155.5 | 193.7 | +25% without sort | Sort overhead > locality benefit at small T |
| 1024 | 472.7 | 268.5 | **-43% without sort** | Sort is critical at medium T |
| 4096 | 502.4 | 262.6 | **-48% without sort** | Sort essential for large prefill |
| 32768 | 399.4 | 246.5 | **-38% without sort** | Sort essential at very large T |

Decode speed is identical (47 tok/s) — sort is not invoked at decode batch size.

## Key Findings

1. **Sorting is critical for prefill performance.** Disabling sort causes 38-48% prefill
   regression at T >= 1024. The `gatherQuantizedMM` memory access pattern benefits enormously
   from expert-grouped token ordering.

2. **At T=128**, sort overhead slightly exceeds its benefit (155 → 194 tok/s without sort).
   This suggests the sort threshold of 64 is slightly too aggressive for very small T.
   A threshold of ~256-512 might give the best of both worlds.

3. **Decode is unaffected** — sort is already skipped at batch=1 (indices.size=8 < 64).

4. **Do not increase the threshold above ~256.** The 1024-context result shows that sorting
   is essential even at moderate prefill sizes. The current default of 64 is safe and
   conservative — only the 128-token case shows marginal overhead.

## Recommendation

Keep default threshold at 64. The 25% improvement at T=128 from disabling sort is marginal
(155 → 194 tok/s, only 0.15s TTFT difference) while the 43-48% regression at larger contexts
would be devastating. The current default is the right tradeoff.
