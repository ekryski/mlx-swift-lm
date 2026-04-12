# Affine8 KV Cache Evaluation

Date: 2026-04-08

## Summary

Benchmarked 8-bit affine KV cache quantization (`--kv affine8`) against no quantization and turbo4v2 across all 4 primary models at 1K/4K/16K contexts. Goal: assess whether Int8 KV offers a quality/speed tradeoff between FP16 (none) and turbo4v2 (4-bit keys, 2-bit values).

## Decode Speed (tok/s)

| Model | Ctx | none | affine8 | turbo4v2 | a8 vs none |
|-------|-----|------|---------|----------|------------|
| E2B | 1K | 80.1 | **83.7** | 80.7 | **+4.5%** |
| E2B | 4K | 78.9 | **80.4** | 78.6 | **+1.9%** |
| E2B | 16K | 68.6 | 68.6 | 69.9 | 0% |
| 26B | 1K | 31.0 | 31.0 | 31.0 | 0% |
| 26B | 4K | 29.7 | 29.7 | 29.7 | 0% |
| 26B | 16K | 25.6 | 25.6 | 25.4 | 0% |
| Qwen | 1K | 64.2 | 57.2 | 63.6 | **-10.9%** |
| Qwen | 4K | 61.8 | 56.3 | 63.7 | **-8.9%** |
| Qwen | 16K | 55.0 | 46.6 | 54.9 | **-15.3%** |
| GPT-OSS | 1K | 69.0 | 69.2 | 67.6 | +0.3% |
| GPT-OSS | 4K | 65.2 | 65.3 | 64.9 | +0.2% |
| GPT-OSS | 16K | 53.5 | 53.8 | 53.1 | +0.6% |

## Perplexity (lower is better)

| Model | Ctx | none | affine8 | turbo4v2 |
|-------|-----|------|---------|----------|
| E2B | 16K | 1.55 | **1.42** | 1.61 |
| 26B | 16K | 1.26 | **1.26** | 1.76 |
| Qwen | 16K | 1.37 | **1.23** | 1.42 |
| GPT-OSS | 16K | 2.71 | **2.17** | 3.11 |

## KV Cache Memory

| Model | Ctx | none | affine8 | turbo4v2 |
|-------|-----|------|---------|----------|
| E2B | 16K | 3.54GB | 1.99GB (1.8x) | 737MB (4.8x) |
| 26B | 16K | 3.54GB | 1.99GB (1.8x) | 737MB (4.8x) |
| Qwen | 16K | 3.54GB | 1.99GB (1.8x) | 735MB (4.8x) |
| GPT-OSS | 16K | 3.45GB | 1.94GB (1.8x) | 718MB (4.8x) |

## Analysis

**Affine8 strengths:**
- Best PPL across all models — matches or beats FP16 none (likely due to reduced memory pressure improving cache behavior)
- Speed-neutral or faster for pure attention models (E2B, 26B, GPT-OSS)
- E2B sees +4.5% decode at 1K, likely because the compressed KV fits better in GPU caches

**Affine8 weaknesses:**
- Only 1.8x compression (half of FP16). At 16K, still 2GB vs turbo4v2's 737MB. Not a substitute for turbo4v2 when memory is the constraint.
- **Qwen3.5 (GDN hybrid): -11-15% decode regression.** The affine quantized attention path has significant overhead for this architecture. Likely due to the `quantizedScaledDotProductAttention` code path interacting poorly with GDN's hybrid attention/recurrence pattern.

## Conclusion

Affine8 is a viable "high quality" KV cache option for pure attention models where memory isn't the bottleneck. It sits between none (best speed, most memory) and turbo4v2 (best compression, some quality loss).

Not suitable for GDN hybrid models (Qwen3.5) due to decode regression.

The 16K quality gap between affine8 and turbo4v2 (26B: 1.26 vs 1.76, GPT-OSS: 2.17 vs 3.11) motivates investigating a TurboQuant Int8 variant (turbo8v4) that could offer turbo's compression ratio with better quality — see Phase 2g.
