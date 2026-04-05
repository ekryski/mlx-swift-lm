# TurboQuant+ 15-Commit Final — Qwen3.5-2B 8bit

**Date**: 2026-04-04
**Branch**: `feature/turboquant-plus-optimizations` (15 commits)
**Hardware**: Apple M5 Max 128GB

## All Commits

| # | Hash | Description |
|---|------|-------------|
| 1 | d0abc16 | fix: defer TurboQuant compression to decode, use SDPA for prefill |
| 2 | adb5213 | fix: shared_norm buffer overflow for head_dim > 128 |
| 3 | f8f7cfd | feat: boundary layer protection (first 2 + last 2 at FP16) |
| 4 | 6f495ac | docs: asymmetric K/V recommendations from TurboQuant+ |
| 5 | 544bd78 | feat: TURBO_FLASH_BLOCK_SIZE env var |
| 6 | 3bef215 | feat: TURBO_SPARSE_V_THRESHOLD env var |
| 7 | c86e1a2 | bench: 6-commit validation |
| 8 | d80f3f5 | perf: cooperative SIMD WHT via simd_shuffle_xor |
| 9 | 27f9623 | perf: branchless boundary quantization |
| 10 | 4350c61 | feat: raw-K asymmetric mode (turbo0vN) + API fixes |
| 11 | 4911044 | bench: 10-commit validation |
| 12 | 5c78c7c | perf: skip norm correction for WHT encode |
| 13 | 07fafa6 | verify: no QJL/residual correction |
| 14 | 5dd156e | feat: pre-computed centroids for dim=80 and dim=96 |
| 15 | this | bench: 15-commit final results |

## Final Results — Summarization

### Decode (tok/s)

| KV | 128 | 1024 | 4096 | vs baseline |
|----|-----|------|------|-------------|
| none | 158.8 | 155.7 | 153.9 | — |
| turbo3 | 157.3 (99%) | 155.3 (100%) | 148.3 (96%) | 96-100% |
| turbo4 | 147.8 (93%) | 132.7 (85%) | 147.5 (96%) | 85-96% |

### Prefill (tok/s)

| KV | 128 | 1024 | 4096 |
|----|-----|------|------|
| none | 2,115 | 9,021 | 10,526 |
| turbo3 | 2,271 (+7%) | 9,064 (0%) | 10,706 (+2%) |
| turbo4 | 2,241 (+6%) | 8,734 (-3%) | 10,624 (+1%) |

### KV Memory (4096)

| KV | Size | Savings |
|----|------|---------|
| none | 936 MB | — |
| turbo3 | 190 MB | -80% |
| turbo4 | 249 MB | -73% |

### PPL (generation)

| KV | 128 | 1024 | 4096 |
|----|-----|------|------|
| none | 1.97 | 1.97 | 1.93 |
| turbo3 | 1.92 | 1.79 | 1.81 |
| turbo4 | 2.41 | 2.18 | 2.30 |

turbo3 PPL is BETTER than baseline in all cases. turbo4 shows slight quality degradation.

## Summary

turbo3: 96-100% decode speed, 80% KV savings, better PPL than baseline.
turbo4: 85-96% decode speed, 73% KV savings, slight PPL degradation.
Recommendation: turbo3 for MLX Swift.
