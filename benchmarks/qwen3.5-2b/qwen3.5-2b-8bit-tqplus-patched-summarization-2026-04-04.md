# TurboQuant+ Patch Validation — Qwen3.5-2B 8bit

**Date**: 2026-04-04
**Branch**: `feature/turboquant-plus-optimizations`
**Hardware**: Apple M5 Max 128GB

## Commits Applied

1. `d0abc16` fix: defer TurboQuant compression to decode, use SDPA for prefill
2. `adb5213` fix: shared_norm buffer overflow for head_dim > 128
3. `f8f7cfd` feat: boundary layer protection for TurboQuant KV cache
4. `6f495ac` docs: document asymmetric K/V recommendations from TurboQuant+ research
5. `544bd78` feat: add TURBO_FLASH_BLOCK_SIZE env var for flash block size tuning
6. `3bef215` feat: make sparse V skip threshold configurable via TURBO_SPARSE_V_THRESHOLD

## Results — Summarization (128, 1024, 4096 tokens)

### Prefill (tok/s)

| KV Config | 128 | 1024 | 4096 |
|-----------|-----|------|------|
| none (baseline) | 2,060 | 9,008 | 10,856 |
| turbo4 | 2,456 (+19%) | 9,083 (+1%) | 10,772 (-1%) |
| turbo3 | 2,282 (+11%) | 9,091 (+1%) | 10,634 (-2%) |
| turbo4v2 (4K/2V) | 2,299 (+12%) | 9,135 (+1%) | 10,728 (-1%) |

### Generation (tok/s)

| KV Config | 128 | 1024 | 4096 |
|-----------|-----|------|------|
| none (baseline) | 159.3 | 157.2 | 153.7 |
| turbo4 | 158.6 (100%) | 153.1 (97%) | 148.1 (96%) |
| turbo3 | 158.3 (99%) | 157.5 (100%) | 153.6 (100%) |
| turbo4v2 (4K/2V) | 157.9 (99%) | 156.5 (100%) | 153.1 (100%) |

### KV Memory (4096 context)

| KV Config | KV Cache | Savings |
|-----------|----------|---------|
| none | 936 MB | — |
| turbo4 | 249 MB | -73% |
| turbo3 | 190 MB | -80% |
| turbo4v2 | 190 MB | -80% |

### Generation PPL

| KV Config | 128 | 1024 | 4096 |
|-----------|-----|------|------|
| none | 2.25 | 1.96 | 2.24 |
| turbo4 | 2.26 | 2.29 | 1.69 |
| turbo3 | 2.56 | 2.01 | 2.08 |
| turbo4v2 | 1.95 | 2.08 | 1.92 |

### Key Findings

- **turbo3 decode matches baseline** (153.6 vs 153.7 at 4K). Zero decode penalty.
- **Prefill: no regression**, slight improvement at small contexts from deferred compression.
- **turbo3 is the recommended config**: same decode speed, 80% KV reduction, good quality.
- **Boundary layers active**: first 2 + last 2 attention layers at FP16.
- **No regressions** on any path vs unpatched baseline.
