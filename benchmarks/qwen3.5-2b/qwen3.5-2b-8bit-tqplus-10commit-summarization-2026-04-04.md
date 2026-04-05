# TurboQuant+ 10-Commit Patch — Qwen3.5-2B 8bit

**Date**: 2026-04-04
**Branch**: `feature/turboquant-plus-optimizations` (10 commits)
**Hardware**: Apple M5 Max 128GB

## Commits

1. fix: defer TurboQuant compression to decode, use SDPA for prefill
2. fix: shared_norm buffer overflow for head_dim > 128
3. feat: boundary layer protection (first 2 + last 2 at FP16)
4. docs: asymmetric K/V recommendations
5. feat: TURBO_FLASH_BLOCK_SIZE env var
6. feat: TURBO_SPARSE_V_THRESHOLD env var
7. bench: 6-commit validation results
8. perf: cooperative SIMD WHT via simd_shuffle_xor
9. perf: fused inv_norm*sign + branchless boundary quantization
10. feat: raw-K asymmetric mode (turbo0vN) + API fixes

## Results — Summarization (128, 1024, 4096)

### Decode Speed (tok/s)

| KV Config | 128 | 1024 | 4096 | vs baseline |
|-----------|-----|------|------|-------------|
| none (baseline) | 156.6 | 154.8 | 152.5 | — |
| turbo4 | 150.2 | 144.9 | 144.3 | 95% |
| turbo3 | 158.7 | 156.0 | 152.2 | **100%** |

### Prefill Speed (tok/s)

| KV Config | 128 | 1024 | 4096 |
|-----------|-----|------|------|
| none | 2,213 | 8,985 | 10,656 |
| turbo4 | 2,409 | 8,567 | 10,081 |
| turbo3 | 2,420 | 8,997 | 10,616 |

### KV Memory (4096 context)

| KV Config | KV Cache | Savings |
|-----------|----------|---------|
| none | 936 MB | — |
| turbo4 | 249 MB | -73% |
| turbo3 | 190 MB | -80% |

### Headline

**turbo3 = 100% baseline decode speed with 80% less KV memory.**
