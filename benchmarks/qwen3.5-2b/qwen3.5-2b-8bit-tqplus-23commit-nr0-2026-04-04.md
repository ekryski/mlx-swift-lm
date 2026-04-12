# NR0=2 Benchmark — Qwen3.5-2B 8bit turbo3

**Date**: 2026-04-04
**Hardware**: Apple M5 Max 128GB
**Commits**: 23 (includes NR0=2 multi-row amortization)

## NR0=2 vs NR0=1 (turbo3, decode tok/s)

| Context | NR0=1 | NR0=2 | Delta | vs Baseline |
|---------|-------|-------|-------|-------------|
| 128 | 157.3 | 158.0 | +0.4% | 99.5% |
| 1024 | 155.3 | 157.0 | +1.1% | 100.8% |
| 4096 | 148.3 | 153.9 | **+3.8%** | **100.0%** |

NR0=2 recovers the 4K decode regression completely.
turbo3 with NR0=2 = baseline decode speed at all context lengths.

## Full NR0=2 Results

| Context | Prefill | Decode | PPL | KV Cache |
|---------|---------|--------|-----|----------|
| 128 | 2,243 | 158.0 | 2.14 | 14 MB |
| 1024 | 9,058 | 157.0 | 1.85 | 54 MB |
| 4096 | 10,483 | 153.9 | 1.73 | 190 MB |
