# TurboQuant+ Patched — Qwen3.5-35B-A3B 8bit

**Date**: 2026-04-04
**Branch**: `feature/turboquant-plus-optimizations` (24 commits)
**Hardware**: Apple M5 Max 128GB
**NR0**: 2

## Decode Speed (tok/s)

| Config | 128 | 1024 | 4096 | vs Baseline |
|--------|-----|------|------|-------------|
| Baseline (unpatched) | 78.7 | 77.9 | 76.1 | — |
| turbo3 NR0=2 | 77.9 | 76.7 | 76.3 | 99-100% |
| turbo4 NR0=2 | 78.0 | 76.4 | 76.6 | 99-101% |
| turbo0v4 (raw-K) | 76.9 | 77.2 | 76.3 | 98-100% |

## Prefill Speed (tok/s)

| Config | 128 | 1024 | 4096 |
|--------|-----|------|------|
| Baseline | 151.9 | 1,494 | 3,614 |
| turbo3 | 114.3 | 2,685 | 3,624 |
| turbo4 | 226.1 | 2,679 | 3,451 |
| turbo0v4 | 228.0 | 2,875 | 3,662 |

## KV Memory (4096 context)

| Config | KV | Savings |
|--------|-----|---------|
| Baseline | 935 MB | — |
| turbo3 | 190 MB | -80% |
| turbo4 | 248 MB | -74% |
| turbo0v4 | 935 MB | 0% (K uncompressed) |

## PPL (generation)

| Config | 128 | 1024 | 4096 |
|--------|-----|------|------|
| Baseline | 1.14 | 1.23 | 1.16 |
| turbo3 | 1.20 | 1.27 | 1.21 |
| turbo4 | 1.27 | 1.23 | 1.18 |
| turbo0v4 | **1.14** | **1.16** | 1.25 |

raw-K (turbo0v4) has best PPL at short context — confirms K precision matters.

## Summary

All configs match baseline decode (98-101%) on 35B MoE.
turbo3: best KV savings (80%) with minimal quality impact.
turbo0v4 (raw-K): best quality, no KV savings for K.
