# Spec 043 Phases 1–4 + Spec 042 — M1 Max implementation summary

**Branch:** `ek/specs-042-043-kernel-uplift`
**Hardware:** Apple M1 Max (applegpu_g13s, Apple GPU Family 7)
**Date:** 2026-05-15
**Status:** Spec 043 Phases 1, 2, 4 implemented + tested; Phase 3 scaffold; Spec 042 deferred (M2+ hardware required).

## Summary

Implementation of spec 043 Phases 1–4 across the 4-repo chain (`mlx` →
`mlx-c` → `mlx-swift` → `mlx-swift-lm`). Phase 4 (bias-aware
`turbo_flash_sdpa_v`) is the load-bearing landmark — it unlocks
GPT-OSS-20B (and any future sinks + bias-correcting model) on the A
path with `TURBO_DEQUANT_SDPA=0`.

The spec's projected per-cell perf lifts (Phase 1: +20–40%; Phase 2:
+5–10%; Phase 3: +10–25% small-model) **do not materialise on M1 Max**.
The cumulative bench data shows mean Δ < 1% per cell across all three
phases. This is consistent with the spec's own framing — every phase
that promised perf gains called out M2+ matrix engine / fp16-native
compute as the regime where the gains land. M1 Max's L1 cache already
absorbs the redundant device-memory loads Phase 1 targets, and the
software-fp16 path on M1 doesn't reward the Phase 2 typedef swap.

## A/B bench — Spec 043 Phase 1 vs baseline

3 samples per cell, 20s cooldown between samples, `--method
summarization --kv turbo4v2`.

| Model | Ctx | Baseline (tok/s) | Phase 1 (tok/s) | Δ |
|-------|----:|----:|----:|----:|
| qwen35-0.8b | 1024 | 129.2 | 127.4 | -1.4% |
| qwen35-0.8b | 8192 | 86.2 | 88.1 | +2.2% |
| gemma4-e2b | 1024 | 65.3 | 63.6 | -2.6% |
| gemma4-e2b | 8192 | 49.4 | 48.6 | -1.6% |
| gpt-oss-20b | 1024 | 67.9 | 67.9 | 0.0% |
| gpt-oss-20b | 8192 | 51.1 | 51.4 | +0.5% |
| **mean** | | | | **-0.5%** |

Per-cell |Δ| ≤ 2.6%. Within thermal-throttling noise on Qwen 0.8B (per-
sample ±11%); tight signal on Gemma 4 E2B (sub-1% per-sample variance,
consistent -2% Δ).

## A/B bench — Spec 043 Phase 2 vs Phase 1

Same shape; Phase 2 adds `typedef half ACC_T` for the `o[]` V
accumulator while keeping softmax `m`/`l` in fp32.

| Model | Ctx | Phase 1 | Phase 2 | Δ |
|-------|----:|----:|----:|----:|
| qwen35-0.8b | 1024 | 127.4 | 131.4 | +3.1% |
| qwen35-0.8b | 8192 | 88.1 | 84.5 | -4.1% |
| gemma4-e2b | 1024 | 63.6 | 63.3 | -0.5% |
| gemma4-e2b | 8192 | 48.6 | 48.1 | -1.0% |
| gpt-oss-20b | 1024 | 67.9 | 67.5 | -0.6% |
| gpt-oss-20b | 8192 | 51.4 | 51.6 | +0.4% |
| **mean** | | | | **-0.4%** |

## Phase 3 — headDim-aware tile autotune

Spec 043 Phase 3 proposed a per-(headDim) static table:
`64 → 32/64`, `128 → 64/128`, `256 → 128/256` for `(ctx≤4k / ctx>4k)`,
with a `turbo8v4`-specific `keyBits=8 → cap blockSize ≤ 64` override.

Implementing the spec's literal table regressed every cell measured —
Gemma 4 E2B × 1024 (headDim=256) went 63.3 → 49.5 tok/s (-22%)
because the spec's `blockSize=128` lost to the current adaptive's
`blockSize=32` at 1k context.

The starting-point values in the spec are heuristic estimates; per-
shape micro-sweep validation on the target hardware is required to
land them. Without that infrastructure I shipped Phase 3 as a 2-arg
overload (`adaptiveBlockSize(tokenCount: dim: keyBits:)`) that forwards
to the existing `adaptiveBlockSize(tokenCount:)`. The dispatch surface
is in place so a future per-shape sweep just needs to swap the body.

## Phase 4 — Bias-aware `turbo_flash_sdpa_v`

End-to-end works. GPT-OSS-20B `--kv turbo4v2`:

| Path | Ctx | Decode tok/s | Output coherence |
|------|----:|----:|---|
| B (`TURBO_DEQUANT_SDPA=1`, existing baseline) | 1024 | 65.8 | ✓ Harmony reasoning |
| A (`TURBO_DEQUANT_SDPA=0`, new bias-aware kernel) | 1024 | 64.1 | ✓ Harmony reasoning |
| A (new) | 8192 | 30.0 | ✓ Harmony reasoning |

A path is **slower than B path on M1 Max** because Phases 1–3 didn't
lift A-path performance on this hardware. The spec's perf gate ("A ≥ B
after Phases 1–3 land") requires M2+; on M1 Phase 4 is a correctness
landmark (GPT-OSS now works on A path) rather than a speed win.

Kernel-level parity gate: new `testTurboFlashSDPAvBiasMatchesReference`
walks 3 `(KeyBits, ValueBits, Dim)` shapes; worst-case
`rtol = 0.003`, well below the 0.1 acceptance threshold.

## Spec 042 — deferred

Spec 042 is the broader Metal-kernel SIMD audit:

| Phase | What | M1 status |
|-------|------|-----------|
| 1a | TurboFlash → MMA (`simdgroup_matrix_*`) | M2+ only; M1 keeps scalar |
| 1b | Affine flash kernel → MMA (subsumes spec 041 Phase 1.2) | M2+ only |
| 2 | `turbo_dequant_rotated` → MMA | M2+ only |
| 3 | `mse_score` / `mse_weighted_sum` → MMA (fallback paths) | M2+ only |
| 4 | `fused_encode_dispatch` profiling + tuning | Bench-driven; M1 inconclusive |

Every spec 042 phase depends on `simdgroup_matrix_multiply_accumulate`
intrinsics, which are M2+ Apple GPU Family 8+. The spec explicitly
calls out a per-`__metal_arch__` branch in the kernel template — M1
keeps the scalar path, M2+ takes MMA. Implementing the MMA path on
M1-only hardware means writing untestable code; deferring to a
session with M2+ access lets the MMA paths be validated as they're
written.

## Commits (chronological)

| Repo | SHA | Phase | What |
|------|-----|-------|------|
| mlx | `37aebde7` | 043 Phase 1 | Per-simdgroup TG packed-word cache + codebook hoist (5 templates) |
| mlx-swift | `8da280a` | 043 Phase 1 | Submodule bump + mirror sync |
| mlx-swift-lm | `c8773bc` | 043 Phase 1 | `testTurboFlashSDPAvBitUnpackReuseAcrossShapes` regression probe |
| mlx | `906ba14e` | 043 Phase 2 | fp16 V accumulator in `turbo_flash_sdpa_v` |
| mlx-swift | `c1f6fa9` | 043 Phase 2 | Submodule bump + mirror sync |
| mlx-swift-lm | `88d2e29` | 043 Phase 3 | `adaptiveBlockSize(tokenCount:dim:keyBits:)` scaffold |
| mlx-c | `3874217` | 043 Phase 4 | `mlx_fast_turbo_flash_sdpa_v` bias ABI extension |
| mlx | `44cb4ccd` | 043 Phase 4 | `tf_has_bias` fc(62) + kernel inner-loop bias add |
| mlx-swift | `aa08d1c` | 043 Phase 4 | Submodule bumps + `MLXFast.turboFlashSDPAv(..., keyBias:...)` |
| mlx-swift-lm | `845b1fb` | 043 Phase 4 | `TurboQuantKVCache` dispatcher routes A path + Swift parity test |

## What lifts on M2+

Per spec, the win curves should look like:

- Phase 1: 20–40% on long-context turbo cells (Qwen 0.8B turbo8v4 8k
  -56% baseline → ≤ -25% target).
- Phase 2: +5–10% across the board.
- Phase 3: +10–25% on small-headDim cells once per-shape sweep lands.
- Phase 4: A ≥ B on GPT-OSS-20B; Phase 1–3 lifts become available to
  GPT-OSS.
- Spec 042 Phase 1a: 8× FP16 throughput vs scalar SIMD via
  matrix-engine intrinsics — the largest single hand-rolled-kernel
  lift.

All of these are M2+-hardware-bound; a session on M2 Max / M3 Max /
M4 Max would re-bench this work and either confirm the spec's targets
or surface new findings.
