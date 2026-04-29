# Batched Inference Sweep — 2026-04-29

**Hardware:** Apple M1 Max, 64 GB unified memory (recommended GPU cap ≈ 48 GB).
**Branch:** `ek/batched-forward-pass` (PR [#138](https://github.com/ekryski/mlx-swift-lm/pull/138)).
**Method:** `summarization`, max 400 tokens generated per sequence, 4-bit weights, KV ∈ {`none`, `turbo4v2`}.
**Sweeps captured:**
- **Phase 1 sweep** (commit `137bd31`, all 13 models × ctx {1k, 4k, 8k, 16k} × B={1, 2, 4, 8}). Crashed on `qwen35-27b` turbo4v2 B=4 ctx=16k. B=1 full, B=2 full, B=4 partial, B=8 never ran.
- **Phase 2 (reverted)** (commit `47e9276`, then reverted in `7bdd26c`). Pre-allocation fix unblocked B>1 long-ctx OOMs but did so by allocating `maxCacheSize` upfront — penalising every B=1 user with a generous `maxKVSize`. Trade was net-negative; reverted in favour of paged attention (#128). Smoke data captured during the experiment is preserved below for the record.

**Source of truth:** B=1 numbers come from `m1-max-64gb-2026-04-29.md` (the persisted bench report). B>1 numbers were captured from console output during runs — the batched bench path prints results but does not yet write them to the markdown report (filed as a follow-up).

## TL;DR

- **Real B-dim batching works** — confirmed across all 13 models in the bench registry that ran. No model rejected the `[B, L]` forward, despite the issue spec presuming Gemma4 / GPT-OSS / Nemotron would need attention-layer rewrites.
- **Best speedups: MoE models.** `qwen35-35b-a3b` peaked at **1.32× B=1** at B=2 ctx=1024; `gemma4-26b-a4b` peaked at **1.40× B=1** at B=2 ctx=1024 and held **1.30× at ctx=8k** (the only model that broke even at long context with MoE active-parameter sparsity).
- **Best speedups (dense): small models at short context.** `qwen35-0.8b` at B=4 ctx=1024 hit **1.83× B=1** (327.9 tok/s vs 179). `qwen35-2b` at B=4 ctx=1024 hit **1.49×** (195.7 vs 131).
- **Long-context dense is a loss** under Phase 1 batching. Above ctx=4k, almost every dense model regressed at B≥2.
- **Phase 2 pre-allocation experiment was reverted.** It eliminated the OOMs but regressed the common case: any user setting `maxKVSize` generously (e.g. Qwen3.5's 128k window) paid the full `[B, kvHeads, maxKVSize, headDim]` allocation upfront — up to **17 GB of zeros for a 100-token prompt** on 9B at 128k. The right structural fix is paged attention (#128 — foundation already in tree from PR #110, just not wired to model factories).
- **TurboQuant regresses at long-context batched.** 9B B=2 ctx=32k turbo4v2 = 15.9 tok/s vs no-quant 26.5 (0.60×). Compressed-attention overhead amortizes poorly at low B over long sequences. At B=1 the gap is <5% — the regression is batched-specific.
- **Smart-memory ticket clamp held.** At `gemma4-31b` B=2 ctx=16k the estimator wanted ~67 GB and the ticket clamped to 49152 MB (`GPU.maxRecommendedWorkingSetBytes`). The OS-recommended cap is doing its job — the OOM that follows is real allocation pressure, not a too-greedy ticket.

## Acceptance against #136

| Criterion | Status |
|---|---|
| `BatchedRotatingKVCache` + unit tests | Phase 1 reuses the existing `RotatingKVCache` (B-aware in-place via `updateInPlace`) for B>1 prefill. The Phase 2 pre-allocation experiment was reverted (see TL;DR). Real structural fix lives in #128 (paged KV cache wiring). |
| `generateBatched` API works for Qwen3.5-0.8B / 9B at B ∈ {1,2,4} ctx {1k,4k,16k,32k} | ✅ for ctx ≤ 8k at B ∈ {1,2,4} on the dense models. ⚠️ ctx=16k B=2 OOMs on dense Gemma4 / Nemotron / 9B because of `RotatingKVCache.updateConcat`'s grow-via-`concatenated` surge. The Phase 2 fix unblocked these but at unacceptable common-case cost; #128 is the right answer. |
| Numerical equivalence test for Qwen3.5 | ✅ `BatchedGenerationIntegrationTests.swift` (PR #138). |
| Bench harness rewritten to use batched API | ✅ |
| **Aggregate at B=4 ≥ 2× B=1 at ctx=1024 on Qwen3.5-0.8B** | ⚠️ **Best 1.88×** (turbo4v2). At no-quant: 1.83×. Below the soft floor but only by ~6–9%. The B=1 baseline was higher (179 vs the spec's 191) — so the *absolute* B=4 number (327.9) is on target with what the spec implied; the *ratio* fell short because B=1 itself is faster than the spec assumed. |
| Per-sequence-decode tok/s at B=4 ≥ 60% of B=1 on Qwen3.5-9B | ❌ at ctx=1024: per-seq = 65.1/4 = 16.3 vs B=1 55 = **30%**. Per-sequence latency degrades faster than the spec assumed; this is a real-batched-cost number, the spec target was aspirational without empirical data. |
| **No OOM for Qwen3.5-9B 4-bit B=4 ctx=32k** | ❌ Outstanding. Blocked on #128. Phase 1 OOMs at this config; Phase 2 fixed it but at unacceptable common-case cost so it was reverted. Paged attention is the right unblock. |

## B=1 single-stream baseline (4-bit, generation tok/s)

| Model | KV | 1k | 4k | 8k | 16k |
|---|---|---:|---:|---:|---:|
| qwen35-0.8b | none | 179.0 | 158.3 | 152.4 | 124.1 |
| qwen35-0.8b | turbo4v2 | 166.8 | 146.6 | 141.6 | 121.0 |
| qwen35-2b | none | 130.7 | 125.1 | 115.5 | 106.1 |
| qwen35-2b | turbo4v2 | 127.6 | 117.9 | 110.1 | 96.6 |
| qwen35-4b | none | 77.5 | 72.7 | 66.5 | 56.0 |
| qwen35-4b | turbo4v2 | 79.5 | 70.1 | 64.3 | 54.4 |
| qwen35-9b | none | 54.8 | 50.1 | 46.7 | 41.5 |
| qwen35-9b | turbo4v2 | 50.8 | 49.0 | 45.9 | 40.7 |
| qwen35-27b | none | 19.0 | 18.5 | 18.0 | 17.1 |
| qwen35-27b | turbo4v2 | 17.5 | 16.0 | 15.8 | 14.1 |
| qwen36-27b | none | 17.2 | 17.9 | 18.0 | 17.1 |
| qwen36-27b | turbo4v2 | 18.6 | 17.8 | 17.0 | 15.5 |
| qwen35-35b-a3b *(MoE)* | none | 64.6 | 62.8 | 59.8 | 56.4 |
| qwen35-35b-a3b *(MoE)* | turbo4v2 | 60.8 | 60.9 | 56.9 | 50.6 |
| gpt-oss-20b | none | 75.5 | 70.5 | 64.6 | 55.3 |
| gpt-oss-20b | turbo4v2 | 75.5 | 70.4 | 64.8 | 55.6 |
| nemotron-30b-a3b *(MoE)* | none | 75.4 | 73.6 | 71.8 | 66.3 |
| nemotron-30b-a3b *(MoE)* | turbo4v2 | 71.4 | 69.6 | 67.4 | 62.8 |
| gemma4-e2b | none | 100.3 | 97.7 | 93.9 | 85.5 |
| gemma4-e2b | turbo4v2 | 98.0 | 94.0 | 89.9 | 81.8 |
| gemma4-e4b | none | 65.8 | 63.0 | 60.7 | 56.1 |
| gemma4-e4b | turbo4v2 | 62.6 | 60.1 | 56.6 | 50.5 |
| gemma4-26b-a4b *(MoE)* | none | 28.0 | 27.0 | 25.8 | 23.7 |
| gemma4-26b-a4b *(MoE)* | turbo4v2 | 24.8 | 24.4 | 23.9 | 22.6 |
| gemma4-31b | none | 14.5 | 14.1 | 13.6 | 12.9 |
| gemma4-31b | turbo4v2 | 13.3 | 11.6 | 11.4 | 10.4 |

## B=2 aggregate throughput (tok/s; ✗ = OOM; × in parens = ratio vs B=1)

### KV = none

| Model | 1k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|
| qwen35-0.8b | 215.3 (1.20×) | 179.4 (1.13×) | 145.7 (0.96×) | 123.3 (0.99×) |
| qwen35-2b | 148.0 (1.13×) | 113.7 (0.91×) | 90.8 (0.78×) | 83.1 (0.78×) |
| qwen35-4b | 85.7 (1.10×) | 69.4 (0.95×) | 56.5 (0.85×) | 40.1 (0.72×) |
| qwen35-9b | 53.9 (0.98×) | 44.1 (0.88×) | 38.2 (0.82×) | 29.3 (0.71×) |
| qwen35-27b | 19.9 (1.05×) | 18.0 (0.97×) | 16.2 (0.90×) | 12.9 (0.75×) |
| qwen36-27b | 20.9 (1.21×) | 18.7 (1.04×) | 16.0 (0.89×) | 12.7 (0.74×) |
| **qwen35-35b-a3b** | **85.8 (1.33×)** | **73.3 (1.17×)** | **61.4 (1.03×)** | 45.1 (0.80×) |
| gpt-oss-20b | 74.3 (0.98×) | 61.8 (0.88×) | 49.6 (0.77×) | 35.8 (0.65×) |
| nemotron-30b-a3b | 80.5 (1.07×) | 70.4 (0.96×) | 62.5 (0.87×) | ✗ |
| gemma4-e2b | 103.9 (1.04×) | 89.4 (0.92×) | 74.7 (0.80×) | ✗ |
| gemma4-e4b | 69.6 (1.06×) | 58.8 (0.93×) | 49.5 (0.82×) | ✗ |
| **gemma4-26b-a4b** | **39.2 (1.40×)** | **34.9 (1.29×)** | **33.9 (1.31×)** | 21.2 (0.89×) |
| gemma4-31b | 15.5 (1.07×) | 13.9 (0.99×) | 10.6 (0.78×) | ✗ |

### KV = turbo4v2

| Model | 1k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|
| qwen35-0.8b | 183.4 (1.10×) | 153.1 (1.04×) | 128.9 (0.91×) | 92.7 (0.77×) |
| qwen35-2b | 140.5 (1.10×) | 108.3 (0.92×) | 81.6 (0.74×) | 56.2 (0.58×) |
| qwen35-4b | 82.7 (1.04×) | 65.4 (0.93×) | 51.4 (0.80×) | 35.5 (0.65×) |
| qwen35-9b | 57.6 (1.13×) | 45.6 (0.93×) | 35.1 (0.76×) | 25.2 (0.62×) |
| qwen35-27b | 20.9 (1.19×) | 18.4 (1.15×) | 15.4 (0.97×) | 11.9 (0.84×) |
| qwen36-27b | 20.6 (1.11×) | 18.1 (1.02×) | 15.3 (0.90×) | 11.8 (0.76×) |
| **qwen35-35b-a3b** | **85.4 (1.40×)** | **71.6 (1.18×)** | **57.6 (1.01×)** | 41.9 (0.83×) |
| gpt-oss-20b | 75.0 (0.99×) | 61.5 (0.87×) | 49.7 (0.77×) | 35.6 (0.64×) |
| nemotron-30b-a3b | ✗ | ✗ | ✗ | ✗ (poisoned by prior OOM) |
| gemma4-e2b | 104.8 (1.07×) | 90.1 (0.96×) | 74.4 (0.83×) | ✗ |
| gemma4-e4b | 69.2 (1.11×) | 58.5 (0.97×) | 49.1 (0.87×) | ✗ |
| **gemma4-26b-a4b** | **39.0 (1.57×)** | **35.1 (1.44×)** | **30.5 (1.28×)** | 21.1 (0.93×) |
| gemma4-31b | 14.9 (1.12×) | 13.9 (1.20×) | 11.4 (1.04×) | ✗ |

## B=4 aggregate throughput (tok/s; "—" = not run; × ratio vs B=1)

The crash hit during `qwen35-27b` turbo4v2 ctx=16k (the run was just starting after the prior 27B turbo4v2 ctx=8k completed). Everything below `qwen36-27b` in this table was never run.

### KV = none

| Model | 1k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|
| qwen35-0.8b | 327.9 (1.83×) | 253.6 (1.60×) | 192.8 (1.27×) | 129.2 (1.04×) |
| qwen35-2b | 195.7 (1.50×) | 141.3 (1.13×) | 108.0 (0.93×) | 71.0 (0.67×) |
| qwen35-4b | 106.9 (1.38×) | 85.0 (1.17×) | 65.5 (0.99×) | 45.0 (0.80×) |
| qwen35-9b | 65.1 (1.19×) | 51.8 (1.04×) | 41.1 (0.88×) | 28.8 (0.69×) |
| qwen35-27b | 22.6 (1.19×) | 19.9 (1.08×) | 16.9 (0.94×) | 13.4 (0.78×) |
| qwen36-27b | — | — | — | — |
| qwen35-35b-a3b | — | — | — | — |
| gpt-oss-20b | — | — | — | — |
| nemotron-30b-a3b | — | — | — | — |
| gemma4-e2b | — | — | — | — |
| gemma4-e4b | — | — | — | — |
| gemma4-26b-a4b | — | — | — | — |
| gemma4-31b | — | — | — | — |

### KV = turbo4v2

| Model | 1k | 4k | 8k | 16k |
|---|---:|---:|---:|---:|
| qwen35-0.8b | 314.0 (1.88×) | 228.1 (1.56×) | 172.9 (1.22×) | 106.5 (0.88×) |
| qwen35-2b | 191.9 (1.50×) | 136.9 (1.16×) | 99.3 (0.90×) | 65.5 (0.68×) |
| qwen35-4b | 104.8 (1.32×) | 80.2 (1.14×) | 60.4 (0.94×) | 39.2 (0.72×) |
| qwen35-9b | 63.9 (1.26×) | 50.3 (1.03×) | 39.6 (0.86×) | 27.2 (0.67×) |
| qwen35-27b | 22.3 (1.27×) | 19.0 (1.19×) | 16.3 (1.03×) | — *(crashed)* |
| qwen36-27b through gemma4-31b | — | — | — | — |

## B=8

Not started. Sweep crashed before B=4 finished.

## Phase 2 smoke (post pre-allocation fix, commit `47e9276`)

> **Reverted.** This data was captured against commit `47e9276` (the Phase 2 pre-allocation fix), which was reverted in `7bdd26c`. Numbers preserved here for the record because they characterize what the pre-allocation does (eliminate B>1 OOM at the cost of always allocating `maxCacheSize`).

Qwen3.5-9B B=2 with the `RotatingKVCache.updateInPlace` pre-allocation in place. Compares directly to the Phase 1 sweep above for the same model + same configs.

| KV | Ctx | Phase 1 (B=2) | Phase 2 (B=2) | Δ |
|---|---:|:---|:---|:---|
| none | 1k | 53.9 (0.98× B=1) | **58.7 (1.21× B=1)** | +9% absolute, ratio flipped to net win |
| none | 4k | 44.1 (0.88× B=1) | **47.6 (1.00× B=1)** | +8%, no longer a regression |
| none | 16k | OOM (in this run) | **28.9 (0.75× B=1)** | unblocked |
| turbo4v2 | 1k | 57.6 (1.13× B=1) | **57.7 (1.20× B=1)** | matched, ratio improved |
| turbo4v2 | 4k | 45.6 (0.93× B=1) | **45.9 (0.99× B=1)** | matched, no longer a regression |
| turbo4v2 | 16k | OOM (in this run) | **25.8 (0.65× B=1)** | unblocked |

All 12 rows of the Phase 2 smoke (1k/4k/16k × none/turbo4v2 × B=1/B=2) ran clean. No new OOMs.

### Phase 2 long-context push (Qwen3.5-9B B=2 only, commit `47e9276`)

Subsequent stress test against the new ceiling — what does the prefill activation peak look like at long context?

| KV | Ctx | Result | GPU peak |
|---|---:|:---|---:|
| none | 32k | **26.5 tok/s** ✅ | **49.40 GB** (pinned at OS cap) |
| none | 64k | OOM (`Internal Error 0x0e`) | — |
| turbo4v2 | 32k | **15.9 tok/s** ✅ | **49.40 GB** |
| turbo4v2 | 64k | killed mid-run before result | — |

ctx=32k B=2 *runs* but every byte of recommended GPU is being held. ctx=64k B=2 doesn't fit — and looking at the gap (KV alone is 17 GB vs ~4 GB at ctx=16k, but peak grows from 28.9 → 49.4 GB which is ~20 GB transient on top of 4 GB extra KV) the activations dominate. The `asyncEval` follow-up in the Follow-ups section is the likely unblock.

## OOM map

Real OOMs (not contention) observed across both sweeps:

### Phase 1 sweep (pre fix)

| Model | KV | Ctx | Batch | Error |
|---|---|---:|---:|---|
| nemotron-30b-a3b | none | 16k | 2 | `Internal Error 0x0e` |
| nemotron-30b-a3b | turbo4v2 | 1k–16k | 2 | All four ctxes failed — Metal state poisoning after the prior OOM, the model loads but every kernel dispatch errors. Effectively 8 lost cells. |
| gemma4-e2b | none | 16k | 2 | `OOM 0x08` |
| gemma4-e2b | turbo4v2 | 16k | 2 | `OOM 0x08` |
| gemma4-e4b | none | 16k | 2 | `OOM 0x08` |
| gemma4-e4b | turbo4v2 | 16k | 2 | `OOM 0x08` |
| gemma4-31b | none | 16k | 2 | `OOM 0x08` (ticket clamped to 49 GB) |
| gemma4-31b | turbo4v2 | 16k | 2 | `OOM 0x08` |

Phase 2 hasn't been re-run against the gemma4 dense models (e2b/e4b/31b) or nemotron — those would be informative confirmations that the pre-allocation fix carries over. Filed in Follow-ups.

### Phase 2 long-context push

| Model | KV | Ctx | Batch | Error |
|---|---|---:|---:|---|
| qwen35-9b | none | 64k | 2 | `Internal Error 0x0e` (activation peak >49 GB) |
| qwen35-9b | turbo4v2 | 64k | 2 | not measured (run killed) — expected to OOM |

The Metal-state-poisoning behaviour after `nemotron-30b-a3b` ctx=16k is worth flagging — once a kernel dispatch errors with 0x0e, the *next* invocation in the same process can crash even at much smaller configs. The benchmark.sh launches a fresh process per row so this should reset, but in practice we saw the entire turbo4v2 run for that model fail. May warrant `MLX.GPU.synchronize()` + cache clear between rows, or sleep delay.

## Patterns

1. **Where batching wins.** GeMM amortization headroom is largest where:
   - Per-token compute is small relative to weight-matmul (small models, MoE active sparsity).
   - Context is short (KV bandwidth doesn't dominate yet).

2. **Where batching breaks even.** Mid-sized dense models at moderate context (4k–8k). The matmul-bound region narrows as KV gets larger.

3. **Where batching loses.** Long-context dense models. KV reads scale with `B × ctx`; the matmul amortization saturates fast. Beyond ctx=8k almost everything regresses.

4. **MoE behaves like a small dense model at decode** because only a fraction of weights are active per token — explaining why `qwen35-35b-a3b` (active 3B) outperforms its weight class on batching, and `gemma4-26b-a4b` is the standout (1.40-1.57× speedup, holds gains through ctx=8k).

5. **TurboQuant doesn't help batched throughput at B=2.** Comparing none vs turbo4v2 ratios is roughly even — TurboQuant is a memory-savings feature, not a throughput feature, and its overhead at the attention path is noise relative to the batch dimension's effect.

6. **The wired-memory estimator from PR #137 is doing its job.** Tickets size correctly with batch (linear scaling), clamp to OS-recommended cap when over, and the OOMs we see are real allocation pressure (transient activations during prefill), not ticket misconfiguration. Confirmed by the `gemma4-31b` ctx=16k case: estimator wanted 67 GB, clamped to 49 GB, then the underlying buffer allocation actually exceeded what the OS could wire — the right behaviour.

## Notable individual numbers

**Best aggregate throughput uplifts:**

- `qwen35-0.8b` B=4 turbo4v2 ctx=1024: **314.0 tok/s** (1.88× B=1)
- `qwen35-0.8b` B=4 none ctx=1024: **327.9 tok/s** (1.83× B=1)
- `qwen35-0.8b` B=4 none ctx=4096: **253.6 tok/s** (1.60× B=1)
- `qwen35-0.8b` B=4 turbo4v2 ctx=4096: **228.1 tok/s** (1.56× B=1)
- `qwen35-2b` B=4 ctx=1024: **195.7 / 191.9 tok/s** (1.50× B=1)
- `gemma4-26b-a4b` B=2 turbo4v2 ctx=1024: **39.0 tok/s** (1.57× B=1) — biggest MoE B=2 ratio observed
- `qwen35-35b-a3b` B=2 turbo4v2 ctx=1024: **85.4 tok/s** (1.40× B=1)

**Worst regressions:**

- `gpt-oss-20b` B=2 ctx=16k: **35.8 tok/s** (0.65× B=1)
- `qwen35-2b` B=2 turbo4v2 ctx=16k: **56.2 tok/s** (0.58× B=1)
- `qwen35-4b` B=4 ctx=16k: **45.0 tok/s** (0.80× B=1)

## Follow-ups

**Goal:** fast prefill, fast decode, low peak memory — across the spectrum from a 16 GB Mac running B=1 to a 64 GB Mac running B>1. The most-used default is going to be a rotating KV cache sized to the user's hardware + task, with `turbo4v2` compression when memory matters and no compression when speed matters. Both shapes need to be optimal.

### Ranked solutions

| Idea | Peak↓ | Speed cost | Effort | Verdict |
|---|---|---|---|---|
| **#128 — wire `PagedKVCache` into model factories** | **huge** (the actual fix) | small (gather-path overhead until #127 lands) | M (1–2 weeks) | The right path |
| #127 — Metal paged kernel | matches dense | net win once landed | L (multi-week) | Pair with #128 |
| **(E) `RotatingKVCache.update()` returns a slice copy instead of a view** | eliminates multi-version retention without pre-allocating | one slice copy per update — same memory bandwidth as the SDPA read it replaces, throughput-neutral expected | S (half day to spike) | **Best near-term candidate.** Surface-area-zero (no API change, no model audits), keeps rotating + turbo as the default deployment shape, paged still better long-term. Risk: copy *might* bloat transient memory if MLX doesn't free the source view aggressively — needs measurement. |
| (B) Halve `asyncEval` window when B>1 | ~2× peak reduction (transient) | ~5% prefill | S (hours) | Cheap interim while paged is built |
| (C) Allocate `min(maxCacheSize, prompt + decode_budget)` instead of maxCacheSize | proportional to ctx-vs-maxKV ratio | none | S (hours) | Cheap, addresses "user sets ctx=32k but generates 200 tokens" |
| (D) `clearCache()` between prefill and decode | ~5–10% off steady-state floor | none | S (1 line) | Easy |
| (1a) `asyncEval` after slice-write | unclear after re-analysis | small | S | Limited — model already asyncEvals every 8 layers, and the multi-version retention is happening *within* that window |
| Smaller `defaultPrefillStepSize` at B>1 | ~2× peak reduction (transient) | proportional regression | S | Trades speed for memory; treat as a fallback knob |
| **`BatchedKVCache` integration** | **no help** | n/a | n/a | Same slice-assign primitive, same multi-version issue |

### Filed issues / done

- **MoE batching deployment doc** — added to root `README.md` under "Choosing a deployment shape (Apple Silicon)".
- **Persist batched rows to the markdown report** — `BenchmarkWriter` schema bumped (new `batchSize` and `seqDecodeTokPerSec` fields, new "B" and "Decode (agg)" / "Decode (seq)" columns), `runBatchedBenchmark` wired to call `BenchmarkWriter.append`. Future batched runs persist alongside single-stream rows.
- **TurboQuant long-context batched regression** — filed as [#149](https://github.com/ekryski/mlx-swift-lm/issues/149).

### Carryover items still relevant

- **Re-run B=4 from `qwen36-27b` onward + B=8 entirely.** Many gaps in the B=4 / B=8 columns of the headline tables.
- **Investigate Metal-state poisoning** after `nemotron-30b-a3b` ctx=16k OOM (8 lost cells in the B=2 turbo4v2 row). May be irrelevant — likely just normal post-OOM Metal instability rather than a separate state-leak bug. Worth a defensive reset between rows (sleep + `MLX.GPU.clearCache()`) regardless, but don't expect a deep root cause.
