# 018 — Per-prompt sweep mode in the bench harness

- **Status:** spec, ready to issue (one-day task)
- **Branch:** small change off `main`

## Problem

Today the bench harness has two ngram-related modes:

1. `--method simple` with `--ngram N --ngram-max-draft D` → one cell, one prompt.
2. `--method ngram-sweep` → 32-cell matrix × 18-prompt suite, one model. Roughly 600 cells, hours of wall time, output is a giant markdown table that nobody reads.

What I actually wanted in the recipe-bulk debugging session was: **one prompt × N candidate configurations**. Pick a single prompt, run a baseline + 4-5 ngram cells, get a small, readable per-prompt table out the back. I scripted this by hand each time.

Fold this into the bench harness so it's reproducible and so the sweep harness produces the same shape of output.

## What we want

Two new method aliases:

### `--method ngram-spot`

Sweeps a small candidate-config matrix on a **single** prompt (the `simple` chat prompt by default, overridable via `MLX_BENCH_PROMPT`).

Default candidate cells (chosen from the sweep findings):

```
baseline (ngram=0)
n=3 D=2  (low-overhead spec)
n=3 D=4  (mid-range)
n=3 D=8  (long-amortising)
n=3 D=12 + adaptive + strict   (mixed-content default)
```

Override via `MLX_BENCH_NGRAM_SPOT_CELLS=n:D:flags,...` where `flags` is a comma-list of `adaptive`, `strict`, `dominance`, `multi`. Mirrors the existing `MLX_BENCH_NGRAM_SWEEP_CELLS` convention.

Output: one summary table at the end (similar to the prose I wrote in the chat session), plus the standard per-cell `[BENCH]` blocks.

### `--method ngram-sweep-summary`

Same 18-prompt × N-cell matrix as today, but with two changes:

1. After all cells run, emit a per-prompt "best cell" pick using a small scoring function (`tok/s` × `accuracy`, where accuracy means token-sequence equality to baseline at the same `MAX_TOKENS`).
2. Roll up to a per-category summary (`code-completion`, `qa-requote`, `recipe-bulk`, etc.) and a global default-config recommendation.

This makes the sweep useful for picking *which default config to ship* rather than as a 600-row diagnostic dump.

## Implementation

`runNgramSweep` in `Tests/Benchmarks/InferenceBenchmark.swift` already has the cell-iteration scaffolding. Add:

1. **`runNgramSpot(family:variant:repoId:kv:)`** — single-prompt variant. ~80 lines.
2. **In-memory results table** — replace the per-cell markdown writer for these methods with an array of `(cellLabel, prompt, baselineTokPerSec, cellTokPerSec, acceptRate, outputMatch: Bool)`.
3. **Accuracy check** — capture baseline output once per prompt, then `output == baselineOutput` for each ngram cell (token-prefix equality up to `min(len(baseline), len(cell))`).
4. **Summary printer** — at end, print:

```
[NGRAM-SPOT] qa-requote/01-bug-report.txt @ Gemma 4 26B A4B 4bit
| Cell                              | tok/s | Speedup | Accept | Match |
|-----------------------------------|------:|--------:|-------:|:-----:|
| baseline                          |  27.3 |  1.00×  |     —  |  ref  |
| n=3 D=2                           |  34.1 |  1.25×  |  60.0% |  ✓    |
| n=3 D=4                           |  31.0 |  1.14×  |  50.0% |  ✓    |
| n=3 D=8                           |  29.5 |  1.08×  |  35.0% |  ✓    |
| n=3 D=12 + adaptive + strict      |  33.8 |  1.24×  |  58.4% |  ✓    |
[NGRAM-SPOT] best: n=3 D=2 (+25%)
```

5. **CLI plumbing** — `--method ngram-spot` and `--method ngram-sweep-summary` in `scripts/benchmark.sh`. Each maps to an env var that the test harness reads.

## Files touched

| File | What |
|---|---|
| `Tests/Benchmarks/InferenceBenchmark.swift` | New methods + summary printer. |
| `scripts/benchmark.sh` | Argv plumbing + method validation. |
| `benchmarks/README.md` | Describe the two new methods + their output shape. |

## Open questions

1. **Where to read `MAX_TOKENS` for the spot mode.** Today the `simple` method honours `MLX_BENCH_MAX_TOKENS` (post spec 013). The spot mode should too — long-output sweeps are exactly when ngram amortisation kicks in.
2. **Default cell list.** The five above are my picks from the recipe-bulk + qa-requote sessions; they cover the regions where wins live (low-D, mid-D, adaptive). Keep these as defaults; expose env override.
3. **Should we deprecate the existing `simple --ngram` path?** No — it's still useful for one-shot debugging. Spot mode is the additive replacement for the bench-driver shell loops.

## Out of scope

- Aggregating across models in the summary roll-up (the sweep harness already iterates the comma-list of models in `--model`).
- KLD scoring on ngram cells. Acceptance + output-match is enough; KLD is for KV-quant evaluation, different concern.
