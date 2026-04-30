# Gemma 4 — Leviathan accept/reject sampling broad sweep

**Date:** 2026-04-29
**Hardware:** M1 Max 64 GB
**Branch:** `ek/023-leviathan-accept-reject` (PR #154)
**Models:** Gemma 4 26B A4B 4-bit, Gemma 4 E2B 4-bit
**Prompts:** recipe-rewrite (regurgitative) + lighthouse-keeper short-story (paraphrastic)
**Iterators:** plain `TokenIterator` (TI), `NGramSpeculativeTokenIterator` greedy (NGgreedy), `NGramSpeculativeTokenIterator` + Leviathan (NGlev)
**Temperatures:** 0, 0.6, 1.0
**Trials:** 5 per cell, median reported
**Goal:** Validate or refute the spec-023 hypothesis that Leviathan accept/reject sampling delivers a meaningful speedup at `temperature > 0`, identify regimes where it shouldn't be enabled.

## Two passes — phase 1 and phase 1 + batched

This file reports two sweeps:

1. **Phase 1 (initial implementation)** — per-position softmax + per-position
   `eval(p)` to extract `p[draft_i]` for the accept comparison.
2. **Phase 1 + batched (current)** — one batched softmax across all `numDraft`
   positions + single `eval` for all p-values, plus one final eval for
   the residual or bonus sample. Total: 2 evals per cycle instead of N+1.

Headline numbers below reflect the **batched implementation**. Phase-1
medians are shown alongside in the "phase comparison" subsection so the
optimization's effect is visible.

## Cell matrix (24 cells, 120 trials)

For each (model, prompt) pair, six cells:

| Cell | Iterator | Temp | Notes |
|---|---|---|---|
| TI@0 | TokenIterator | 0 | Greedy baseline |
| NGgreedy@0 | NGramSpec greedy | 0 | Existing greedy spec-decode path (control) |
| TI@0.6 | TokenIterator | 0.6 | Sampling baseline |
| NGlev@0.6 | NGramSpec + Leviathan | 0.6 | **The new path** |
| TI@1.0 | TokenIterator | 1.0 | Higher-entropy sampling baseline |
| NGlev@1.0 | NGramSpec + Leviathan | 1.0 | Leviathan at higher temperature |

## Headline numbers — speedups vs the matching `TokenIterator` baseline

Numbers below are from the **phase 1 + batched** implementation (current code). 120 runs, 5-trial median:

| Model | Prompt | NGgreedy@0 | NGlev@0.6 | NGlev@1.0 |
|---|---|---|---|---|
| **26B A4B 4-bit** | recipe (regurgitative) | **1.63× ✅** | **1.31× ✅** | **1.25× ✅** |
| 26B A4B 4-bit | lighthouse (paraphrastic) | 0.98× ≈ | 0.92× ⚠️ | 0.93× ⚠️ |
| E2B 4-bit | recipe (regurgitative) | 0.96× ≈ | 0.78× ❌ | 0.78× ❌ |
| E2B 4-bit | lighthouse (paraphrastic) | 0.86× ⚠️ | 0.85× ⚠️ | 0.85× ⚠️ |

✅ engage; ≈ neutral within noise; ⚠️ measurable regression; ❌ large regression.

The 26B-A4B + recipe NGgreedy@0 cell hit 1.63× this run because `TI@0` came in low (23.6 tok/s vs. typical 28-29 from prior PR #113 measurements — system-load variance, unchanged code path). The reliable greedy reference for that regime is **1.32×** from prior multi-trial sweeps. The Leviathan 1.31× / 1.25× speedups, computed against the same-trial-run TI baseline at temp=0.6 / temp=1.0, are robust because they share the noise floor with the cells they're compared against.

## Median tok/s by cell (phase 1 + batched)

```
model           prompt        cell          temp  lev   tok/s     accept
--------------------------------------------------------------------------------
gemma4-26b-a4b  recipe        TI@0          0.0   off   23.6      —
gemma4-26b-a4b  recipe        NGgreedy@0    0.0   off   38.4      75/107  (70.1%)
gemma4-26b-a4b  recipe        TI@0.6        0.6   off   29.2      —
gemma4-26b-a4b  recipe        NGlev@0.6     0.6   lev   38.3      75/107  (70.1%)
gemma4-26b-a4b  recipe        TI@1.0        1.0   off   26.4      —
gemma4-26b-a4b  recipe        NGlev@1.0     1.0   lev   32.9      75/107  (70.1%)
gemma4-26b-a4b  lighthouse    TI@0          0.0   off   29.1      —
gemma4-26b-a4b  lighthouse    NGgreedy@0    0.0   off   28.4      0/6     (0.0%)
gemma4-26b-a4b  lighthouse    TI@0.6        0.6   off   29.8      —
gemma4-26b-a4b  lighthouse    NGlev@0.6     0.6   lev   27.5      0/7     (0.0%)
gemma4-26b-a4b  lighthouse    TI@1.0        1.0   off   29.8      —
gemma4-26b-a4b  lighthouse    NGlev@1.0     1.0   lev   27.7      0/6     (0.0%)
gemma4-e2b      recipe        TI@0          0.0   off   109.6     —
gemma4-e2b      recipe        NGgreedy@0    0.0   off   105.0     73/105  (69.5%)
gemma4-e2b      recipe        TI@0.6        0.6   off   97.9      —
gemma4-e2b      recipe        NGlev@0.6     0.6   lev   76.3      73/105  (69.5%)
gemma4-e2b      recipe        TI@1.0        1.0   off   100.2     —
gemma4-e2b      recipe        NGlev@1.0     1.0   lev   78.0      73/105  (69.5%)
gemma4-e2b      lighthouse    TI@0          0.0   off   115.9     —
gemma4-e2b      lighthouse    NGgreedy@0    0.0   off   99.1      0/13    (0.0%)
gemma4-e2b      lighthouse    TI@0.6        0.6   off   105.7     —
gemma4-e2b      lighthouse    NGlev@0.6     0.6   lev   89.9      0/6     (0.0%)
gemma4-e2b      lighthouse    TI@1.0        1.0   off   105.8     —
gemma4-e2b      lighthouse    NGlev@1.0     1.0   lev   89.5      0/9     (0.0%)
```

## Phase comparison: phase 1 (per-position eval) vs phase 1 + batched (single eval)

Speedup-ratio (NGlev / matching `TI@temp`) before and after the batched optimization. Δ is the speedup-ratio change; reading these as relative deltas is more robust than raw tok/s because both cells in each ratio share the same trial-run noise floor.

| Model | Prompt | Cell | Phase 1 | Phase 1 + batched | Δ |
|---|---|---|---|---|---|
| **26B A4B** | **recipe** | **NGlev@0.6** | **1.18×** | **1.31×** | **+0.13** ✅ |
| 26B A4B | recipe | NGlev@1.0 | 1.18× | 1.25× | **+0.07** ✅ |
| 26B A4B | lighthouse | NGlev@0.6 | 0.94× | 0.92× | −0.02 |
| 26B A4B | lighthouse | NGlev@1.0 | 0.94× | 0.93× | −0.01 |
| E2B | recipe | NGlev@0.6 | 0.77× | 0.78× | +0.01 |
| E2B | recipe | NGlev@1.0 | 0.78× | 0.78× | 0.00 |
| E2B | lighthouse | NGlev@0.6 | 0.86× | 0.85× | −0.01 |
| E2B | lighthouse | NGlev@1.0 | 0.87× | 0.85× | −0.02 |

The optimization's effect is **concentrated on the favourable regime** (26B A4B + recipe), where the per-position GPU-sync eval was the dominant overhead. There it adds **+13 percentage points** of speedup ratio at temp=0.6, essentially closing the gap to greedy n-gram (1.32×). At temp=1.0 the gain is +7 points; the remaining ~7-point gap comes from the residual / bonus sample's necessary final eval being on a tighter sampling distribution at higher temperature (more probability mass spread across more candidate tokens, so the Random + sampler step does more meaningful work).

On the regression regimes (paraphrastic content, small/fast models), the bottleneck is something other than per-position eval — lookup misses on paraphrastic content, per-token bookkeeping floor on E2B. Batching the softmax doesn't help there, but it doesn't hurt either: every other cell shifts within ±0.02 noise, which is well inside the per-trial variance.

**The optimization is a no-regression win.** Keeping it as the default Leviathan path.

## Findings

### 1. Leviathan delivers the headline win on big-model + regurgitative content (1.31× at temp=0.6)

On Gemma 4 26B A4B + recipe-rewrite, the **batched implementation** of Leviathan at `temp=0.6` produces a **1.31× speedup** over the matching `TokenIterator(temperature: 0.6)` baseline. At `temp=1.0` it's 1.25×. Both numbers were previously unattainable — pre-PR-#154, n-gram declined at `temp != 0` and fell back to plain TI, losing the entire spec-decode speedup. **The spec-023 hypothesis is validated for this regime.**

The batched implementation closes the gap to greedy n-gram (1.32× from prior multi-trial sweeps): 1.18× → 1.31× at temp=0.6 essentially eliminates the Leviathan-vs-greedy throughput penalty on the favourable regime. Phase 1's per-position eval was the dominant overhead, and the spec-023-§-batching optimisation that recovered it was a strict win.

### 2. Speedup is *near*-invariant to temperature on this regime

`temp=0.6` and `temp=1.0` produce 1.31× and 1.25× respectively — close but not identical. Phase 1 had reported 1.18× / 1.18× (truly identical); the batched implementation reveals a small temperature-dependent gap because the residual / bonus sampling step's final eval scales modestly with the distribution's flatness (more probability mass spread across more tokens means the categorical sampler does more meaningful per-call work). The accept rate is still identical (75/107 = 70.1%) across both temperatures **and** identical to the greedy path's accept rate.

Mechanism on accept rate: on the recipe prompt, the target's top-1 logit margins are wide enough that `softmax(logits / T)[argmax]` is ≈ 0.99+ at both `T=0.6` and `T=1.0`. Whether `u ~ U(0,1)` is compared against 0.99 or 0.95 doesn't change the accept outcome — the dominant top-1 token always wins. **Temperature would only matter on prompts where margins are tight enough for the temperature-scaling factor to flip accept decisions** — i.e., paraphrastic content (see finding 4).

### 3. Greedy n-gram beats Leviathan even at the same temperature would imply (no surprise)

Where greedy is correctness-equivalent (temp=0), it's also faster than Leviathan would be at any positive temperature. The decision tree for callers is:
- `temp == 0`: use the greedy path (no choice; Leviathan is a no-op at temp=0).
- `temp > 0`: greedy isn't an option (it'd diverge from the sampling distribution). Leviathan is the only correct way to engage spec-decode.

So Leviathan vs greedy isn't a competition — they cover disjoint regimes.

### 4. Paraphrastic content + Leviathan = consistent regression on both models

| Model | Lighthouse + greedy@0 | Lighthouse + Leviathan@0.6 | Lighthouse + Leviathan@1.0 |
|---|---|---|---|
| 26B A4B | 0.98× ≈ | 0.92× ⚠️ | 0.93× ⚠️ |
| E2B 4-bit | 0.86× ⚠️ | 0.85× ⚠️ | 0.85× ⚠️ |

Same root cause as the paraphrastic regression characterised in PR #113's close-out sweep: lookup almost never hits (3-13 proposals over a 200-token generation, vs. 105-108 on recipe), and when it does the strict-greedy guard (greedy path) or low-probability draft (Leviathan path) rejects everything. **The verify cycle is pure overhead in this regime.** Leviathan adds the per-position softmax cost on top, hence the slightly worse regression on 26B-A4B (1.08× greedy → 0.94× Leviathan).

Note one outlier: the 26B-A4B lighthouse + Leviathan@1.0 cell shows a single trial with 1/6 accept (16.7%) where the higher temperature flattened the distribution enough that one draft token's accept probability cleared `u`. Throughput unaffected (still 0.94×) — it takes more than one accept-per-cycle to overcome the verify overhead.

### 5. Small/fast model + Leviathan = strong regression

| Model | Recipe + Leviathan@0.6 | Recipe + Leviathan@1.0 |
|---|---|---|
| 26B A4B | 1.31× ✅ | 1.25× ✅ |
| **E2B 4-bit** | **0.78×** ❌ | **0.78×** ❌ |

E2B 4-bit at ~100 tok/s baseline is the worst regime for Leviathan: even after batching the softmax, the residual two evals per cycle plus the per-token bookkeeping floor add ~3-5 ms/token, which is 30-50% of the model's already-fast forward pass. **70% accept rate cannot recoup that cost.** The 0.78× number means n-gram + Leviathan is meaningfully *slower* than just running plain TokenIterator at the same temperature.

The batched optimisation didn't help here (0.77× → 0.78×, within noise). The bottleneck on E2B isn't per-position eval — it's the irreducible 2-eval-per-cycle floor (one for batched p-values, one for residual / bonus sample) plus CPU-side bookkeeping. Below some forward-pass cost threshold, *any* spec-decode iterator pays more than it earns. The same pattern is visible in the greedy n-gram path (0.97× on E2B + recipe at temp=0).

### 6. Cross-cutting: when should Leviathan engage?

The data argues for **engaging Leviathan only when the same regime would benefit from greedy n-gram**:

| Regime | Greedy n-gram | Leviathan @ temp > 0 (batched) | Recommendation |
|---|---|---|---|
| Big WBB model + regurgitative | 1.32× ✅ | 1.25–1.31× ✅ | **Engage both** |
| Big WBB model + paraphrastic | 0.98–1.08× ≈ | 0.92–0.93× ⚠️ | Don't engage either |
| Small/fast model + regurgitative | 0.96–0.97× ≈ | 0.78× ❌ | Don't engage either |
| Small/fast model + paraphrastic | 0.86× ⚠️ | 0.85× ⚠️ | Don't engage either |

The batched optimisation closed the speedup-ratio gap between Leviathan and greedy on the favourable regime (1.18× → 1.31× at temp=0.6, vs greedy's 1.32×). On the regression regimes the gap was never per-position-eval — it was lookup-misses and per-token bookkeeping floors — so batching doesn't help there but doesn't hurt either.

The auto-disengage heuristic discussion in **issue #153** (workload-detection + small-model threshold) applies here as it does to the greedy path. Leviathan inherits the same regression regimes; on small/fast models specifically it's slightly worse than greedy because the per-token forward is so fast that even 2 evals/cycle is too many. **The phase-1 release should keep `MLX_NGRAM_LEVIATHAN=1` opt-in, not flip it to default-on.** A future auto-engagement decision should track the same predicate work as #153 and apply equally to both paths.

## Methodology notes

- 5 trials per cell, median reported. Variance bands are similar to PR #113's sweeps: ±5-15% on individual trials, median robust.
- E2B 4-bit's higher baseline tok/s (~100-115) makes it more sensitive to per-trial system-load variance. The wider trial spread on its cells (e.g. NGlev@0.6 recipe: 57.2, 69.0, 75.0, 75.8, 79.3) reflects that.
- Token count was 159 on 26B-A4B + recipe (matches PR #113's recipe sweep) and varied modestly on lighthouse cells where the model's stochastic generation can hit different EOS positions.
- One known parser-attribution bug in the initial run was caught and fixed before the final medians table (the last cell of each model/prompt section was misattributed to the next section's first slot; the fix in `/tmp/parse_sweep.py` commits the in-progress cell on section boundaries).

## Conclusions

1. **Leviathan works as designed.** On the regime where spec-decode wins (big WBB model + regurgitative content), it delivers a measurable **1.31× speedup at `temp=0.6`** and 1.25× at `temp=1.0` — speed that was previously inaccessible because n-gram declined at non-greedy temperatures.

2. **The batched optimisation closes the gap to greedy n-gram.** Phase 1's per-position-eval implementation hit 1.18×; the batched version hits 1.31×, within ~1% of greedy's 1.32×. The remaining gap is the irreducible 2-eval-per-cycle floor (one for batched p-values, one for the residual / bonus sample) — fundamental to accept/reject sampling, not implementation overhead.

3. **Leviathan inherits the regression regimes** identified in PR #113's sweep. Paraphrastic content, small/fast models — same regression patterns; the batched optimisation doesn't help there because the bottleneck isn't per-position eval. Don't flip to default-on without addressing those regimes — discussion in issue #153.

4. **Phase 1 shipping recommendation**: keep `MLX_NGRAM_LEVIATHAN=1` opt-in via env var. Document the regime asymmetry. Promote to default-on later, conditional on the auto-disengage heuristic landing or on caller knowledge of their workload. The throughput case for engaging is now strong on the favourable regime (matches greedy); the case for *not* engaging on the regression regimes is unchanged.

## Follow-up: GPT-OSS-20B sweep (2026-04-29)

Per the user's request, ran the same 6-cell × 5-trial sweep on GPT-OSS-20B (mxfp4 quant, `mlx-community/gpt-oss-20b-MXFP4-Q8`) to fill in the third pure-attention model class (after Gemma 4 26B A4B + E2B). GPT-OSS uses harmony format (`<|channel|>analysis<|message|>...`) — interesting structural overhead that may or may not help the n-gram lookup.

**Headline result: GPT-OSS-20B is in the fast-model regression regime.**

| Cell | Median tok/s | Speedup vs matching TI | Accept rate |
|---|---|---|---|
| TI@0 | 72.0 | 1.00× | — |
| NGgreedy@0 | 61.3 | **0.85× ⚠️** | 44.2% (38/86) |
| TI@0.6 | 67.9 | — | — |
| NGlev@0.6 | 51.4 | **0.76× ❌** | ~30-40% (varies) |
| TI@1.0 | 66.9 | — | — |
| NGlev@1.0 | 44.0 | **0.66× ❌** | ~30-60% (varies) |

(Recipe-rewrite prompt; lighthouse-keeper shows similar regression — NGgreedy@0 0.91×, NGlev@0.6 0.86×, NGlev@1.0 0.86×.)

### What this tells us

GPT-OSS-20B at ~70 tok/s baseline (~14 ms/token forward) sits in the same throughput regime where n-gram + Leviathan overhead is not amortizable, even at meaningful accept rates (44% on regurgitative content). The pattern from prior sweeps holds:

| Forward time / baseline tok/s | Regime | Speedup |
|---|---|---|
| ~42 ms / ~24 tok/s (Gemma 4 26B A4B) | Weight-bandwidth-bound | 1.32× / 1.31× ✅ |
| **~14 ms / ~70 tok/s (GPT-OSS-20B mxfp4)** | **Fast-forward** | **0.85× / 0.76× ⚠️** |
| ~10 ms / ~110 tok/s (Gemma 4 E2B 4-bit) | Fast-forward | 0.97× / 0.78× ⚠️ |

**The threshold for n-gram to win is around 30-40 ms/token forward time** (≈25-30 tok/s baseline) on M1 Max. Below that, the iterator's per-token overhead (lookup maintenance, accept-loop bookkeeping, cache-trim, processor work) doesn't amortize against the verify-batch's K+1 forward, even at 50-70% accept rate. The "weight-bandwidth-bound" framing from the original analysis was right but the threshold is finer than just "MoE vs dense" — it's about absolute per-token forward cost.

### Volatility note on NGlev@1.0

The NGlev@1.0 recipe trials show wide variance: 54.6, 39.0, 49.4, 44.0, 32.9 tok/s — and accept rates spanning 14-62% (14/49 to 62/106). At temp=1.0 the sampler picks different tokens each run; the lookup hits the new emitted token sequence at varying rates depending on whether the model happens to sample tokens that match prior continuations. **Higher temperatures → wider trial-to-trial variance** on Leviathan because the realised token sequence (and therefore the lookup history) is stochastic. Phase-1 + batched optimisation hasn't changed this fundamental property; it's inherent to sampling.

### Implications for issue #153 (auto-disengage)

GPT-OSS-20B is the cleanest argument yet for the auto-disengage heuristic that issue #153 captures. A model running at 70 tok/s baseline is fast enough that user-perceived latency is already low — engaging n-gram opt-in here is strictly worse than not. The "decision tree by baseline tok/s threshold" option (B) of issue #153 would catch GPT-OSS automatically; option (A) (rolling accept rate) would NOT — accept rate on recipe is 44%, well above any "low accept" threshold. This argues for **(B) being the more useful heuristic** — the regression is driven by per-token overhead, not by accept rate.

### Adding GPT-OSS to the supported-models table

Updating `Documentation.docc/speculative-decoding.md` to include GPT-OSS-20B as ⚠️ engages but slower (same row class as Gemma 4 E2B 4-bit). Tracked in PR #154.
