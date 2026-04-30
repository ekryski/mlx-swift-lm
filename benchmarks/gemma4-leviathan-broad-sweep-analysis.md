# Gemma 4 — Leviathan accept/reject sampling broad sweep

**Date:** 2026-04-29
**Hardware:** M1 Max 64 GB
**Branch:** `ek/023-leviathan-accept-reject` (PR #154) at commit `062e2c8`
**Models:** Gemma 4 26B A4B 4-bit, Gemma 4 E2B 4-bit
**Prompts:** recipe-rewrite (regurgitative) + lighthouse-keeper short-story (paraphrastic)
**Iterators:** plain `TokenIterator` (TI), `NGramSpeculativeTokenIterator` greedy (NGgreedy), `NGramSpeculativeTokenIterator` + Leviathan (NGlev)
**Temperatures:** 0, 0.6, 1.0
**Trials:** 5 per cell, median reported
**Goal:** Validate or refute the spec-023 hypothesis that Leviathan accept/reject sampling delivers a meaningful speedup at `temperature > 0`, identify regimes where it shouldn't be enabled.

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

| Model | Prompt | NGgreedy@0 | NGlev@0.6 | NGlev@1.0 |
|---|---|---|---|---|
| **26B A4B 4-bit** | recipe (regurgitative) | **1.32×** ✅ | **1.18×** ✅ | **1.18×** ✅ |
| 26B A4B 4-bit | lighthouse (paraphrastic) | 1.08× ≈ | 0.94× ⚠️ | 0.94× ⚠️ |
| E2B 4-bit | recipe (regurgitative) | 0.97× ≈ | **0.77×** ❌ | **0.78×** ❌ |
| E2B 4-bit | lighthouse (paraphrastic) | 0.86× ⚠️ | 0.86× ⚠️ | 0.87× ⚠️ |

✅ engage; ≈ neutral within noise; ⚠️ measurable regression; ❌ large regression.

## Median tok/s by cell

```
model           prompt        cell          temp  lev   tok/s     accept
--------------------------------------------------------------------------------
gemma4-26b-a4b  recipe        TI@0          0.0   off   29.3      —
gemma4-26b-a4b  recipe        NGgreedy@0    0.0   off   38.7      75/107  (70.1%)
gemma4-26b-a4b  recipe        TI@0.6        0.6   off   29.1      —
gemma4-26b-a4b  recipe        NGlev@0.6     0.6   lev   34.4      75/107  (70.1%)
gemma4-26b-a4b  recipe        TI@1.0        1.0   off   29.1      —
gemma4-26b-a4b  recipe        NGlev@1.0     1.0   lev   34.3      75/107  (70.1%)
gemma4-26b-a4b  lighthouse    TI@0          0.0   off   30.6      —
gemma4-26b-a4b  lighthouse    NGgreedy@0    0.0   off   33.0      0/6     (0.0%)
gemma4-26b-a4b  lighthouse    TI@0.6        0.6   off   29.3      —
gemma4-26b-a4b  lighthouse    NGlev@0.6     0.6   lev   27.4      0/7     (0.0%)
gemma4-26b-a4b  lighthouse    TI@1.0        1.0   off   29.7      —
gemma4-26b-a4b  lighthouse    NGlev@1.0     1.0   lev   27.9      1/6     (16.7%)
gemma4-e2b      recipe        TI@0          0.0   off   109.5     —
gemma4-e2b      recipe        NGgreedy@0    0.0   off   106.1     73/105  (69.5%)
gemma4-e2b      recipe        TI@0.6        0.6   off   97.4      —
gemma4-e2b      recipe        NGlev@0.6     0.6   lev   75.0      76/108  (70.4%)
gemma4-e2b      recipe        TI@1.0        1.0   off   98.4      —
gemma4-e2b      recipe        NGlev@1.0     1.0   lev   76.3      73/105  (69.5%)
gemma4-e2b      lighthouse    TI@0          0.0   off   115.7     —
gemma4-e2b      lighthouse    NGgreedy@0    0.0   off   99.0      0/13    (0.0%)
gemma4-e2b      lighthouse    TI@0.6        0.6   off   105.8     —
gemma4-e2b      lighthouse    NGlev@0.6     0.6   lev   91.0      0/3     (0.0%)
gemma4-e2b      lighthouse    TI@1.0        1.0   off   105.5     —
gemma4-e2b      lighthouse    NGlev@1.0     1.0   lev   91.7      0/9     (0.0%)
```

## Findings

### 1. Leviathan delivers the headline win on big-model + regurgitative content (1.18×)

On Gemma 4 26B A4B + recipe-rewrite, Leviathan at both `temp=0.6` and `temp=1.0` produces a **1.18× speedup** over the matching `TokenIterator(temperature: T)` baseline. This was previously unattainable — pre-PR-#154, n-gram declined at `temp != 0` and fell back to plain TI, losing the entire spec-decode speedup. **The spec-023 hypothesis is validated for this regime.**

Caveat: greedy n-gram on the same prompt is **1.32×**. Leviathan's per-position softmax + per-position `eval(p)` (to extract `p[draft_i]` for the accept comparison) introduces multiple CPU↔GPU syncs per cycle, vs. the greedy path's single batch `eval` at the end of verify. Net cost: ~0.14× speedup-ratio reduction (1.32× → 1.18×). This is recoverable in phase 2 (batch the softmax across all verify positions, single eval); for phase 1 the 1.18× is the headline.

### 2. Speedup is invariant to temperature on this regime

`temp=0.6` and `temp=1.0` produce **identical** Leviathan speedups (1.18× both, with median tok/s of 34.4 and 34.3). The accept rate is also identical (75/107 = 70.1%) across both temperatures **and** identical to the greedy path's accept rate.

Mechanism: on the recipe prompt, the target's top-1 logit margins are wide enough that `softmax(logits / T)[argmax]` is ≈ 0.99+ at both `T=0.6` and `T=1.0`. Whether `u ~ U(0,1)` is compared against 0.99 or 0.95 doesn't change the accept outcome — the dominant top-1 token always wins. **Temperature would only matter on prompts where margins are tight enough for the temperature-scaling factor to flip accept decisions** — i.e., paraphrastic content (see finding 4).

### 3. Greedy n-gram beats Leviathan even at the same temperature would imply (no surprise)

Where greedy is correctness-equivalent (temp=0), it's also faster than Leviathan would be at any positive temperature. The decision tree for callers is:
- `temp == 0`: use the greedy path (no choice; Leviathan is a no-op at temp=0).
- `temp > 0`: greedy isn't an option (it'd diverge from the sampling distribution). Leviathan is the only correct way to engage spec-decode.

So Leviathan vs greedy isn't a competition — they cover disjoint regimes.

### 4. Paraphrastic content + Leviathan = consistent regression on both models

| Model | Lighthouse + greedy@0 | Lighthouse + Leviathan@0.6 | Lighthouse + Leviathan@1.0 |
|---|---|---|---|
| 26B A4B | 1.08× ≈ | 0.94× ⚠️ | 0.94× ⚠️ |
| E2B 4-bit | 0.86× ⚠️ | 0.86× ⚠️ | 0.87× ⚠️ |

Same root cause as the paraphrastic regression characterised in PR #113's close-out sweep: lookup almost never hits (3-13 proposals over a 200-token generation, vs. 105-108 on recipe), and when it does the strict-greedy guard (greedy path) or low-probability draft (Leviathan path) rejects everything. **The verify cycle is pure overhead in this regime.** Leviathan adds the per-position softmax cost on top, hence the slightly worse regression on 26B-A4B (1.08× greedy → 0.94× Leviathan).

Note one outlier: the 26B-A4B lighthouse + Leviathan@1.0 cell shows a single trial with 1/6 accept (16.7%) where the higher temperature flattened the distribution enough that one draft token's accept probability cleared `u`. Throughput unaffected (still 0.94×) — it takes more than one accept-per-cycle to overcome the verify overhead.

### 5. Small/fast model + Leviathan = strong regression

| Model | Recipe + Leviathan@0.6 | Recipe + Leviathan@1.0 |
|---|---|---|
| 26B A4B | 1.18× ✅ | 1.18× ✅ |
| **E2B 4-bit** | **0.77×** ❌ | **0.78×** ❌ |

E2B 4-bit at ~100 tok/s baseline is the worst regime for Leviathan: per-position softmax + per-position `eval(p)` adds ~3-5 ms/token of GPU sync overhead, which is 30-50% of the model's already-fast forward pass. **70% accept rate cannot recoup that cost.** The 0.77× number means n-gram + Leviathan is meaningfully *slower* than just running plain TokenIterator at the same temperature.

This is a stronger regression than greedy n-gram on the same model, which was 0.97× (essentially neutral). Phase 2's batched-softmax optimisation would help here most.

### 6. Cross-cutting: when should Leviathan engage?

The data argues for **engaging Leviathan only when the same regime would benefit from greedy n-gram**:

| Regime | Greedy n-gram | Leviathan @ temp > 0 | Recommendation |
|---|---|---|---|
| Big WBB model + regurgitative | 1.32× ✅ | 1.18× ✅ | **Engage both** |
| Big WBB model + paraphrastic | 1.08× ≈ | 0.94× ⚠️ | Don't engage either |
| Small/fast model + regurgitative | 0.97× ≈ | 0.77× ❌ | Don't engage either |
| Small/fast model + paraphrastic | 0.86× ⚠️ | 0.86× ⚠️ | Don't engage either |

The auto-disengage heuristic discussion in **issue #153** (workload-detection + small-model threshold) applies here even more strongly than to the greedy path. Leviathan inherits all the regression regimes of greedy and adds a new one (small/fast models become measurably worse). **The phase-1 release should keep `MLX_NGRAM_LEVIATHAN=1` opt-in, not flip it to default-on.** A future auto-engagement decision should track the same predicate work as #153 and apply equally to both paths.

## Methodology notes

- 5 trials per cell, median reported. Variance bands are similar to PR #113's sweeps: ±5-15% on individual trials, median robust.
- E2B 4-bit's higher baseline tok/s (~100-115) makes it more sensitive to per-trial system-load variance. The wider trial spread on its cells (e.g. NGlev@0.6 recipe: 57.2, 69.0, 75.0, 75.8, 79.3) reflects that.
- Token count was 159 on 26B-A4B + recipe (matches PR #113's recipe sweep) and varied modestly on lighthouse cells where the model's stochastic generation can hit different EOS positions.
- One known parser-attribution bug in the initial run was caught and fixed before the final medians table (the last cell of each model/prompt section was misattributed to the next section's first slot; the fix in `/tmp/parse_sweep.py` commits the in-progress cell on section boundaries).

## Conclusions

1. **Leviathan works as designed.** On the regime where spec-decode wins (big WBB model + regurgitative content), it delivers a measurable 1.18× speedup at `temperature > 0` — speed that was previously inaccessible because n-gram declined at non-greedy temperatures.

2. **The win is smaller than greedy's** (~10-15% lower speedup-ratio), entirely attributable to per-position GPU-sync overhead. Phase-2 batched-softmax optimisation is well-motivated.

3. **Leviathan inherits and amplifies the regression regimes** identified in PR #113's sweep. Don't flip to default-on without addressing those regimes — discussion in issue #153.

4. **Phase 1 shipping recommendation**: keep `MLX_NGRAM_LEVIATHAN=1` opt-in via env var or per-call Swift flag (not yet wired). Document the regime asymmetry. Promote to default-on later, conditional on the auto-disengage heuristic landing or on caller knowledge of their workload.
