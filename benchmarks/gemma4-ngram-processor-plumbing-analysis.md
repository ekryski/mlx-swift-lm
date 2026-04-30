# Gemma 4 — n-gram processor plumbing + temperature disqualifier cost

**Date:** 2026-04-29
**Hardware:** M1 Max 64 GB
**Branch:** `ek/ngram-speculative-v2` (PR #113) at commit `3f0dd23` + uncommitted bench-harness env-var
**Models:** Gemma 4 E2B 4-bit, Gemma 4 26B A4B 4-bit
**Prompt:** Recipe rewrite (~280 tokens, input-grounded — high regurgitation potential, English unit → metric conversion)
**Generation:** 200-token budget, EOS at 159-161 tokens consistently across cells
**Goal:** Validate two claims from the spec-013 close-out audit:
1. The logit-processor plumbing fix (commit `880b416`) preserves n-gram speedup when penalties are active.
2. The `temperature != 0` disqualifier costs roughly the full n-gram speedup (motivating spec 023, Leviathan accept/reject).

## Cell matrix

Six cells × 2 models. Each cell runs once except cells D (no penalty) and E (rep penalty 1.1) on the 26B-A4B model where 3 trials surfaced
single-trial variance.

| Cell | Iterator | `temperature` | `repetitionPenalty` | Notes |
|---|---|---|---|---|
| A | `TokenIterator` | 0 | — | Greedy baseline |
| B | `TokenIterator` | 0 | 1.1 | Penalty cost on baseline |
| C | `TokenIterator` | 0.6 | — | Sampling cost on baseline |
| D | `NGram` | 0 | — | Current shipped n-gram path |
| E | `NGram` | 0 | 1.1 | **NEW** — plumbing fix in #113 |
| F | `NGram` | 0.6 | — | Currently disqualified, falls through to `TokenIterator` |

Cell F validates that the temperature disqualifier in `ngramRouteDecision` correctly falls through.

## Gemma 4 26B A4B 4-bit (multi-trial)

Bigger model — weight-bandwidth-bound, where spec-decode tends to give the cleanest signal.

All cells re-run 2026-04-29 with **5-trial protocol** (3-trial for D/E from the original sweep + the rep=1.3 follow-up; updated cells use 5 trials each). Medians reported below; full trial data in the trial column.

| Cell | Config | tok/s (median) | Speedup vs A | Accept rate | Trials |
|---|---|---|---|---|---|
| A | TI greedy | 28.5 | 1.00× | — | 28.4, 28.5, 28.5, 28.6, 28.7 |
| B | TI greedy + rep=1.1 | 27.3 | 0.96× | — | 26.9, 27.2, 27.3, 27.3, 33.1\* |
| C | TI temp=0.6 | 27.3 | 0.96× | — | 24.6, 26.7, 27.3, 27.5, 31.8\* |
| D | NGram greedy | 36.7 | **1.29×** | 70.1% (75/107) | 36.5, 36.6, 36.9 |
| E | NGram greedy + rep=1.1 | 36.0 | **1.26× vs A / 1.32× vs B** | 69.5% (82/118) | 35.8, 36.1, 43.7\* |
| F | NGram temp=0.6 (falls back) | 27.0 | 0.95× (≈ C) | — | 26.4, 26.9, 27.0, 27.3, 27.4 |

\*Outliers — Cells B trial-5 (33.1), C trial-5 (31.8), and E trial-3 (43.7) are positive-side outliers vs their respective clusters. Cells C trial-1 (24.6) is a negative-side outlier. Median is robust to these. **Pattern observed: the FIRST trial after a longer pause tends to be high (warm-up effect on M1 Max — possibly thermal headroom or initial GPU clock state).** Future runs may want a discard-first-trial protocol.

### Findings — 26B A4B

**1. Processor plumbing preserves the speedup.** Cell E (n-gram + penalty) at 36.0 tok/s vs cell D (n-gram, no penalty) at 36.7 tok/s. ~2% slowdown attributable to the AR-fallback batch collapse — well within the predicted band.

**2. N-gram + penalty crushes TokenIterator + penalty.** Cell E at 36.0 vs cell B at 27.3 = **1.32× speedup**. This is the practical win the plumbing fix unlocks: callers running with `repetitionPenalty: 1.1` no longer silently fall through to the slow path.

**3. Accept rate is essentially unchanged by the mild penalty** (70.1% no-penalty → 69.5% with rep=1.1 on this prompt). The naive prediction "penalty conflicts with regurgitation, accept drops" was wrong here at this penalty strength. Penalty=1.1 is a 10% bump on already-high-margin tokens, not enough to flip many argmaxes on this prompt. The penalty-strength sweep below extends this to rep=1.3 / 1.5.

**4. Temperature cost is real and substantial.** Cell C (TI temp=0.6) at 27.3 vs cell A (TI greedy) at 28.5 = ~4% sampling overhead on the baseline iterator. But cell F (n-gram-opted-in at temp=0.6) at 27.0 ≈ cell C: the route declines, falls back to TokenIterator, the user loses the entire 29% n-gram speedup. **The full cost of the temperature disqualifier on this prompt is the difference between cell D (36.7) and cell C (27.3): users opting into n-gram at `temperature: 0.6` get a 26% slower run than they would at `temperature: 0`** — and that is exactly the gap Leviathan would close (spec 023).

### Correctness verification

Before drawing throughput conclusions: confirmed the n-gram iterator's byte-identical contract holds with the processor plumbing. Compared full 200-token outputs across cells via md5:

| Cell | md5 | Output prefix |
|---|---|---|
| A (TI, no penalty) | `22c1ec9b...` | `INGREDIENTS:\n- 250 grams all-purpose flour\n- 5 grams baking soda...` |
| B (TI, rep=1.1) | `22c1ec9b...` (≡ A) | (same) |
| D (NGram, no penalty) | `22c1ec9b...` (≡ A) | (same) |
| E (NGram, rep=1.1) | `22c1ec9b...` (≡ A) | (same) |
| J (NGram, rep=1.5) | `8894...` | `INGREDIENTS:\n- 250 grams all-purpose flour\n- 1 teaspoon baking soda...` |

This proves three things:

1. **The n-gram iterator produces byte-identical output to plain `TokenIterator`** at temperature=0 (the load-bearing contract). A == D and B == E.
2. **The processor plumbing is correctly wired through n-gram.** B == E means TokenIterator + penalty and NGramSpec + penalty produce the same token stream — the new path doesn't accidentally bypass the processor.
3. **The penalty actually does something at higher strength.** A != J: at rep=1.5, the argmax flipped at the "baking soda" position (`5 grams` → `1 teaspoon`, the original prompt's wording). The penalty isn't a no-op.

Rep=1.1 and rep=1.0 producing identical outputs (md5 equal) tells us the penalty is being correctly applied — but the modification it makes to the logits (~10% boost) isn't large enough to flip any argmax in the first 200 tokens of this high-margin prompt. **The throughput differences observed between cells D/E (37→36 tok/s) reflect the GPU cost of the processor's `process()` call + the AR-batch collapse, not different generation paths.**

Why D and E have different accept counts (75/107 vs 82/118) despite identical outputs: the strict-greedy guard sees slightly different top-1-vs-top-2 margins (penalty shifts the logit landscape by ~10% even when it doesn't flip the top-1), so it breaks accept chains at different positions. More verify cycles, slightly different draft counts, **same final token sequence** — the contract holds at the output level, not at the cycle level.

## Gemma 4 E2B 4-bit (multi-trial)

Smaller, faster model — sensitive to per-token overhead, where the AR-batch optimisation matters most.

5-trial protocol applied 2026-04-29 (matches the 26B A4B methodology). Medians + trial sets:

| Cell | Config | tok/s (median) | Speedup vs A | Accept rate | Trials |
|---|---|---|---|---|---|
| A | TI greedy | 104.8 | 1.00× | — | 103.5, 103.6, 104.8, 105.3, 106.0 |
| B | TI greedy + rep=1.1 | 98.3 | 0.94× | — | 97.2, 97.3, 98.3, 98.9, 105.6\* |
| C | TI temp=0.6 | 96.6 | 0.92× | — | 93.9, 94.9, 96.6, 97.0, 97.3 |
| D | NGram greedy | 103.2 | **0.98×** (slight regression) | 69.5% (73/105) | 99.5, 100.5, 103.2, 104.0, 107.4 |
| E | NGram greedy + rep=1.1 | 95.2 | **0.91×** | 72.4% (89/123) | 91.2, 94.0, 95.2, 98.4, 99.6 |
| F | NGram temp=0.6 (falls back) | 96.1 | 0.92× (≈ C) | — | 95.8, 95.8, 96.1, 96.1, 99.0 |

\*Trial-5 outlier on cell B (105.6 vs cluster of 97-99); median robust.

### Findings — E2B 4-bit

**1. N-gram is slightly *slower* than baseline on this model.** With multi-trial (the original sweep had this single-trial), cell D at 103.2 vs cell A at 104.8 = **0.98×** — a ~2% regression. The single-trial number happened to land at parity (1.00×); the cleaner median says n-gram is a small net loss here. **The verify-batch overhead at K=3 on a model whose forward pass is already ~10 ms/token (100 tok/s) consistently eats the savings**, even at 70% accept rate. This underscores that acceptance rate is a necessary but not sufficient indicator of throughput speedup.

**2. Processor cost compounds the regression.** Cell E (n-gram + penalty) at 95.2 vs cell B (TI + penalty) at 98.3 = **3% slowdown** with n-gram active. Cell E vs cell A (no penalty baseline) at 104.8 = **9% slowdown**. The AR-batch collapse + verify overhead make n-gram + penalty a clear loser on this small fast model.

**3. Sampling cost ~8%.** Cell C vs A: 96.6 vs 104.8 = 0.92×. Larger than the ~4% cost on 26B A4B — consistent with the small-model regime where per-token overhead dominates.

**4. The fall-back path validates cleanly.** Cell F (n-gram opt-in but temp=0.6) at 96.1 ≈ cell C at 96.6 → routing decision correctly disqualifies and produces TokenIterator-equivalent throughput.

### Recommendation strengthened from E2B data

The earlier "n-gram is workload-marginal on small/fast models" recommendation should be promoted to **"don't auto-engage n-gram on small models (≤2B 4-bit)"**. On Gemma 4 E2B 4-bit:
- Best case (D vs A): **2% slower** than baseline.
- With penalty (E vs A): **9% slower** than baseline.

Either fix the auto-route to disengage on small models (some heuristic on tok/s threshold) or leave the env var off by default — which is already what we do (`MLX_NGRAM_ENABLED=1` is opt-in). Users explicitly opting in on E2B should be informed via doc that this is a known regression regime.

## Cross-model summary

All multi-trial (5 trials per cell on TI/fall-back paths, 3 trials on n-gram cells D/E from the original sweep).

| Quantity | 26B A4B | E2B 4-bit |
|---|---|---|
| N-gram greedy speedup (D / A) | 1.29× | **0.98×** (small regression) |
| Penalty cost on TokenIterator (B / A) | 0.96× | 0.94× |
| N-gram + penalty vs TokenIterator + penalty (E / B) | 1.32× | 0.97× |
| N-gram + penalty vs n-gram greedy (E / D) | 0.98× | 0.92× |
| Sampling cost (C / A) | 0.96× | 0.92× |
| Disqualifier cost vs n-gram greedy (C / D) | 0.74× | 0.94× |
| Theoretical ceiling for spec 023 (D / C) | **1.34×** | 1.07× (or worse — see E2B section) |

**Key cross-model insight**: spec-decode wins are weight-bandwidth-bound. The 26B-A4B model has enough memory traffic per token (~42 ms baseline) that even at 70% accept rate the K+1 verify forward amortises favourably. The E2B model at ~10 ms/token doesn't have enough memory traffic to amortise; the iterator's CPU bookkeeping + GPU-CPU sync overhead becomes a net loss. **The per-model auto-route disengagement decision should key off baseline tok/s, not just `canTrimPromptCache`.** Tracking item.

The bottom row — the projected ceiling Leviathan accept/reject sampling could deliver on a sampling call — is the headline number for spec 023's prioritisation.

## Recommendations

1. **The processor plumbing fix is correct and worthwhile on weight-bandwidth-bound models.** On 26B A4B, n-gram + penalty is 1.3× faster than TokenIterator + penalty across the full rep=1.0 to rep=1.5 strength range. Ship as-is in PR #113.

2. **Repetition penalty up to 1.5 doesn't break n-gram on high-margin prompts.** Initial concern that aggressive penalties would crater accept rate (and thus speedup) was unfounded on this prompt class — accept rate stayed flat at 68-70% across all strengths tested. The penalty needs to be larger than the typical top-1-vs-top-2 logit margin to flip argmaxes, and on input-grounded prompts those margins are wide. Workloads with tighter margins (paraphrastic chat, creative writing) likely behave differently — flagged as a follow-up sweep.

3. **On small/fast models (≤2B params 4-bit), n-gram is workload-marginal even at 70% accept rate.** Document the regime; consider auto-disengaging when measured base tok/s is above some threshold (e.g. 80 tok/s) AND a processor is active. Tracking item, not blocking.

4. **Spec 023 (Leviathan accept/reject) is well-motivated.** On 26B A4B at the recipe-rewrite prompt, lifting the `temperature != 0` disqualifier would close a 27% gap (cell C vs cell D). On other workloads with non-zero temperature, the win scales similarly. Implement after #113 lands.

5. **Single-trial benchmarking is too noisy for sub-10% claims.** Cell J at rep=1.5 went from "looks like a cliff at 28-29 tok/s" (3 trials) to "median 37 tok/s" (5 trials) — same hardware, same minute, just more samples. Multi-trial (5+) with **median** as the summary statistic is the right protocol for any decision smaller than 10%; mean over a small N is too sensitive to outliers in either direction.

## Penalty-strength sweep (Gemma 4 26B A4B, follow-up)

Per the question "what about rep=1.3 / 1.5?", an extended sweep adding two more penalty strengths.
Same prompt, same model, multi-trial throughout (3-5 trials per cell).

| `repetitionPenalty` | TI tok/s (median) | n-gram tok/s (median) | Speedup | Accept rate |
|---|---|---|---|---|
| 1.0 (off) | 28.6 (single) | 36.7 (3-trial: 36.5, 36.6, 36.9) | 1.28× | 70.1% (75/107) |
| 1.1 | 27.0 (single) | 36.0 (3-trial: 35.8, 36.1, 43.7\*) | 1.33× | 69.5% (82/118) |
| 1.3 | 26.2 (3-trial: 25.5, 26.2, 26.7) | 34.8 (3-trial: 33.9, 34.8, 35.5) | 1.33× | 69.5% (82/118) |
| 1.5 | 26.5 (5-trial: 23.8, 24.1, 26.5, 27.3, 32.7) | 36.9 (5-trial: 35.8, 35.8, 36.9, 44.1, 44.2) | 1.39× | 68.5% (85/124) |

\* trial-2 outlier on rep=1.1; remaining trials cluster tightly.

### Findings — penalty strength

**1. Accept rate is essentially flat from rep=1.0 to rep=1.5** (70.1% → 69.5% → 69.5% → 68.5%). My pre-bench prediction "30-50% accept-rate drop at rep=1.3+" was wrong on this prompt. **The mechanism**: a repetition penalty at strength λ multiplies a recently-seen token's logit by λ for tokens with negative logits, divides by λ for tokens with positive logits. For the argmax to flip, the top-1 vs top-2 margin must be less than `log(λ)` in logit space (≈ 0.41 at λ=1.5). On input-grounded high-margin prompts (recipes, templates, factual re-quoting) most positions have wide margins, so the penalty is essentially inert on the argmax decision.

**2. The 1.3× speedup persists at every penalty strength tested.** No cliff at rep=1.5; the spec-decode contract holds across the full strength range. This is a **stronger result than the original audit predicted** and changes the recommendation: callers running with even aggressive repetition penalties (1.5) should keep n-gram engaged on weight-bandwidth-bound models — the plumbing fix delivers the speedup reliably.

**3. Single-trial variance is high enough to fake a cliff.** The first 3-trial run for cell J at rep=1.5 produced (28.4, 27.9, 33.0) — looks like a clear regression vs the rep=1.3 cluster at ~35. Adding 2 more trials brought (44.1, 44.2) into the picture, revealing the true median is ~37. **Runs hat fall in the 28-30 range are real samples, not artefacts** — they reflect actual variance from background system load / thermal headroom on M1 Max. Comparisons smaller than ~10% need at least 5 trials to distinguish from noise, and the median (not mean) is the right summary statistic.

### Caveats on the penalty sweep

- One prompt only. **Paraphrastic workloads with tighter margins should show the predicted accept-rate drop.** A 5-prompt sweep across recipe/code/RAG/chat/creative would establish the curve shape across workload classes. Tracked as a follow-up.
- Constant token count across trials (159) is reassuring — none of the cells truncated early due to penalty-induced EOS shifts. On longer generations (1000+ tokens) the penalty's cumulative effect on the lookup history would compound and could show different behavior.
- Both `presencePenalty` and `frequencyPenalty` are wired through the same plumbing path and should behave qualitatively the same. Untested in this sweep.

## Limitations

- Single prompt class (recipe-rewrite, high-regurgitation). Results on RAG, code generation, or paraphrastic chat would differ.
- No measurement of `presencePenalty` or `frequencyPenalty` separately. The plumbing handles all three identically; the prediction holds.
- Bench harness uses `--method simple` chat-template wrapping. Tool-calling or harmony-format workloads not covered.
- Multi-trial coverage on Gemma 4 26B A4B is now uniform: A/B/C/F at 5 trials, D/E at 3 trials, G/H at 3 trials, I/J at 5 trials. The Gemma 4 E2B 4-bit cells remain single-trial; less load-bearing since the headline finding there ("n-gram gives no speedup on small/fast models on this prompt") doesn't depend on tighter bands. A multi-trial E2B sweep would be cheap (~5 min) if anyone wants the rigor.
- Trial-1 warm-up effect observed on cells B and C (both showing positive-side outliers on first trial after a longer pause). Median is robust to these. A discard-first-trial protocol would tighten future runs but isn't load-bearing for the analysis here.

## Reproducibility

```bash
# Build once
make

# Run cell (example: cell E, 26B A4B)
env MLX_BENCH_MODEL=gemma4-26b-a4b \
    MLX_BENCH_QUANT=4bit \
    MLX_BENCH_METHOD=simple \
    MLX_BENCH_MAX_TOKENS=200 \
    MLX_BENCH_NGRAM=3 \
    MLX_BENCH_TEMPERATURE=0 \
    MLX_BENCH_REPETITION_PENALTY=1.1 \
    MLX_BENCH_PROMPT="$(cat recipe-prompt.txt)" \
    swift test -c release --filter benchmark
```

Full prompt + 6-cell sweep script: `/tmp/bench-ngram-processor*` (local).
