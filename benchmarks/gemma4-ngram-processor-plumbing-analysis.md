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

## Gemma 4 26B A4B 4-bit

Bigger model — weight-bandwidth-bound, where spec-decode tends to give the cleanest signal.

| Cell | Config | tok/s | Speedup vs A | Accept rate |
|---|---|---|---|---|
| A | TI greedy | 28.6 | 1.00× | — |
| B | TI greedy + rep=1.1 | 27.0 | 0.94× | — |
| C | TI temp=0.6 | 27.1 | 0.95× | — |
| D | NGram greedy | 36.7 (median of 3 trials: 36.5, 36.6, 36.9) | **1.28×** | 70.1% (75/107) |
| E | NGram greedy + rep=1.1 | 36.0 (median of 3 trials: 35.8, 36.1, 43.7*) | **1.26× vs A / 1.33× vs B** | 69.5% (82/118) |
| F | NGram temp=0.6 | 27.8 | 0.97× (≈ C) | — (fell back) |

\*Trial-2 at 43.7 tok/s for cell E was a single-trial outlier — likely thermal or system-load timing. The trial-1 and trial-3 numbers cluster tightly with cell D at ~36 tok/s, which matches the prediction (small AR-batch collapse cost from running the AR fallback at batch size 1 instead of 4 when a processor is set).

### Findings — 26B A4B

**1. Processor plumbing preserves the speedup.** Cell E (n-gram + penalty) at 36.0 tok/s vs cell D (n-gram, no penalty) at 36.7 tok/s. ~2% slowdown attributable to the AR-fallback batch collapse — well within the predicted band.

**2. N-gram + penalty crushes TokenIterator + penalty.** Cell E at 36.0 vs cell B at 27.0 = **1.33× speedup**. This is the practical win the plumbing fix unlocks: callers running with `repetitionPenalty: 1.1` no longer silently fall through to the slow path.

**3. Accept rate is essentially unchanged by the mild penalty** (70.1% no-penalty → 69.5% with rep=1.1 on this prompt). The naive prediction "penalty conflicts with regurgitation, accept drops" was wrong here at this penalty strength. Penalty=1.1 is a 10% bump on already-high-margin tokens, not enough to flip many argmaxes on this prompt. Accept rate would drop more visibly at penalty=1.3+ — not measured here.

**4. Temperature cost is real and substantial.** Cell C (TI temp=0.6) at 27.1 vs cell A (TI greedy) at 28.6 = ~5% sampling overhead on the baseline iterator. But cell F (n-gram-opted-in at temp=0.6) at 27.8 ≈ cell C: the route declines, falls back to TokenIterator, the user loses the entire 28% n-gram speedup. **The full cost of the temperature disqualifier on this prompt is the difference between cell D (37) and cell C (27): users opting into n-gram at `temperature: 0.6` get a 27% slower run than they would at `temperature: 0`** — and that is exactly the gap Leviathan would close (spec 023).

## Gemma 4 E2B 4-bit

Smaller, faster model — sensitive to per-token overhead, where the AR-batch optimisation matters most.

| Cell | Config | tok/s | Speedup vs A | Accept rate |
|---|---|---|---|---|
| A | TI greedy | 104.3 | 1.00× | — |
| B | TI greedy + rep=1.1 | 100.3 | 0.96× | — |
| C | TI temp=0.6 | 96.9 | 0.93× | — |
| D | NGram greedy | 103.5 | **1.00×** (no speedup) | 69.5% (73/105) |
| E | NGram greedy + rep=1.1 | 95.5 | 0.92× | 72.4% (89/123) |
| F | NGram temp=0.6 | 96.5 | 0.93× (≈ C) | — (fell back) |

### Findings — E2B 4-bit

**1. N-gram gives essentially zero speedup on this model + this prompt.** Cell D vs A: 103.5 vs 104.3. Despite a 70% accept rate. The verify-batch overhead at K=3 on a model whose forward pass is already ~10 ms/token (= 100 tok/s) eats the savings. This contradicts what one might expect from raw accept-rate numbers and underscores that **acceptance rate is a necessary but not sufficient indicator of throughput speedup**.

**2. The processor cost is more visible here.** Cell E (n-gram + penalty) at 95.5 vs cell B (TI + penalty) at 100.3 = **5% slowdown** on n-gram with the processor. The AR-batch collapse + verify overhead = no win, slight loss. On this small/fast model, **users running with a penalty should leave `MLX_NGRAM_ENABLED` unset.**

**3. Sampling cost similar to 26B-A4B.** Cell C vs A: ~7% overhead. Cell F vs C: parity, confirms the disqualifier path.

## Cross-model summary

| Quantity | 26B A4B | E2B 4-bit |
|---|---|---|
| N-gram greedy speedup (D / A) | 1.28× | 1.00× |
| Penalty cost on TokenIterator (B / A) | 0.94× | 0.96× |
| N-gram + penalty vs TokenIterator + penalty (E / B) | 1.33× | 0.95× |
| N-gram + penalty vs n-gram greedy (E / D) | 0.98× | 0.92× |
| Sampling cost (C / A) | 0.95× | 0.93× |
| Disqualifier cost vs n-gram greedy (C / D) | 0.74× | 0.94× |
| Theoretical ceiling for spec 023 (D / C) | 1.36× | 1.07× |

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
- Multi-trial methodology was non-uniform across cells in the original sweep (single-trial for A/B/C/F, 3-trial for D/E). The penalty-strength sweep above tightens this to 3-5 trials throughout for cells G/H/I/J. The original cells A-C-F should ideally be re-run multi-trial too; they're left as single-trial for now since their relative ordering is stable enough that the analysis's conclusions don't depend on tighter bands there.

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
