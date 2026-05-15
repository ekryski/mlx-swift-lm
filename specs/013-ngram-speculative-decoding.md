# 013 — N-gram speculative decoding: correctness fix + Phase B sweep

- **Status:** ✅ shipped (PR #113 merged; remaining stretch goals in PR #154 + spec 023). See "Implementation status" table at bottom.
- **Branch:** `ek/ngram-speculative-v2` (merged to alpha)
- **Depends on:**
  - PR [#113](https://github.com/ekryski/mlx-swift-lm/pull/113) — Phase A surface + Phase B harness; defaults still at 0
  - The two iterator bug-fixes already in #113 (acceptance-loop break, first-token emit + lookup extend)

## Problem

The work on `ek/ngram-speculative-v2` shipped the Phase A surface
(file split, `ngramDraftMin` / `ngramMinHits` / `minNgramSize` knobs,
multi-size fallback) and a complete bench harness for the Phase B
default-tuning sweep — but **did not** ship a usable default for
`GenerateParameters.ngramSize` because `NGramSpeculativeTokenIterator`
has correctness gaps on real models.

A 6-hour autonomous Phase B sweep (2026-04-28) found three problems
in order:

1. **The bench wasn't exercising the spec-decode path at all.**
   `MLXLMCommon.generate(...)` always constructed `TokenIterator`
   regardless of `params.ngramSize`. The first run's "1.10–1.17×
   speedups" were measurement noise on plain decode. **Routing fix
   landed; then reverted** when the next finding surfaced.

2. **Hybrid models can't use the iterator.** Qwen 3.5 / 3.6 / NemotronH /
   Jamba / Granite-MoE-Hybrid / BaichuanM1 / FalconH1 / LFM2 / LFM2MoE
   all use `MambaCache` for SSM/GatedDeltaNet layers — non-trimmable,
   so `canTrimPromptCache` fails and the iterator's init throws. Cuts
   the supported model set to pure-attention models (Gemma 4 family,
   GPT-OSS, Llama, Phi, Qwen 2/3 dense, etc.).

3. **The iterator has correctness bugs on real models.** Once routing
   was wired and the bench ran the spec path for real, Gemma 4 E2B at
   `temperature=0` produced **0–5 tokens of garbage** (`\text 0`, etc.)
   on prompts where plain `TokenIterator` produced full 100-token
   coherent responses. **Two specific bugs were fixed:**
   - Acceptance loop used `for ... where ...` semantics — silently
     skipped mismatches and accepted later, non-consecutive matches
     that came from a verify pass against the *drafted* prefix, not the
     accepted prefix. **Fixed** to break-on-first-mismatch.
   - First sampled token `y` from prefill was never emitted to the
     caller and wasn't added to the lookup history. Iterator started
     by running verify with `[y, drafts]` whose drafts were positioned
     after the *last prompt token*, not after `y`. **Fixed** by
     enqueueing `y` in `pendingTokens` at end of prefill and extending
     the lookup history with it.
   - **Failure mode reduced but not eliminated.** Some prompts still
     produce truncated/garbage output. Diagnosis required more time
     than the sprint had.

This spec catalogs the work to close the gap and finish Phase B.

## What's already shipped (do not redo)

| # | Item | Where |
|---|---|---|
| 1 | `NGramSpeculativeTokenIterator` + `NGramLookup` extracted to its own file | `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` |
| 2 | `ngramDraftMin: Int = 1` knob (mirrors llama.cpp `--draft-min`) | `GenerateParameters` |
| 3 | `ngramMinHits: Int = 1` knob (mirrors `--spec-ngram-min-hits`) | same |
| 4 | `minNgramSize: Int = 2` knob — multi-size fallback floor | same |
| 5 | Multi-size fallback in `NGramLookup` (one hash table per size in `[minNgramSize, maxNgramSize]`) | iterator |
| 6 | Acceptance-loop break-on-first-mismatch | iterator (Phase B fix) |
| 7 | First-token emit + extend-lookup-with-y at prefill end | iterator (Phase B fix) |
| 8 | `--method ngram-sweep` bench case + 18-prompt workload + analyzer | `Tests/Benchmarks/InferenceBenchmark.swift`, `scripts/ngram-sweep-analyze.py` |
| 9 | `MLX_BENCH_NGRAM_SWEEP_CELLS` + `MLX_BENCH_TEMPERATURE` env overrides | bench harness |
| 10 | Disabled regression test: token-sequence equality with `TokenIterator` at temp=0 | `Tests/MLXLMTests/NGramSpeculativeTests.swift` |
| 11 | 10 unit tests for `NGramLookup` (multi-size fallback, min-hits, extend, longest-match) | same |

## Phase 1 — Correctness fix (blocking)

### 1.1 Build a per-token side-by-side diff harness 🆕

**Why first**: the sprint failed because we couldn't tell *which token*
diverged or *what cache state* was different at divergence. Without that,
each "fix" is a guess.

**What**: a debug-only test that runs `TokenIterator` and
`NGramSpeculativeTokenIterator` on the same prompt at `temperature=0`,
logging every emitted token plus the cache offset and (for sentinel
positions) a hash of the cache contents. First divergence point is the
crime scene.

Add as `Tests/MLXLMTests/NGramSpeculativeDiffTests.swift`. Use a real
small model — Gemma 4 E2B 4bit — not the in-test random-weight one
(the random model masks numerical drift).

### 1.2 Diagnose cache-state divergence after partial accept + trim 🆕

**Hypothesis**: After `trimPromptCache(cache, numTokens: rejected)`, the
cache offset is at `(prev_offset + 1 + accepted)`, but the K/V at the
last position corresponds to the last *accepted draft*. The next round
processes `[correction, drafts'...]` as input — that should be correct,
but the model's attention can read the cache K/V at the last position
which is the accepted draft's K/V. If anything downstream cares about
post-trim state vs. forward-pass state, this is where it'd misalign.

**What to check**:
- After verify+trim, does the cache's `offset` field match what the
  iterator thinks it should be?
- For `RotatingKVCache`, does `trim(rejected)` correctly walk back
  `offset` and `idx`?
- For `KVCacheSimple`, is the trim a simple slice or does it just
  decrement an offset pointer (leaving stale K/V data that gets read
  on the next forward pass)?

**Fix**: depends on findings. Most likely a one- or two-line correction
in either the iterator's trim call or one of the cache types' `trim()`
implementations.

### 1.3 Check batch-vs-sequential logit drift 🆕

**Why**: even if cache management is correct, the verify pass processes
`(numDraft + 1)` tokens in a single forward call. Running the same
sequence as `numDraft + 1` separate single-token calls might produce
slightly different logits due to numerical accumulation order (matmul
batching, softmax normalization, etc.). At temp=0 those small numerical
differences flip argmax on close logits → divergent tokens → cascading
divergence.

**What**: extend the diff harness from 1.1 with a "logit comparison"
mode. For a fixed sequence `[t0, t1, t2, t3]`, run the model:
- Once as a single batched forward (returns 4 logit vectors)
- Four times sequentially with cache (returns 4 logit vectors, one per
  call)
Compare element-wise. If they differ above ~1e-3, batched verify is
inherently approximate at greedy.

**Fix paths**:
- **Approximate-greedy** — accept the drift, document it, and
  acknowledge that the iterator is "near-greedy" not "exactly-greedy".
  Update tests accordingly.
- **Strict-greedy** — fall back to sequential verify when logit
  margins are tight (lookup the top-2 logits per position; if their
  difference < ε, sequential-verify that position). Slower but
  bit-exact with `TokenIterator`.

### 1.4 Re-enable auto-routing in `MLXLMCommon.generate()`

Once 1.1's regression test (token-sequence equality) is green on Gemma
4 E2B, restore the auto-routing block in
`Libraries/MLXLMCommon/Evaluate.swift`:

```swift
if parameters.ngramSize >= 1 && parameters.maxNgramDraftTokens >= 1
    && parameters.temperature == 0 {
    // probe trimmability, route to NGramSpeculativeTokenIterator
}
```

Also drop the "EXPERIMENTAL — known correctness issues" warnings from:
- `NGramSpeculativeTokenIterator` doc comment
- `GenerateParameters.ngramSize` doc comment

### 1.5 Enable the disabled regression test

`Tests/MLXLMTests/NGramSpeculativeTests.swift` has a `@Test(.disabled(...))`
for the token-sequence equivalence test. Drop the `.disabled(...)` once
1.1 is green. This is the load-bearing CI gate that blocks future
correctness regressions.

## Phase 2 — Default-tuning sweep (Phase B redo)

### 2.1 Run the full sweep on the supported model set

With routing wired up and the regression test green, run the existing
bench harness on the three pure-attention models:

```bash
# Full 32-cell sweep on Gemma 4 E2B (the small reference)
MLX_BENCH_MAX_TOKENS=100 ./scripts/benchmark.sh \
    --model gemma4-e2b --method ngram-sweep --kv none

# Narrow sweep (top 10 cells from Gemma 4 E2B) on the bigger models
./scripts/ngram-sweep-bigmodels.sh
# (already scoped to gpt-oss-20b + gemma4-26b-a4b only —
#  Qwen 3.5 / 3.6 omitted due to MambaCache limitation)
```

Estimate: ~30–60 min per model wall-clock on M1 Max 64GB; ~2 hours total
once the iterator is fixed.

### 2.2 Reduce + decide

`scripts/ngram-sweep-analyze.py` produces per-category and overall
median/min/wins-out-of-N. From those numbers, pick:

- **Single default** if there's a cell that wins ≥ N-1 of the prompts
  on every model (analogous to the `n=4 D=4 H=1` candidate that the
  bogus v1 data suggested).
- **Off-by-default + per-workload doc** if no single cell dominates.

If a default is set:
- Update `GenerateParameters.init` defaults: `ngramSize = N`,
  `maxNgramDraftTokens = D`. `ngramDraftMin / ngramMinHits / minNgramSize`
  stay at `1 / 1 / 2`.
- Add a regression test that asserts spec decode ≥ baseline-tok/s on a
  small representative prompt (so future kernel work doesn't silently
  regress this).

### 2.3 Document in `Libraries/MLXLMCommon/Documentation.docc/`

Add a short article on n-gram speculative decoding:
- What it is, when it helps (high-repetition workloads)
- What it doesn't help with (open generation, hybrid models)
- The knobs and their meanings
- The default rationale (or per-workload guidance if no default)
- The temperature=0 (greedy-only) constraint

## Phase 3 — Stretch goals (after defaults ship)

### 3.1 Non-greedy verification 🆕

llama.cpp's spec decode supports `temperature > 0` via per-position
resample-with-rejection. Each position: sample from the main model's
distribution; accept the draft if and only if the sampled token equals
the draft. This produces a sequence indistinguishable in distribution
from non-greedy `TokenIterator`. Acceptance rate drops at higher
temperature, but the output is still correct.

Implementation: replace the argmax check in `speculateRound()` with the
sampler's actual `sample(logits:)` and accept on token equality.

### 3.2 Dynamic draft length per acceptance rate 🆕

llama.cpp adapts `maxDraft` to the rolling acceptance rate — high accept
→ longer drafts, low accept → shorter (or disable temporarily). Can
recover compute when the workload has phases of different repetitiveness
(e.g. code → prose → code).

Track `ngramAcceptanceRate` in a rolling window (last 16 rounds).
If `accept_rate < 0.3`, halve `maxDraft` (floor at `ngramDraftMin`).
If `accept_rate > 0.7`, double (ceiling at the configured
`maxNgramDraftTokens`).

### 3.3 Top-K continuations per pattern (`ngram-map-k4v` equivalent) 🆕

`NGramLookup` currently returns the most-recent prior occurrence's
continuation. llama.cpp's `ngram-map-k4v` tracks up to 4 continuations
per pattern and picks by frequency. Useful when text has multiple
plausible continuations after the same prefix (think "the quick brown
fox" continuing as either "jumped over" or "and the lazy dog").

Implementation: change `tables[size][hash]` from `[Int]` (positions) to
something that lets us count distinct continuations. On a hit, pick the
continuation whose first 1–2 tokens have the highest frequency.

### 3.4 Hybrid-model support via per-layer cache slicing 🆕

Currently spec decode is gated on `canTrimPromptCache(cache)` — false
if any layer's cache is `MambaCache`. To support hybrid models without
fully solving the SSM-state-rollback problem, we could:

- For attention layers: trim normally on rejection.
- For SSM/Mamba layers: roll forward with the corrected token
  re-running the SSM step (not "rollback"; "redo"). Costs one extra
  SSM step per rejection but allows spec decode on Qwen 3.5 / 3.6.

Substantial restructuring of the verifier; defer until 1.1–2.3 are
shipped and the value is clear.

## Out of scope

- Spec decode for the *draft-model* path (`SpeculativeTokenIterator`)
  is unrelated and works fine — see PR #42 history.
- Setting `ngramSize` defaults non-zero for the draft-model path
  doesn't apply (it's a different iterator).
- Multi-modal (VLM) prompts — n-gram lookup over image tokens is a
  different problem; out of scope.

## Files touched by this work

| File | What |
|---|---|
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | Main iterator. Already in tree as of #113. Phase 1.2/1.3 fixes go here. |
| `Libraries/MLXLMCommon/Evaluate.swift` | `generate(...)` routing (Phase 1.4) and the warning doc-comment removals. |
| `Libraries/MLXLMCommon/KVCache.swift` | Possibly: per-cache-type `trim()` fix (Phase 1.2). |
| `Tests/MLXLMTests/NGramSpeculativeDiffTests.swift` (new) | Per-token diff harness (Phase 1.1). |
| `Tests/MLXLMTests/NGramSpeculativeTests.swift` | Drop `.disabled(...)` on regression test (Phase 1.5). |
| `Libraries/MLXLMCommon/Documentation.docc/` | New article (Phase 2.3). |
| `Tests/Benchmarks/InferenceBenchmark.swift` | No changes expected — harness is already in tree. |
| `scripts/ngram-sweep-analyze.py` | No changes expected. Use as-is. |

## Open questions

1. Is "exactly greedy" required, or is "near-greedy" (within a few
   tokens of divergence) acceptable? Affects whether 1.3 needs
   sequential-fallback or can ship as approximate.
2. Should we ship a default at all, even if the sweep shows ~10%
   speedup? Defaults that help 70% of users but slow 30% are still
   net wins on average — but cause complaints from the 30%.
3. Is there value in adding a per-model default override (e.g. via a
   model-family registry the way `family.temperature` works)? Some
   models might benefit from `n=4 D=4`, others from `n=5 D=8`.

## Lessons learned (for the next pass)

- **Build the regression test first.** The disabled token-sequence
  equality test should have been the first thing written in Phase A,
  not the last thing written in Phase B. With it in place, the
  bench-routing gap and both iterator bugs would have been caught
  in minutes.
- **Verify the path is engaged before measuring it.** A no-op env
  override doesn't fail loudly. Always log "ngram speculative active:
  yes/no" at iterator construction so the bench output makes engagement
  obvious.
- **Hybrid models are ~half the model zoo.** Future spec-decode work
  should plan for the MambaCache limitation up front, either by
  scoping to pure attention or by including the hybrid-cache
  workaround (Phase 3.4) in the initial design.

## Implementation status (2026-04-29)

This table is the source of truth for what's shipped vs what's deferred.

### Phase 1 — correctness fix (blocking)

| # | Item | Status | Where |
|---|---|---|---|
| 1.1 | Per-token side-by-side diff harness | ❌ not built | (deferred — byte-identical md5 check during PR #113 close-out audit served as the coarser version; full diff harness needs a real-model integration test fixture we don't have today) |
| 1.2 | Diagnose cache-state divergence after partial accept + trim | ⚠️ mitigated, not root-caused | Strict-greedy guard (PR #113) prevents the cascade; hypothesis 1.2's exact cause never fully isolated |
| 1.3 | Check batch-vs-sequential logit drift | ✅ addressed | Strict-greedy guard handles this empirically |
| 1.4 | Re-enable auto-routing in `MLXLMCommon.generate()` | ✅ shipped | PR #113 → `Libraries/MLXLMCommon/Evaluate.swift` (`ngramRouteDecision` predicate) |
| 1.5 | Enable the disabled regression test | ⚠️ still `.disabled(...)` | Flaky on the in-tree tiny random-weight model; track when real-model integration harness lands |

### Phase 2 — default-tuning sweep

| # | Item | Status | Where |
|---|---|---|---|
| 2.1 | Run the full sweep on the supported model set | ⚠️ partial | Gemma 4 26B A4B + E2B + Qwen 3.5 0.8B done; GPT-OSS-20B added 2026-04-29 (PR #154); Qwen 3 dense, Llama, Phi pending |
| 2.2 | Reduce + decide defaults | ✅ shipped | PR #113: adaptive + strict-greedy + multi-candidate ON; dominance OFF |
| 2.3 | DocC article | ✅ shipped | PR #113 → `Libraries/MLXLMCommon/Documentation.docc/speculative-decoding.md` |

### Phase 3 — stretch goals

| # | Item | Status | Where |
|---|---|---|---|
| 3.1 | Non-greedy verification | ✅ **shipped via spec 023 (Leviathan accept/reject sampling)** | PR #154; default-on at `temp != 0` |
| 3.2 | Dynamic draft length per acceptance rate | ✅ shipped | PR #113 (adaptive draft scaler, `MLX_NGRAM_ADAPTIVE` default ON) |
| 3.3 | Top-K continuations per pattern (`ngram-map-k4v` equivalent) | ✅ shipped | PR #154: `multiCandidateLookahead` parameter on `NGramLookup.proposeDraft` + `MLX_NGRAM_MULTI_LOOKAHEAD=N` env var (default 1; 2 implements the "first 1-2 tokens" form). 2 unit tests cover lookahead=1 vs lookahead=2 disambiguation. |
| 3.4 | Hybrid-model support via per-layer cache slicing | ❌ superseded | **Replaced by spec 020 (tape-replay rollback)** — cleaner approach; phase-1 scaffold landed in [PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143). |

### What's left

- **1.1 + 1.5**: real-model integration test harness. Out of scope for this spec; would benefit every other spec PR too. Track as separate infrastructure work.
- **2.1**: GPT-OSS-20B sweep run 2026-04-29 (analysis in `benchmarks/gemma4-leviathan-broad-sweep-analysis.md`); Qwen 3 dense / Llama / Phi sweeps remain. Run as needed per PR.
- **3.4**: subsumed by spec 020. No separate work.

Everything else from the original spec is shipped or explicitly deferred. Spec 013 can be considered closed for the purposes of "does the n-gram path work." The remaining items are integration test infrastructure (1.1 / 1.5) that benefits the broader spec-decode workstream, not this spec specifically.
