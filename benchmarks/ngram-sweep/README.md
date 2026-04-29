# N-gram speculative decoding — Phase B sweep results

**Status**: Inconclusive. Defaults shipped at `ngramSize=0` (off). Auto-routing
in `MLXLMCommon.generate()` is **not enabled** pending iterator correctness
fix. See "Honest summary" at the bottom.

## Background

The Phase B goal was to run a workload sweep over `(ngramSize, maxDraft, minHits)`
on a representative model spread (Qwen3.5-0.8B, Gemma 4 E2B, then larger) and
pick defaults that work across workloads. Three findings forced a re-scope:

1. **The bench harness wasn't actually exercising the spec-decode path.**
   `MLXLMCommon.generate()` always constructed `TokenIterator`, never
   `NGramSpeculativeTokenIterator`, regardless of `params.ngramSize`. The
   first sweep (594 runs on Qwen 0.8B) was measuring runtime noise on the
   plain decode path — the `1.10–1.17×` "speedups" we observed in v1 are
   coincidental variance, not real spec-decode effects. **All v1 data
   discarded.**

2. **Hybrid models can't use the iterator at all.** Qwen 3.5 (any size,
   including 35B-A3B MoE), Qwen 3.6 (Qwen3-Next architecture), NemotronH,
   Jamba, Granite-MoE-Hybrid, BaichuanM1, FalconH1, and LFM2 / LFM2MoE all
   use `MambaCache` for their SSM/GatedDeltaNet layers. `MambaCache` is
   non-trimmable (recurrent state is cumulative — there's no per-token
   slot to roll back when a draft is rejected), so the iterator's
   `canTrimPromptCache` precondition fails and construction throws.

3. **The iterator has real correctness bugs**, surfaced once we fixed
   the bench wiring and got it actually running. On Gemma 4 E2B at
   `temperature=0`, several prompts produced 0–5 tokens of garbage
   output (`\text 0`, etc.) where plain `TokenIterator` produced
   coherent 100-token responses. Two specific bugs identified:

   - **Acceptance loop used `for ... where ...` semantics**, which skips
     mismatches but continues iterating — producing non-consecutive
     accepts (e.g., reject token 3, then accept token 4) that came from
     a verify pass against the *drafted* prefix rather than the actually-
     accepted prefix. **Fixed** to break-on-first-mismatch.
   - **First sampled token `y` was never emitted** to the caller and
     wasn't added to the lookup history. Iterator started by running
     verify with `[y, drafts]` whose drafts were positioned after the
     last *prompt* token, not after `y`. **Fixed** by enqueueing `y` to
     `pendingTokens` at end of prefill and extending the lookup history
     with it.

   These two fixes reduced but did not eliminate the failure mode.
   Remaining failures are likely some combination of cache-state
   divergence after partial-accept-then-trim and batch-vs-sequential
   logit drift causing argmax flips that cascade into early stop-token
   emission. Diagnosing requires more time than this sweep budgeted.

## What we actually shipped

- All Phase A knobs and the file split land as planned.
- The two iterator bug fixes land (acceptance-loop break, first-token
  emit + lookup extend) — incremental progress, but not enough to ship
  routing as a default.
- `MLXLMCommon.generate(input:cache:parameters:context:wiredMemoryTicket:)`
  does **not** auto-route to `NGramSpeculativeTokenIterator`. Setting
  `parameters.ngramSize > 0` on the standard generate pipeline is
  currently a no-op. Documented inline.
- A disabled stricter test (`N-gram spec decode matches TokenIterator
  under greedy (sequence)`) is in tree as a regression target. Once it
  passes on a real-model integration harness, routing can be re-enabled.
- The bench `--method ngram-sweep` infrastructure stays in tree (workload,
  CLI plumbing, narrow-cells override, `MLX_BENCH_TEMPERATURE` override)
  for use once the iterator is fixed.

## What changed in `GenerateParameters`

Defaults all unchanged:

```swift
ngramSize: Int = 0
maxNgramDraftTokens: Int = 0
ngramDraftMin: Int = 1
ngramMinHits: Int = 1
minNgramSize: Int = 2
```

`ngramDraftMin`, `ngramMinHits`, and `minNgramSize` are still well-defined
knobs and still flow through to the iterator when constructed directly —
they just don't engage automatically through `generate()`.

## Sweep design (preserved for the next attempt)

For each model, run every prompt in `Tests/Benchmarks/Resources/ngram-sweep-prompts/`
under:

- 1 baseline (`ngramSize = 0`, pure autoregressive)
- N sweep cells over `(ngramSize, maxDraft, minHits)` triples.

**Small models**: full 32-cell matrix
(`n ∈ {2,3,4,5}`, `D ∈ {4,8,12,16}`, `H ∈ {1,2}`). 18 prompts × 33 runs = 594
runs each.

**Big models**: narrow sweep over the top 10 cells from combined small-model
data via `MLX_BENCH_NGRAM_SWEEP_CELLS=...` to keep wall-clock manageable.

Workload categories (3 prompts each, 18 total):

- **code-refactor** — high-repetition (function names, types, structure)
- **code-completion** — medium-high (variable names, control flow)
- **qa-requote** — medium-high (answer often re-quotes prompt)
- **chat-instruction** — low-medium (control)
- **summarization** — low-medium (control)
- **open-generation** — low (control)

All runs use `MLX_BENCH_MAX_TOKENS=100` and `--kv none`. The sweep forces
`MLX_BENCH_TEMPERATURE=0` so the verifier accepts on argmax match
(greedy-equivalence is the only correctness regime today).

## Running the sweep (after iterator is fixed)

```bash
# Full sweep on a small model
MLX_BENCH_MAX_TOKENS=100 ./scripts/benchmark.sh \
    --model gemma4-e2b --method ngram-sweep --kv none

# Narrow sweep on a big model (when small-model winners are known)
MLX_BENCH_MAX_TOKENS=100 \
MLX_BENCH_NGRAM_SWEEP_CELLS="4:4:1,5:12:1,5:12:2,4:4:2,4:8:1,3:16:1,3:16:2,4:12:2,4:16:1,5:16:2" \
./scripts/benchmark.sh --model gpt-oss-20b --method ngram-sweep --kv none

# Or use the chain script (needs `generate()` routing to be wired up first)
./scripts/ngram-sweep-bigmodels.sh
```

## Analysis

```bash
python3 scripts/ngram-sweep-analyze.py LOG.log [LOG.log ...]
```

## Honest summary

Going into Phase B I expected to ship `n=4 D=4 H=1` as a default with ~10–15%
median speedup. I should have built a stricter regression test (token-sequence
equality vs. count equality) at the start of Phase A — that would have caught
the bench-routing bug and the iterator correctness bugs much earlier.

What this sweep is good for:

- The **two iterator bugs identified and fixed** are real improvements.
  Whoever picks this up next is starting from a meaningfully better
  baseline than the original implementation.
- The **bench infrastructure works end-to-end** once the iterator is
  fixed: workload, CLI, sweep matrix, narrow-cells override, analysis
  script, and a writeup template are all in tree.
- The **scope of model coverage is documented** — hybrid models with
  MambaCache can't use the iterator without significant restructuring
  (the verifier would need to either skip the SSM layers' cache update
  during draft, or track per-step state checkpoints).

What this sweep is *not* good for:

- Producing reliable performance numbers. Don't trust v1 data; v3/v4
  data was collected with the iterator still buggy and shows correctness
  drift (truncated outputs).

## Next steps (for whoever picks this up)

1. **Investigate cache-state divergence post-trim** on real models. The
   most likely suspect is: after `trimPromptCache(cache, numTokens: rejected)`,
   the cache is at offset `(prev_offset + 1 + accepted)` but no token has
   been processed at the *new* end position. The next round's verify pass
   processes `[correction, drafts'...]` against a cache whose last K/V
   entry corresponds to the last *accepted* draft (correct) — but does
   the model's forward pass index this correctly? Worth a print-debug
   compared to TokenIterator's per-token forward pass.

2. **Check batch-vs-sequential logit drift.** Run the same `[y, draft_0,
   draft_1, ...]` through `mainModel(...)` once as a 4-token batch and
   then again as 4 sequential 1-token calls. Compare the logits at each
   position. If they differ above a numerical noise floor, that's the
   source of argmax flips and the iterator can never be greedy-equivalent
   in batched-verify form. Fix is either to fall back to sequential-verify
   on tie-prone positions, or to accept the drift as bounded and document
   that "greedy-equivalent" is approximate.

3. **Once the disabled sequence-equivalence test passes**, re-enable
   the routing in `MLXLMCommon.generate()`, drop the warning in the doc
   comment, and re-run the Phase B sweep on Gemma 4 E2B + GPT-OSS-20B +
   Gemma 4 26B-A4B (the three pure-attention models in our zoo big enough
   to see meaningful speedup).
