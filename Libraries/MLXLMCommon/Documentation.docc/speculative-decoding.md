# Speculative Decoding

`MLXLMCommon` ships a prompt-lookup speculative decoder
(``NGramSpeculativeTokenIterator``) that drafts continuation tokens from
the prompt and accepted-output history, then verifies them in a single
batched forward pass on **the same target model you'd use otherwise**.
On input-grounded workloads (code, templates, factual re-quoting, RAG)
this produces a 1.3–1.6× decode speedup over ``TokenIterator``:

- **Greedy** (`temperature: 0`): output is **byte-identical** to plain
  `TokenIterator` — same token stream, same final state.
- **Sampling** (`temperature > 0`): output is **distributionally
  equivalent** via Leviathan accept/reject sampling — samples from
  the same distribution as plain `TokenIterator(temperature: T)` would
  draw from. Default-on (set `MLX_NGRAM_LEVIATHAN=0` to disable).

On paraphrastic or open-ended workloads (creative writing, opinion,
generative content) it can be a 5–14% *regression*, and on small/fast
models (≤2B 4-bit at ~100 tok/s baseline) it's a 2-9% regression even
on favourable workloads — see "When does it help (and when does it
hurt)?" below for the full regime table.

It's **opt-in**. With a bare `GenerateParameters()` you get
``TokenIterator`` exactly as before — speculative decoding never
engages without an explicit signal from the caller.

## How n-gram speculative decoding works (briefly)

There is **only one model** in this loop: your target model (the one
you'd be running anyway). The "drafts" come from a CPU-side hash table
over the prompt + tokens already generated — no separate draft model,
no extra weights to load. Each cycle:

1. **Look up** the last few generated tokens in the hash table; if they
   appeared earlier in the prompt or generation, propose the K tokens
   that followed previously as the draft.
2. **Verify**: run the target model on `[last_token, draft_1, ..., draft_K]`
   in one batched forward pass — same cost as a single decode step on
   modern Apple-Silicon GPUs because attention is weight-bandwidth-bound.
3. **Accept** the longest matching prefix where the target's argmax at
   each position matches the draft. Emit those tokens plus the target's
   "bonus" token at the next position.
4. **Trim** the KV cache for any rejected draft positions and continue.

Throughout, the **target model's logits** are what the verify step
samples — penalties, processors, samplers all act on those. The
"draft" is just integer token positions from a lookup table; nothing
on the draft side accepts a logit processor.

Steps 1+3 are pure Swift / CPU; step 2 is the same forward pass plain
``TokenIterator`` would do (just over K+1 tokens instead of 1). When the
lookup table doesn't have a useful match, the iterator falls back to
**autoregressive (AR) decode** — one normal decode step at a time —
the same behavior you'd get from ``TokenIterator``.

## Three opt-in paths

The auto-routing in
``MLXLMCommon/generate(input:cache:parameters:context:wiredMemoryTicket:)``
checks for any of these signals; the first match wins:

### 1. Swift `GenerateParameters` (production)

Set both `ngramSize` and `maxNgramDraftTokens` to ≥ 1. This is the path
production code paths should use when they know they want speculative
decoding for a known set of requests.

```swift
let params = GenerateParameters(
    maxTokens: 512,
    temperature: 0,
    ngramSize: 3,           // primary n-gram size
    maxNgramDraftTokens: 4  // per-round draft cap
)
let stream = try generate(
    input: lmInput, parameters: params, context: context)
```

### 2. Environment variable (one-off / benchmark runs)

`MLX_NGRAM_ENABLED=1` enables n-gram speculative decoding without code
changes, applying sensible defaults (`ngramSize: 3`,
`maxNgramDraftTokens: 4` — picked from the `ngram-spot` sweep on the
supported model set). Useful for ad-hoc experimentation, before/after
benchmark comparisons, and bench harness flags that don't want to thread
extra parameters through.

```bash
MLX_NGRAM_ENABLED=1 swift run my-app
```

Explicit Swift parameters always win over the env-var defaults: setting
`ngramSize: 5` in code and leaving `MLX_NGRAM_ENABLED=1` set yields
`ngramSize: 5, maxNgramDraftTokens: 4` (the cap takes the env-var
default since the caller didn't set it).

### 3. Benchmark harness flag

The bench harness exposes `--method ngram-sweep` (and `--method
ngram-spot` for single-prompt cell sweeps), which configures the
parameters internally. See `benchmarks/README.md` for usage.

```bash
./scripts/benchmark.sh --method ngram-sweep --model gemma4-26B-A4B
```

## Models supported

Tested models on M1 Max 64 GB. "Engages" means n-gram speculative decoding kicks in; "auto-fallback" means the route correctly disqualifies and runs plain `TokenIterator` (no regression vs. baseline).

| Model | Architecture | Status | Notes |
|---|---|---|---|
| Gemma 4 26B A4B 4-bit | MoE attention | ✅ Engages | 1.32× greedy / 1.31× Leviathan @ temp=0.6 on input-grounded prompts |
| Gemma 4 E2B 4-bit | Dense attention | ⚠️ Engages but slower | 2-9% regression on per-token bookkeeping floor (model runs at ~100 tok/s baseline; verify-batch overhead doesn't amortise) |
| GPT-OSS-20B mxfp4 | Dense attention (harmony format) | ⚠️ Engages but slower | 15-34% regression — ~70 tok/s baseline puts it in the same fast-forward regime as E2B despite higher param count. The 30-44% accept rate isn't enough to amortise per-token overhead at this throughput. |
| Qwen 3.5 0.8B 4-bit ★ | Hybrid GDN | 🔁 Auto-fallback | Falls back to `TokenIterator` cleanly; parity with TI baseline |
| Other pure-attention 4-bit (Llama 3.x, Phi, Qwen 3 dense) | Dense attention | ✅ Expected (untested in this PR) | Should behave like Gemma 4 dense; size determines whether it's a win or regression — bigger ≈ better, smaller ≈ marginal |
| Qwen 3.5 / 3.6 (any size > 0.8B) ★ | Hybrid GDN | 🔁 Auto-fallback | Same path as Qwen 3.5 0.8B |
| Nemotron-H ★ | Hybrid Mamba | 🔁 Auto-fallback | Untested but route-decision is identical |
| Jamba ★ | Hybrid Mamba | 🔁 Auto-fallback | Untested but route-decision is identical |

★ **GDN / Mamba hybrid models are not currently supported.** Their KV cache contains layers whose recurrent state can't be rolled back positionally on draft rejection. Spec 020's tape-replay rollback ([PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143)) is the planned generalisation — phase 1 (the protocol + dispatch helpers) landed in that PR; phases 2 + 3 need to land before n-gram can engage on these targets. Until then, the auto-route correctly disqualifies and runs plain `TokenIterator` for all hybrid targets.

## When does the n-gram path engage vs. fall back?

Auto-routing in
``MLXLMCommon/generate(input:cache:parameters:context:wiredMemoryTicket:)``
inspects each request's parameters + the target's cache shape and picks
one of two iterators:

- **`NGramSpeculativeTokenIterator`** when n-gram is opted in (any of
  the three paths above) **and** both compatibility conditions below
  hold.
- **`TokenIterator`** otherwise — the standard generation path. This is
  always-correct for any parameters; the only thing you "lose" by
  falling back is the speculative-decode speedup.

**Compatibility conditions for engaging n-gram:**

| Condition | If not satisfied |
|---|---|
| Target's KV cache is fully trimmable (i.e. all attention; no GatedDeltaNet / Mamba / SSM layers) | falls back to `TokenIterator`; same model, same output, just no speedup. See "Models supported" above for the GDN/Mamba follow-up. |

Penalties (`repetitionPenalty`, `presencePenalty`, `frequencyPenalty`)
and any `additionalProcessors` you set on `GenerateParameters` are
**applied to the target model's logits exactly the same way
`TokenIterator` would** — the n-gram iterator plumbs them through both
the verify forward and the AR fallback. They neither block n-gram from
engaging nor change behaviour vs. the non-speculative path.

### About the sampling-temperature path (Leviathan)

At `temperature == 0` the iterator runs the **greedy verify** path:
draft tokens are accepted when they match the target's argmax. Output
is byte-identical to plain `TokenIterator(temperature: 0)`.

At `temperature != 0` the iterator runs the **Leviathan accept/reject
sampling** path: each draft token is accepted with probability
`p_target(draft)` where `p_target` is the target's
temperature-scaled distribution. Output is *distributionally
equivalent* to plain `TokenIterator(temperature: T)` — samples from
the same distribution but not the same realisation as a baseline
sequential run. This is the spec-decode correctness guarantee from
Leviathan et al. (2023), specialised to n-gram drafts.

Leviathan is **default-on**; set `MLX_NGRAM_LEVIATHAN=0` to disable
(reverts to the pre-2026-04-29 behaviour where the route declined at
`temperature != 0` and fell back to `TokenIterator`). Disabling is
useful only as a diagnostic A/B; in production the default delivers
~1.31× speedup on the favourable regime (Gemma 4 26B A4B + recipe-
rewrite at temp=0.6) — within ~1% of greedy n-gram's 1.32× — and is
neutral or modest-regression elsewhere (same regime asymmetry as
greedy; see "When does it help?" below).

`top-p` / `top-k` / `min-p` samplers route through the same Leviathan
path; the residual sampling on rejection delegates to the configured
sampler so its truncation logic (top-p threshold, top-k cap, etc.)
applies natively to the residual distribution.

### About the cache condition

Hybrid SSM/Mamba models (Qwen 3.5/3.6 GatedDeltaNet, Nemotron-H, Jamba)
have layers whose KV state can't be rolled back position-by-position on
draft rejection. Spec 020's tape-replay rollback is the planned
generalization; until it lands, those targets fall back to
`TokenIterator` automatically.

### Throughput note: AR fallback with processors set

When a logit processor is active (any of the penalties, or
`additionalProcessors`), the iterator's **AR fallback** (the path it
takes when the n-gram lookup table doesn't have a useful match) runs
one decode step at a time instead of async-pipelining
`MLX_NGRAM_AR_BATCH` steps (default 4). This is because each step's
`didSample` updates the processor's token-ring state, and the next
step's logits must be conditioned on that update — a fundamental
chain dependency that can't be pipelined while preserving byte-
identicality.

The cost depends on the model:

| Model class | Baseline tok/s | AR-batch collapse cost | Relative impact |
|---|---|---|---|
| Big weight-bandwidth-bound (Gemma 4 26B A4B 4-bit) | ~24 | ~5 ms/tok | ~12% on AR-fallback positions |
| MoE flash decoder (Qwen 3.5-35B-A3B 4-bit) | ~140 | ~5 ms/tok | ~70% on AR-fallback positions |
| Small/fast (Gemma 4 E2B 4-bit) | ~100 | ~5 ms/tok | ~50% on AR-fallback positions |

The 5 ms/token figure is the asymptotic difference between async-batched
AR-4 and unbatched AR-1 on M1 Max — not "the cost of the processor",
which is itself ~0.1–0.5 ms/token (the gather + multiply for the
penalty calculation). **What you're paying for is the loss of
pipelining**, not the penalty math itself.

This cost is paid only on AR-fallback positions (when the lookup
misses or returns too-short a draft). On regurgitative workloads
where the lookup hits often, it's a small minority of cycles and the
n-gram speedup absorbs it. On paraphrastic workloads where the
lookup mostly misses, this is the dominant cost — combined with the
zero-accept verify overhead, it's why `repetitionPenalty` + n-gram +
paraphrastic content compounds into the 14% regression measured on
the lighthouse-keeper prompt in the close-out sweep.

## What `MLX_NGRAM_ENABLED` turns on

When the env-var path engages with caller defaults, the iterator runs
with the **good-defaults profile** validated by the spec-013 close-out
sweep on Gemma 4 26B A4B and Qwen 3 dense:

| Field | Function | Default | Env var override |
|---|---|---|---|
| `ngramSize` | Primary n-gram order — the length of the suffix matched in token history when proposing draft continuations. Higher values are stricter priors but match less often. | 3 | `MLX_NGRAM_ENV_DEFAULT_SIZE` *(future)* |
| `maxNgramDraftTokens` | Per-round cap on draft length. With adaptive scaling on, this is the upper bound the iterator floats up to as accept rate rises. | 4 | *(future)* |
| `ngramDraftMin` | Minimum draft length required to engage the verify-batch path. Shorter drafts fall through to the AR fallback because the verify overhead doesn't amortize on them. | 1 | — |
| `ngramMinHits` | Minimum number of times an n-gram pattern must appear in token history to be eligible as a draft source. Higher values trade recall for precision. | 1 | — |
| `minNgramSize` | Floor of the multi-size fallback ladder — the lookup walks `ngramSize, ngramSize-1, … minNgramSize` until it hits. Set equal to `ngramSize` to disable fallback. | 2 | — |
| Adaptive draft scaling | Scales the per-round draft cap in `[1, maxNgramDraftTokens]` based on rolling accept rate. Expands on regurgitative prompts, shrinks on paraphrastic ones so verify overhead never dominates. | **on** | `MLX_NGRAM_ADAPTIVE=0` to disable |
| Multi-candidate selection | When an n-gram has multiple prior occurrences, groups continuations by their first token and picks the most-frequent group (tiebreak: most-recent). Beats plain "most-recent" on long-context workloads. | **on** | `MLX_NGRAM_MULTI_CANDIDATE=0` |
| Strict-greedy guard | Stops the draft chain at any verify position where the top-1 vs top-2 logit margin is tight. Prevents batched-vs-sequential numerical drift from cascading into incorrect emissions. | **on** | `MLX_NGRAM_STRICT_GREEDY=0` |
| Dominance gate | Requires the winning candidate group to dominate (`max_count > 2 × sum_others`) before drafting; otherwise falls back to a shorter n-gram or AR. Trades recall for precision on ambiguous patterns. | off | `MLX_NGRAM_DOMINANCE=1` to enable |
| AR-fallback batch size | When the lookup misses, runs N async-pipelined AR steps before syncing. Larger values pipeline more aggressively but waste up to N-1 forwards if EOS lands mid-batch. | 4 | `MLX_NGRAM_AR_BATCH=N` |
| Debug tracing | When on, each speculation round logs a `[NGRAM]` line with draft length, accept count, KV cache offset, and the AR/verify branch taken. Off-state has zero overhead. | off | `MLX_NGRAM_DEBUG=1` |
| Force AR | Bypasses draft proposal entirely; every step goes through the AR fallback. Diagnostic only — isolates verify-path bugs from AR-path bugs. | off | `MLX_NGRAM_FORCE_AR=1` (diagnostic) |

The three "default ON" feature toggles are what give the n-gram path its
practical wins on mixed workloads. Adaptive scaling expands the draft
cap on regurgitative prompts (where most drafts hit) and shrinks it on
paraphrastic ones (where most miss), so the verify-batch overhead never
dominates. Multi-candidate selection picks the most-frequent
continuation when an n-gram has multiple prior occurrences, beating
plain "most-recent" tiebreak on long-context workloads. Strict-greedy
stops the chain at any verify position whose top-1 vs top-2 logit
margin is tight, preventing batched-vs-sequential numerical drift from
compounding across drafts.

## When does it help (and when does it hurt)?

### Quick decision table

This is the rolled-up summary across all the close-out benchmarks on
M1 Max 64 GB. Each row is a regime where we have multi-trial data
(see `benchmarks/gemma4-ngram-processor-plumbing-analysis.md` for full
trial sets).

| Regime | Iterator engaged | Measured | Recommendation |
|---|---|---|---|
| **Big weight-bandwidth-bound model + regurgitative workload** (Gemma 4 26B A4B + recipe-rewrite) | NGram greedy | **1.29× speedup**, 70% accept | ✅ Engage — n-gram is the right default |
| ... with `repetitionPenalty` 1.0 to 1.5 | NGram + processor | **1.32–1.39× over TI+penalty**, 68–70% accept | ✅ Engage — the plumbing fix unlocks the win |
| **Small/fast model** (Gemma 4 E2B 4-bit at ~100 tok/s) | NGram greedy | **0.98× (2% regression)** | ⚠️ Don't auto-engage — verify overhead doesn't amortize |
| ... with penalty | NGram + processor | **0.91× (9% regression)** | ⚠️ Avoid — penalty + small model compounds the loss |
| **Paraphrastic workload** (creative writing, opinion) | NGram greedy | **0.89× (11% regression)**, 0% accept | ⚠️ Don't engage — every verify cycle is wasted GPU work |
| ... with penalty | NGram + processor | **0.86× (14% regression)** | ⚠️ Avoid |
| **Hybrid GDN/Mamba target** (Qwen 3.5 family, Nemotron-H, Jamba) | Auto-fallback to TokenIterator | parity with TokenIterator (no engagement) | ✅ Hybrid-safe — disqualifier catches it cleanly |
| **Sampling baseline** (`temperature != 0`) | Auto-fallback to TokenIterator | parity with TokenIterator (no engagement) | ⏳ ~34% headroom for spec 023 (Leviathan accept/reject) |

### Per-workload class detail (Gemma 4 26B A4B, regurgitative-leaning prompts)

The headline numbers from the spec-013 close-out sweep:

| Workload class | Baseline tok/s | N-gram tok/s | Speedup | Accept rate |
|---|---|---|---|---|
| Code refactor (input-grounded) | 24.1 | 36.8 | 1.53× | 71% |
| Recipe bulk (templated) | 25.3 | 39.7 | 1.57× | 75% |
| Bug-report Q&A (re-quoting) | 24.7 | 35.4 | 1.43× | 65% |
| RFC writing (PM template) | 25.0 | 31.2 | 1.25× | 48% |
| Open-ended generation (haiku) | 24.5 | 23.9 | 0.97× | 12% |

The speedup tracks acceptance rate. On **input-grounded** workloads
where the model regurgitates from the prompt, accept rates run 50–80%
and the iterator wins comfortably. The "open-ended generation" row
above is a borderline case (12% accept, ~3% regression); a more
strongly paraphrastic workload like the lighthouse-keeper short-story
prompt in the close-out sweep produces 0% accept and an 11%
regression.

### Why paraphrastic content can regress

Two mechanisms compound on creative / open-ended prompts:

1. **The lookup table rarely hits.** Paraphrastic content has few
   repeated n-grams, so most cycles bypass speculation entirely and
   go straight to AR fallback. On a 200-token lighthouse-keeper
   short-story generation, the lookup proposed only 6 drafts total
   (vs. 107 on a 200-token recipe rewrite).
2. **When the lookup *does* hit, the strict-greedy guard rejects
   every draft.** Paraphrastic generation has tighter top-1 vs.
   top-2 logit margins (the model is uncertain about word choice),
   so the guard fires aggressively to prevent batched-vs-sequential
   numerical drift from compounding into wrong commitments. **Doing
   the right thing for correctness; killing speedup as a side effect.**

The verify-batch overhead at K=3 is pure GPU cost when 0% of drafts
accept. The adaptive draft scaler shrinks the cap toward 1, but even
at K=1 the verify forward processes 2 positions per cycle vs. 1 for
plain decode. **If your workload looks like this, leave
`MLX_NGRAM_ENABLED` unset for the relevant calls.**

### Why small/fast models can regress

Spec-decode wins are **weight-bandwidth-bound**: the verify forward's
K+1-position cost is approximately the same as a 1-position decode
because attention is bottlenecked on weight loads, not compute. This
amortization argument depends on the model being big enough that
weight-loading dominates. On Gemma 4 E2B 4-bit (~10 ms/token forward,
100 tok/s baseline), the per-token CPU bookkeeping (lookup table
maintenance, accept/reject loop, cache trim) becomes a meaningful
fraction of total time. Even at 70% accept rate, the iterator nets
out at 0.98× of plain `TokenIterator` throughput.

Rule of thumb from the data: if your target model runs at >80 tok/s
baseline on your hardware, n-gram is unlikely to win. The decision
isn't currently automated — see issue #153 for the discussion of
whether it should be.

## Output is byte-identical at `temperature: 0`

This is the load-bearing contract. For any workload, any prompt, and
any model where the route engages, the n-gram iterator emits the same
token stream as ``TokenIterator`` would at `temperature: 0`. The strict-
greedy guard preserves this even on tight-margin positions where
batched-vs-sequential numerical drift could theoretically cause the
verify pass's argmax to disagree with a hypothetical sequential
reference.

If you observe divergence in practice, it's a bug — please file an
issue with prompt + model + parameters.

## Diagnostics

Set `MLX_NGRAM_DEBUG=1` to print one `[NGRAM]` line per speculation
round showing draft length, accept count, AR-vs-verify branch, and
adaptive-scaling decisions. Useful for understanding why a particular
workload isn't getting the speedup you expected.

```text
[NGRAM] iterator engaged: ngramSize=3 maxDraft=4 draftMin=1 minHits=1 ...
[NGRAM] verify draft=4 accepted=3 rejected=1 emit=4 next_y=12345
[NGRAM] adaptive: rate=0.85 draft 4→6
[NGRAM] AR-batch draft=0 size=4 emit=[12, 34, 56, 78]
```

`MLX_NGRAM_FORCE_AR=1` bypasses the n-gram lookup entirely and runs the
iterator's AR-fallback path on every round — useful for isolating
verify-path bugs from AR-path bugs when investigating a regression.

## See also

- ``GenerateParameters/ngramSize``, ``GenerateParameters/maxNgramDraftTokens``,
  and the related fields for the full Swift-level configuration surface.
- ``NGramSpeculativeTokenIterator`` for direct iterator construction
  when you don't want to go through `generate(...)`.
- `papers/speculative-decoding-on-apple-silicon.md` in the repository
  root for the full design rationale, ablation results, and benchmark
  methodology.
- `specs/013-ngram-speculative-decoding.md` for the design spec.
- `benchmarks/gemma4-ngram-processor-plumbing-analysis.md` for the
  full multi-trial sweep data behind the regime table above (recipe
  vs. lighthouse-keeper paraphrastic, E2B vs. 26B-A4B, with full
  trial sets and md5-verified byte-identical-output checks).
- Issue #153 for the design discussion on whether to auto-disengage
  in the known regression regimes (small/fast models, paraphrastic
  workloads) vs. leave the decision to consumers.
- `Libraries/MLXLMCommon/SpeculativeDecoding.swift` →
  ``SpeculativeTokenIterator`` for the orthogonal **draft-model**
  speculative path. The two paths are complementary: n-gram needs no
  draft model; draft-model speculative needs no input-grounding.
