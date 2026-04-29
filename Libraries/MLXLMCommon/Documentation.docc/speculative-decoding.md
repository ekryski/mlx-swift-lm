# Speculative Decoding

`MLXLMCommon` ships a prompt-lookup speculative decoder
(``NGramSpeculativeTokenIterator``) that drafts continuation tokens from
the prompt and accepted-output history, then verifies them in a single
batched forward pass on **the same target model you'd use otherwise**.
On input-grounded workloads (code, templates, factual re-quoting, RAG)
this produces a 1.3–1.6× decode speedup over ``TokenIterator`` while
emitting **byte-identical** output at `temperature: 0`.

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
| `parameters.temperature == 0` (greedy) | falls back to `TokenIterator` *for that call*; sampling still works |
| Target's KV cache is fully trimmable (i.e. all attention; no GatedDeltaNet / Mamba / SSM layers) | falls back to `TokenIterator`; same model, same output, just no speedup |

Penalties (`repetitionPenalty`, `presencePenalty`, `frequencyPenalty`)
and any `additionalProcessors` you set on `GenerateParameters` are
**applied to the target model's logits exactly the same way
`TokenIterator` would** — the n-gram iterator plumbs them through both
the verify forward and the AR fallback. They neither block n-gram from
engaging nor change behavior vs. the non-speculative path.

### About the temperature condition

Speculative decoding's verify step compares the draft against the
target's argmax (greedy). When `temperature != 0`, the target uses a
stochastic sampler and "argmax-matches-draft" no longer captures
"sampling-would-have-produced-this-token." Lifting the limit needs
Leviathan-style accept/reject sampling, which is a separate workstream
tracked as a follow-up.

This is a per-call decision: `temperature: 0` calls in your app engage
n-gram (when opted in), `temperature > 0` calls run plain
`TokenIterator`. You don't have to make a global choice. A typical
pattern is `temperature: 0` for tool-calling / structured output / code
generation (where n-gram helps the most) and `temperature > 0` for
creative writing.

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
`didSample` introduces an in-band CPU↔GPU dependency. The cost is
<5 ms/token on M1 Max — small relative to the work the processor
itself does, but a measurable difference vs. running with no processor
on a workload that's mostly hitting the AR fallback.

## What `MLX_NGRAM_ENABLED` turns on

When the env-var path engages with caller defaults, the iterator runs
with the **good-defaults profile** validated by the spec-013 close-out
sweep on Gemma 4 26B A4B and Qwen 3 dense:

| Field | Default | Env var override |
|---|---|---|
| `ngramSize` | 3 | `MLX_NGRAM_ENV_DEFAULT_SIZE` *(future)* |
| `maxNgramDraftTokens` | 4 | *(future)* |
| `ngramDraftMin` | 1 | — |
| `ngramMinHits` | 1 | — |
| `minNgramSize` | 2 | — |
| Adaptive draft scaling | **on** | `MLX_NGRAM_ADAPTIVE=0` to disable |
| Multi-candidate selection | **on** | `MLX_NGRAM_MULTI_CANDIDATE=0` |
| Strict-greedy guard | **on** | `MLX_NGRAM_STRICT_GREEDY=0` |
| Dominance gate | off | `MLX_NGRAM_DOMINANCE=1` to enable |
| AR-fallback batch size | 4 | `MLX_NGRAM_AR_BATCH=N` |
| Debug tracing | off | `MLX_NGRAM_DEBUG=1` |
| Force AR | off | `MLX_NGRAM_FORCE_AR=1` (diagnostic) |

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

## When does it help?

The headline numbers from the spec-013 close-out sweep on Gemma 4 26B
A4B (M1 Max 64 GB):

| Workload class | Baseline tok/s | N-gram tok/s | Speedup | Accept rate |
|---|---|---|---|---|
| Code refactor (input-grounded) | 24.1 | 36.8 | 1.53× | 71% |
| Recipe bulk (templated) | 25.3 | 39.7 | 1.57× | 75% |
| RFC writing (PM template) | 25.0 | 31.2 | 1.25× | 48% |
| Open-ended generation (haiku) | 24.5 | 23.9 | 0.97× | 12% |
| Bug-report Q&A (re-quoting) | 24.7 | 35.4 | 1.43× | 65% |

The speedup tracks acceptance rate. On **input-grounded** workloads
where the model regurgitates from the prompt, accept rates run 50-80%
and the iterator wins comfortably. On **paraphrastic / open-ended**
workloads where the model rarely emits an exact prompt span, accept
rates are low (<20%) and adaptive scaling shrinks the draft cap toward
1; the iterator costs at most a few percent over baseline (the
single-token verify-batch overhead).

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
- `Libraries/MLXLMCommon/SpeculativeDecoding.swift` →
  ``SpeculativeTokenIterator`` for the orthogonal **draft-model**
  speculative path. The two paths are complementary: n-gram needs no
  draft model; draft-model speculative needs no input-grounding.
