# 023 — Leviathan accept/reject sampling for n-gram speculative decoding

**Status:** spec, ready to turn into a stacked PR after #113 lands
**Branch:** new branch off `ek/ngram-speculative-v2` (post-merge: off `alpha`)
**Depends on:**
- PR [#113](https://github.com/ekryski/mlx-swift-lm/pull/113) — close-out of greedy n-gram + processor plumbing. The accept-loop refactor in this spec sits directly on top of the per-position verify path that landed there.
- Spec 013 — the n-gram iterator's existing surface (`NGramSpeculativeTokenIterator`, `NGramLookup`, route decision).

## Problem

`NGramSpeculativeTokenIterator` declines the route when
`parameters.temperature != 0`. The iterator's verify step accepts a
draft token only when it matches the target's argmax (greedy decision),
so any sampling regime — `temperature > 0`, `topP < 1`, `topK > 0`,
`minP > 0` — would diverge from the equivalent plain `TokenIterator`
run. The current escape hatch is to fall back to plain `TokenIterator`
on a per-call basis.

This is a sharp limitation in practice: most production calls use
non-zero temperature (chat assistants, creative writing, RAG with
diversity). They get **none** of n-gram's 1.3–1.6× speedup. Greedy
calls are the minority on real workloads (tool-calling, structured
output, deterministic codegen).

The fix is well-known in the literature: **Leviathan accept/reject
sampling** (Leviathan, Kalman, Matias 2023, "Fast Inference from
Transformers via Speculative Decoding," arXiv:2211.17192). For
draft-model speculative decoding, the algorithm guarantees the output
distribution is identical to `TokenIterator`'s sampling distribution —
provably, not just in expectation. For n-gram speculative decoding the
algorithm degenerates nicely because the "draft distribution" is
deterministic, simplifying the implementation.

## Algorithm

Leviathan's general formulation: given draft tokens `x_1..x_K` sampled
from draft distribution `q`, target distribution `p` at each position
(computed in one verify forward), and a uniform sample `u_i ~ U(0,1)`
per position:

```
For i in 1..K:
    if u_i ≤ p(x_i) / q(x_i):
        accept x_i; advance
    else:
        reject; sample replacement from (p - q)+ renormalized; stop chain

If all accepted:
    sample bonus token from p at position K+1
```

Output is provably distributed exactly as `p` — same statistical
behavior as a sequential run of `TokenIterator` at temperature `T` /
`topP` / etc.

### Specialization for n-gram drafts

N-gram drafts come from a deterministic CPU lookup, so the draft
distribution `q` is degenerate:

```
q(token) = 1 if token == lookup_proposed_token else 0
```

Substituting into Leviathan's accept condition:

```
u_i ≤ p(x_i) / q(x_i) = p(x_i) / 1 = p(x_i)
```

And the residual `(p - q)+` renormalized becomes "p with the proposed
token zeroed out, renormalized" — which is just sampling from `p`
excluding the rejected token.

The algorithm for n-gram speculative sampling is therefore:

```
For each draft token at position i (verify forward gives logits L_i):
    Apply processor → L'_i
    p_i = softmax(L'_i)
    Sample u_i ~ U(0, 1)
    if u_i < p_i[draft_i]:
        accept draft_i; advance
    else:
        # Sample replacement from residual:
        L_residual = L'_i with draft_i set to -inf
        replacement = sampler.sample(L_residual)
        emit replacement; stop chain

If all K accepted:
    bonus = sampler.sample(L'_{K+1})
    emit bonus
```

For non-greedy samplers (`TopPSampler`, `CategoricalSampler`),
"replacement = sampler.sample(L_residual)" must produce a token
distributed exactly as `p_i / (1 - p_i[draft_i])` over the non-draft
support. The existing samplers compose with this if we mask the
rejected token to `-inf` before sampling — which is what
`TopPSampler` / `CategoricalSampler` already does for sampling
truncation, just with a single index instead of a distribution-shape
truncation.

**Greedy degenerate case (temperature = 0):** at temperature 0, the
sampler is `ArgMaxSampler`, and `p_i` is a one-hot at the argmax. The
accept condition becomes `u_i < 1[draft_i == argmax_i]`, which is
exactly the existing greedy compare. So the new code path subsumes the
old one — we can ship just the Leviathan path and remove the dual
implementation. (Worth doing in phase 2 once the new path is validated;
phase 1 keeps both for diff/regression sanity.)

## Design

### 1. Iterator changes

`NGramSpeculativeTokenIterator.speculateRound` becomes mode-aware on
`parameters.temperature`:

- `temperature == 0`: existing greedy path (argmax compare, strict-
  greedy guard, AR fallback). Unchanged.
- `temperature > 0` (or `topP/topK/minP` set): new Leviathan path,
  per-position accept/reject with random sampling.

The existing per-position verify loop (already the path used when a
processor is set, since #113's plumbing) is the right hook. The
forward pass is identical; only the accept rule changes.

### 2. Residual sampling helper

New free function in `NgramSpeculativeDecoding.swift`:

```swift
/// Sample a replacement token from the target distribution at this
/// verify position, with the rejected draft token excluded.
/// Implements Leviathan's `(p - q)+` step for the n-gram case where
/// q is degenerate at `rejectedToken`.
internal func sampleResidual(
    logits: MLXArray,
    rejectedToken: Int,
    sampler: LogitSampler
) -> MLXArray
```

Implementation: clone logits, set `logits[rejectedToken] = -.infinity`,
delegate to `sampler.sample(logits:)`. The samplers handle the
truncated distribution natively.

### 3. Random-number source

Each verify position needs a uniform `u_i ∈ [0, 1)`. Two options:

- **MLX side**: `MLXRandom.uniform([numDraft+1])` using the global key.
- **Swift side**: `SystemRandomNumberGenerator` via `Float.random(in:)`.

MLX side keeps everything on-device, but ties the iterator to MLX's
random-key state — which the tests would have to mock for reproducibility.
Swift side is simpler (`Float.random(in: 0..<1)`); we bring `u_i` to
the GPU only if needed for batch processing (we're already per-position
sequential, so we don't need batching).

**Recommendation**: Swift side, `SystemRandomNumberGenerator`, seedable
via `init(rng: any RandomNumberGenerator = SystemRandomNumberGenerator())`
for tests that need determinism.

### 4. Probability extraction

We need `p_i[draft_i]` per position. After applying the processor:

```swift
var logits = mainLogits[0..., verifyStart + i, 0...]
logits = verifyProcessor.process(logits: logits)
let logProbs = MLX.logSoftmax(logits, axis: -1)
let logP = logProbs[0..., draftIdx]  // shape [1]
let p = MLX.exp(logP)
eval(p)
let pValue = p.item(Float.self)
```

`p_i[draft_i]` is a single index lookup so the per-position cost is
modest. The strict-greedy margin computation (already paid on the
processor-active path) costs comparable GPU work.

### 5. Strict-greedy guard semantics

The strict-greedy guard (`MLX_NGRAM_STRICT_GREEDY=1`) is a greedy-only
concept — it's about argmax stability under batched-vs-sequential
numerical drift. With sampling there's no analogous "drift" because
we're using actual probabilities, not picking a winner.

Phase 1 ships: strict-greedy guard ignored when `temperature != 0`.
Document this in the doc comment.

### 6. Routing decision

`ngramRouteDecision(parameters:)` (introduced in #113) currently
disqualifies on `temperature != 0`. After this spec lands, the gate
relaxes:

- Old: `temperature != 0` → fall back.
- New: any `temperature`, any sampler — engage. The iterator picks the
  greedy or Leviathan path internally.

The hybrid-cache disqualifier (cache trimmability) stays — that's
orthogonal.

### 7. Auto-disable for the small-model regime

The pre-spec analysis (commit `aac716c` doc + chat thread) flagged
that with processors active on small/fast models (Gemma 4 E2B 4-bit,
Qwen 3.5 35B-A3B 4-bit), the AR-fallback's loss of async pipelining
costs ~5 ms/token, or up to 50% of decode time. Sampling with
Leviathan inherits this cost (per-position sequential is necessary).

Optional phase-2 mitigation: an `MLX_NGRAM_MIN_ACCEPT_RATE=0.X`
env-var-controlled self-disengage. When the rolling accept rate falls
below threshold for N consecutive rounds, switch to plain
`TokenIterator` for the rest of the request. Already partially handled
by adaptive draft scaling (caps draft to 1) but doesn't kick the
verify-batch overhead. Phase-2 tracking item.

## Implementation phases

### Phase 1 — Leviathan path on the existing iterator

- Add `sampleResidual` helper.
- Add Leviathan path in `speculateRound` gated on
  `parameters.temperature > 0 || topP < 1 || topK > 0 || minP > 0`.
- Reuse the per-position verify loop (already there for processor
  plumbing).
- Relax the route-decision temperature disqualifier.
- Update doc comments + `Documentation.docc/speculative-decoding.md`.

Estimated diff: ~150 LOC code, ~80 LOC docs. Roughly the same shape as
the processor-plumbing fix in commit `880b416`.

**Status: shipped in PR #154.** Includes a phase-2 follow-up
(batched p-value computation: collapses N+1 evals → 2 evals per
cycle) that eliminated the bulk of the per-position-eval overhead and
brought the favourable-regime speedup from 1.18× to 1.31× — within
~1% of greedy n-gram's 1.32×. **Default flipped to ON (`MLX_NGRAM_LEVIATHAN`
defaults to enabled; set to `0` to disable)** after the broad sweep
in `benchmarks/gemma4-leviathan-broad-sweep-analysis.md` validated the
mechanism. Callers who opted into n-gram (via Swift parameters or
`MLX_NGRAM_ENABLED=1`) and use sampling now automatically get
Leviathan; the regression regimes that affect both paths are tracked
in issue #153.

### Phase 2 — Unify with greedy path

**Status: deferred / not pursuing.** When PR #154 was reviewed, the
"~50 LOC cleanup" framing turned out to underestimate the surface area:

- **The greedy batch-sample-no-processor path is a real optimisation.**
  When no processor is set, the greedy path runs ONE batched
  `sampler.sample(verifyLogits)` over the entire `[numDraft+1, V]`
  tensor in a single eval. Leviathan-style per-position sampling at
  `temperature=0` would lose that — even with batched p-value
  extraction, we'd add unnecessary softmax + comparison work to the
  temp=0 path that pure argmax doesn't need.
- **The strict-greedy guard has no Leviathan analog.** It exists to
  prevent batched-vs-sequential numerical drift from compounding wrong
  commitments at temp=0 — a greedy-specific concern. Leviathan
  compares against actual probabilities, so a tight margin naturally
  accepts ~50%. Removing strict-greedy means changing observable
  behaviour on real models (we shipped it default-on after seeing real
  failures on Gemma 4 26B A4B summarisation in PR #113).
- **At `temperature == 0`, `softmax(logits / 0)` is undefined.** Needs
  a special case anyway, so the unification isn't pure.

Net assessment: closer to ~30 LOC reduction with material
correctness/perf risk. We keep greedy and Leviathan as separate
paths.

### Phase 3 — Self-disengage on chronically-low accept

**Status: subsumed by issue #153.** Issue #153 captures the same
heuristic (`MLX_NGRAM_MIN_ACCEPT_RATE` + rolling-window threshold) as
a discussion item that should apply uniformly to *both* the greedy
and Leviathan paths. The decision tree (always-on heuristic vs.
opt-in parameter vs. doc-only) is the same in both cases. Whatever
form of auto-disengagement we ship for greedy n-gram should apply
identically to Leviathan; tracking in one place avoids divergence.

## Testing

The output guarantee is **distributional equivalence**, not byte
identity — `TokenIterator(temperature: T)` and the new path produce
samples from the same distribution `p` but not the same realization.
Testing approaches:

### Unit tests (pure-Swift, no MLX)

- `sampleResidual` correctness: feed a known logit vector, verify the
  rejected index has zero mass and the renormalized distribution
  matches the analytic answer.
- `n-gram accept/reject probability computation`: feed scripted logits
  + known draft + known `u`, verify the accept/reject decision matches
  the formula.

### Integration tests (real model)

- Statistical equivalence on a small model. Run `TokenIterator(temp:
  0.6)` vs Leviathan-iterator-at-`temp: 0.6` for N=200 samples each on
  the same prompt. Compare token-frequency histograms via χ² or
  Kullback-Leibler divergence; expect KL → 0 as N → ∞. This needs a
  real-model test harness we don't have today — defer to phase 1B.
- **Cheap sanity check**: at `temperature: 0`, the new Leviathan path
  must produce byte-identical output to the existing greedy path. Tests
  the degenerate-case correctness without needing statistical
  verification.

### Benchmark validation

Run the `ngram-spot` harness (PR #140) with new axes:

- `temperature ∈ {0, 0.6}`
- `repetitionPenalty ∈ {1.0, 1.1}`
- Across the supported model set (Gemma 4 E2B / 26B A4B, Qwen 3 dense)

Measure: tok/s, accept rate, AR-fallback fraction. Compare:
- Plain `TokenIterator` at each (temp, penalty) cell.
- Leviathan-iterator at each cell.

Expected outcomes:
- `temp=0, penalty=1.0`: same as today — no regression vs. current
  greedy path.
- `temp=0.6, penalty=1.0`: meaningful speedup (1.2–1.4× on
  input-grounded workloads). This is the headline win.
- `temp=*, penalty>1.0`: lower accept rate (penalty ↔ history
  conflict), small-model AR cost becomes visible.

## Risks

1. **Distributional-equivalence verification.** Hard to test per-sample.
   Mitigation: the degenerate greedy case (temp=0) is byte-identical
   and we can verify it cheaply. For sampled cases, multi-sample bench
   validation is the answer; defer to phase 1B with the real-model
   harness.

2. **Numerical precision around small `p`.** `p_i[draft_i]` can be very
   small for non-favored tokens; `u < p` decisions near the
   floating-point floor could flip on different hardware. Mitigation:
   compute `log p` and compare `log u < log p` (log-space avoids
   underflow). MLXOps has `logSoftmax` / `log` / `exp` natively.

3. **Sampler composition.** `TopPSampler` truncates the distribution
   before sampling. `sampleResidual` further sets a single index to
   `-inf`. The two truncations interact — TopP's mass-cumulative
   threshold is computed on the original distribution, so masking
   afterward could leave the sampler with too few candidates. Need to
   ensure the masking happens *inside* the sampler's own truncation
   path. Probably fine since TopP normalizes after truncation; verify
   in the unit test pass.

4. **Phase-1 strict-greedy interaction.** Doc says "strict-greedy
   ignored at `temperature > 0`". Need to make sure the env-var-driven
   guard doesn't accidentally fire on the Leviathan path; check at
   route entry.

5. **No real-model tests.** Phase 1's unit tests only validate the
   degenerate greedy case + the helper math. The real correctness
   contract (sample distributions match) needs a multi-sample real-model
   regression harness that doesn't exist today. We can ship phase 1 with
   the cheap tests and a manual validation note; spec 013-style
   close-out demands the real-model harness, deferred to phase 1B
   alongside the n-gram greedy-equivalence regression target.

## Files touched

| File | What | Lines (est.) |
|---|---|---|
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | Leviathan path in `speculateRound`, `sampleResidual` helper, route-decision update | ~150 |
| `Libraries/MLXLMCommon/Documentation.docc/speculative-decoding.md` | Drop the temperature disqualifier wording, document the sampling regime | ~80 |
| `Tests/MLXLMTests/NGramSpeculativeTests.swift` | Unit tests for residual + accept formula; integration test for greedy-degenerate equivalence | ~120 |
| `specs/IMPLEMENTATION-PLAN.md` | Add Leviathan as a tracked follow-up | ~20 |

Total ~370 lines changed.

## Open questions

1. **Should the relaxation of the temperature disqualifier be opt-in via
   env var initially** (e.g. `MLX_NGRAM_LEVIATHAN=1`) so we can ship
   phase 1 without immediately routing all sampling calls through it?
   Pros: lets the field-test the path on real workloads before flipping
   the default. Cons: extra knob, eventual cleanup PR. Probably worth
   it for one release cycle.

2. **Does `topP < 1` / `topK > 0` / `minP > 0` count as "non-greedy"
   even at `temperature: 0`?** At `temperature: 0` with `topK: 1` the
   sampler is still argmax-equivalent. Worth checking carefully when
   we generalize the route decision.

3. **Strict-greedy + Leviathan**: should a sampling-mode strict-greedy
   analog exist? E.g. "reject the chain if the probability margin
   between top-1 and top-2 is tight" — preserves `p` but trades a
   little speedup for less drift compounding under quantization.
   Probably overkill for phase 1; revisit if we see drift in practice.

## References

- Leviathan, Kalman, Matias 2023. ["Fast Inference from Transformers
  via Speculative Decoding"](https://arxiv.org/abs/2211.17192). The
  foundational paper. §3.1 has the algorithm; Theorem 1 has the
  distributional-equivalence proof.
- Chen et al. 2023. ["Accelerating Large Language Model Decoding with
  Speculative Sampling"](https://arxiv.org/abs/2302.01318). Concurrent
  derivation; same algorithm, slightly different notation.
- llama.cpp's `common/sampling.cpp::common_sampler_sample` and
  `examples/lookup/lookup.cpp` — the reference implementation we'd
  benchmark against. Their greedy n-gram is the same shape as ours;
  their sampling extension is what we'd be porting.
- The existing `SpeculativeTokenIterator` in `Evaluate.swift:1383+`
  — uses Leviathan for the draft-model case. The shape we're porting
  to n-gram is a strict simplification of that.
