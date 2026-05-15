# 030 — Multi-token prediction (MTP) support

- **Status:** spec
- **Branch:** new feature branch (probably `ek/mtp-self-speculative`)
- **Depends on:** Tier 1 #5 ([spec 020](020-tape-replay-rollback-generalised.md)) for hybrid GDN target rollback. Otherwise standalone.
- **Related:** [spec 015](015-dflash-diffusion-speculative-decoding.md) (companion-draft path — separate workstream), [spec 013](013-ngram-speculative-decoding.md) (n-gram fallback), [spec 023](023-leviathan-accept-reject-sampling.md) (accept/reject sampler — re-used here).

## Problem and goal

Several model families now ship with **multi-token prediction (MTP) heads** baked into the
same checkpoint as the base model. At decode time these heads predict the next
`k` tokens (`k` typically 1–4) in a single forward pass; if the target's own
argmax/sample agrees, the speculative tokens are accepted. This is
**self-speculative**: the same model is both target and draft, so there is no
separate draft model to load, no draft KV cache, no tokenizer-mismatch risk, and
no separate model registry.

Headline numbers from the reference implementations (CPU/GPU mlx-lm + MTPLX):

| Model | Mode | Speedup vs AR baseline |
|---|---|---|
| DeepSeek-V3-4bit (mlx) | native MTP, k=1 | 1.4–1.8× |
| Qwen3-Next-80B-A3B | native MTP (mtplx), k=2 | 2.24× |
| MiMo-7B-RL | native MTP, k=2 | 1.9× |
| GLM-4.5-Air | native MTP via `mtp1` | 1.4–1.6× |
| Gemma-4-26B-A4B + assistant | EAGLE-style assistant draft | up to 2× (at temp 0) |
| Gemma-4-31B + assistant | EAGLE-style assistant draft | up to 2× |

Today, mlx-swift-lm strips MTP/nextn weights at sanitize time on every model
that ships them. Confirmed sites:

| Model | Sanitize file | Strip rule |
|---|---|---|
| DeepSeek-V4 | [DeepseekV4.swift:902-912](../Libraries/MLXLLM/Models/DeepseekV4.swift#L902) | drops `model.layers.{numHiddenLayers}.*` and `mtp.*` |
| Qwen3.5 | [Qwen35.swift:949-955](../Libraries/MLXLLM/Models/Qwen35.swift#L949) | filters `mtp.` substring |
| Qwen3-Next | [Qwen3Next.swift:734-737](../Libraries/MLXLLM/Models/Qwen3Next.swift#L734) | nils every `mtp.` key |
| MiMoV2-Flash | [MiMoV2Flash.swift:456](../Libraries/MLXLLM/Models/MiMoV2Flash.swift#L456) | drops `model.mtp` prefix |
| MiMo / GLM4-MoE-Lite | [MiMo.swift:229](../Libraries/MLXLLM/Models/MiMo.swift#L229) / [GLM4MOELite.swift:665](../Libraries/MLXLLM/Models/GLM4MOELite.swift#L665) | parses `num_nextn_predict_layers` but ignores nextn weights at runtime |

This spec is about *not* dropping them, wiring them into a runnable MTP
forward, and exposing a self-speculative iterator.

## On the "MLX bundles have MTP heads pruned out" claim

Partially correct, with an important nuance. Field investigation summary
(via HF model cards + safetensors index inspection):

| Family | Original (HF main) ships MTP keys | mlx-community ships MTP keys | Notes |
|---|---|---|---|
| DeepSeek-V3 / V4 | **Yes** — `model.layers.{N}.*`, `eh_proj`, `enorm`, `hnorm`, shared `embed_tokens`/`shared_head` | **No** | `mlx_lm.convert` runs `tree_flatten(model.parameters())` — anything not bound to a Python `Module` is silently dropped |
| Qwen3-Next-80B-A3B | **No** in base bundle | **No** | drafts are intended to ship via separate `*-DFlash` repos |
| MiMo-7B / MiMoV2-Flash | **Yes** in original (`num_nextn_predict_layers > 0`) | **No** | same convert-side pruning |
| GLM-4.5-Air / GLM-4-MoE-Lite | **No** in main bundle | **No** | upstream points at vLLM's `glm4_moe_mtp.py` — MTP shipped only in vLLM-specific bundles |
| Gemma 4 (E2B / E4B / 26B-A4B / 31B) | **No** in base | **No** in base; **Yes** in `-assistant-bf16` mirrors | Google's MTP for Gemma-4 is **EAGLE-style** — a small companion draft transformer, not in-trunk nextn heads |

So the user's hunch is sharply correct for DeepSeek-V3/V4 and MiMo (where
the originals ship integrated MTP weights and our convert path drops them
into the void). For Qwen3-Next / GLM / Gemma-4, the originals don't ship
in-trunk MTP heads at all — those families publish drafts as separate
companion repos.

There is no `--keep-mtp` flag in `mlx_lm.convert`. The implicit pruning
happens because `mlx_lm/models/deepseek_v3.py` (and the others) define no
MTP submodule, so loading with `strict=False` silently drops layer-`N`
weights; conversion then saves only what the model object exposed. Three
remediation paths, ordered by effort:

1. **Add an MTP submodule to the Python model class** (e.g. `mlx_lm/models/deepseek_v3.py`
   gains `DeepseekV3MTP` mapping `model.layers.{N}.*` + `eh_proj/enorm/hnorm`),
   then re-run `mlx_lm.convert` against the original — **but** that PR
   lives in `ml-explore/mlx-lm`, not here. We can land it as an upstream
   contribution but should not block on it.
2. **Custom converter** (`scripts/mtp_convert.py`) that bypasses model-class
   binding: read the original safetensors index, copy MTP-namespaced
   tensors verbatim into a sibling `mtp.safetensors` file, quantise with
   the same scheme as the rest of the bundle, write a side-car
   `model.safetensors.index.json`. Mirrors how the Gemma-4 assistant flow
   ships drafts. **Recommended path 1** — full control, no upstream PR
   dependency.
3. **Mirror the assistant/DFlash drafter as a standalone MLX repo** —
   already exists for Gemma-4 (`mlx-community/gemma-4-*-it-assistant-bf16`).
   For our purposes we need to (a) stop stripping at our sanitize boundary
   for in-trunk nextn (DeepSeek, Qwen3.5, MiMo) and (b) load the
   Gemma-4 assistant repos as companion drafts.

This spec assumes **path 2** for in-trunk families and **path 3** for
companion-draft families. Both end up loadable through the same iterator.

## Taxonomy

There are two architecturally distinct "MTP" patterns in the wild and we
need to support both:

| Variant | Where the extra heads live | Examples | Decode shape |
|---|---|---|---|
| **A. In-trunk MTP heads** (a.k.a. "native MTP" / "nextn") | Stacked on top of the *same* base trunk; share embedding + LM head; predict tokens at offsets `+1..+k` from the same hidden state | DeepSeek-V3/V4, Qwen3-Next-80B (when `num_nextn_predict_layers > 0`), Qwen3.5/3.6 hybrid, MiMo, MiMoV2-Flash, GLM-4-MoE | Single forward emits `(k+1)` token logits; greedy or accept/reject vs. AR re-verify |
| **B. Companion assistant draft** (a.k.a. "EAGLE-style" / Google's "MTP" branding) | Small standalone transformer, separate weights repo, shares embedding + LM head with target | google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant | Standard speculative — draft proposes K, target verifies in one batched pass |

Despite Google calling variant B "MTP" on their model cards, the
mechanics are EAGLE-3-style assistant decoding, not the
DeepSeek-style in-trunk head. They share the iterator surface (target
verifies a block, accept by argmax/RA-sampling, trim on rejection) but
differ in:
- weight loading (single bundle vs. two bundles),
- KV cache topology (single cache vs. main + draft cache; the draft cache
  is autoregressive and tiny),
- failure modes (in-trunk variant cannot have tokenizer mismatch; companion
  variant must validate vocab match at registration).

Variant B is functionally close to spec 015 (DFlash) without the
block-diffusion draft and without target-hidden-state cross-attention. It
sits between the existing `SpeculativeTokenIterator` (autoregressive draft)
and DFlash. The cleanest implementation is to **route variant B through
the existing `SpeculativeTokenIterator`** with a small wrapper that
constructs the assistant-draft model from a registry, and to add a
**dedicated `MTPSelfSpeculativeTokenIterator`** for variant A.

This spec covers both. Phases 1–4 cover variant A (in-trunk); phases 5–6
cover variant B (assistant-draft). Phase 7 is sampling-correctness
generalisation that benefits both.

## Architecture overview — variant A (in-trunk MTP)

The reference implementation we're adopting is
[`youssofal/MTPLX`](https://github.com/youssofal/MTPLX). MTPLX is the
authoritative MLX-Python library for self-speculative decoding via
in-trunk MTP heads.

The decode loop per cycle:

1. **AR forward** on the last committed token. Returns `(logits, hiddenState)`
   from the trunk. Caller samples the bonus token from `logits` (this is
   "free" — same forward we'd run in plain AR).
2. **MTP draft** runs the `k` MTP heads on the same `hiddenState`,
   producing `k` proposed tokens (with an MTP-side KV update).
3. **Verify** runs the trunk on the `k` proposed tokens (batched
   `[1, k]` forward) to get the target's own next-token logits at each
   position.
4. **Accept** — for each position `i`, accept if `argmax(target_logits[i]) == proposed[i]`
   (greedy mode) or via accept/reject sampling à la spec 023 (stochastic
   mode). Stop at first rejection; emit the bonus from position
   `(accepted_count - 1)` if any.
5. **Trim** — main cache trims to committed length; MTP cache trims
   accordingly. Cycle repeats with new bonus token as the next AR input.

### MTP-head module shape

In-trunk MTP heads share the embedding and LM head with the trunk, plus
per-head: an embed-norm (`enorm`), a hidden-norm (`hnorm`), an embedding
projection (`eh_proj`), one transformer block (attention + MLP) and a
norm. Forward pass for head `i`:

```
h_i = trunk_hidden                                # from step 1
e_i = embed(prev_token_i)                         # prev_token_i = previous proposed token (or AR token for i=0)
h_i = eh_proj(concat([norm_h(h_i), norm_e(e_i)])) # concat order varies by family
h_i = block(h_i, mtp_cache_i)                     # block_i has its own KV
logits_i = lm_head(final_norm(h_i))               # shared LM head
```

Two model-family-specific axes constrain the implementation:

- **`hidden_variant`** — `pre_norm` (DeepSeek-V3) or `post_norm` (Qwen3-Next)
- **`concat_order`** — `embedding_hidden` (DeepSeek-V3) or `hidden_embedding` (Qwen3-Next)

MTPLX abstracts these into an `MTPContract` value type. We will mirror it.

### MTP cache topology

Each MTP head has its own KV cache because the head is its own
transformer block. The MTP caches are populated at *verify* time (each
verify forward writes K/V at every head). On rejection of the
`(accepted+1)`th proposal, the MTP caches' last `(k - accepted)` entries
are trimmed. This is symmetric to the trunk cache trim and uses the same
`canTrimPromptCache` invariant ([Evaluate.swift:1429](../Libraries/MLXLMCommon/Evaluate.swift#L1429)).

For hybrid GDN models (Qwen3.5/3.6, Qwen3-Next), the trunk cache trim
needs spec 020's tape-replay rollback. Variant A on those models is
**hard-blocked on spec 020 phase 2/3** — same hard dependency as DFlash on
hybrid models.

## Architecture overview — variant B (companion assistant draft)

Variant B is structurally a thinner wrapper on top of the existing
`SpeculativeTokenIterator`:

- The assistant draft is a 78M–500M-parameter transformer with its own
  weights bundle and config (`mlx-community/gemma-4-*-it-assistant-bf16`).
- It shares the target's tokenizer and (per the model card) produces
  byte-identical output at temp=0 for `--draft-block-size 6`.
- The draft is plain autoregressive — no block-diffusion, no
  hidden-state cross-attention.

The minimal path is therefore to **register a `(targetID → draftID)` map**
and let `SpeculativeTokenIterator` do the work, with a small extension
that:
1. Auto-loads the assistant draft from `mlx-community` based on a registry.
2. Validates vocab parity at registration (assistant drafts must match
   target tokenizer; refuse with a clear error otherwise).
3. Bumps the default `numDraftTokens` per the model card (Gemma-4 single =
   6, batched = 3).
4. Falls back gracefully when no assistant draft exists for the target.

This is a smaller workstream than variant A and can ship independently.

## Implementation phases

### Phase 1 — Bootstrap: MTP loader + DeepSeek-V4 forward (in-trunk)

Goal: load DeepSeek-V4 with the MTP head **kept** (not stripped at sanitize),
prove a single MTP forward produces the right logits.

- Stop dropping `mtp.*` keys at sanitize on DeepSeek-V4. Add
  `DeepseekV4MTP` module behind a feature flag (env: `MLX_MTP=1`,
  parameter: `GenerateParameters.mtpEnabled`).
- Implement `MTPInjector` — protocol that surfaces `(numHeads, contract,
  weightsKeyPrefix, makeCache)` for a given target.
- Add `MTPLoader` in `Libraries/MLXLMCommon/MTPLoader.swift`. Scans target
  weights map for `MTP_KEY_PREFIXES = ["mtp.", "language_model.mtp.",
  "model.mtp.layers.", "model.layers.{N}."]`. Out-of-band file support
  (`mtp.safetensors`, `mtp/weights.safetensors`, `model-mtp.safetensors`)
  for users who run our custom-converter path.
- Round-trip test: AR `[t0, t1, t2, ..., t_n]` → MTP forward `[t_n]` →
  compare emitted logits against MTPLX's reference output for the same
  weights/seed. Token-by-token equality at temp=0.

### Phase 2 — Self-speculative iterator (greedy, full-attention only)

- New `MTPSelfSpeculativeTokenIterator: TokenIteratorProtocol` in
  `Libraries/MLXLMCommon/MTPSelfSpeculativeDecoding.swift`. Mirrors
  `SpeculativeTokenIterator` ([Evaluate.swift:1347](../Libraries/MLXLMCommon/Evaluate.swift#L1347))
  but uses one model and an `MTPInjector` for the draft side.
- Per cycle: AR forward → bonus sample → MTP forward (`k` proposals) →
  verify forward (`[1, k]` shape) → greedy match → trim.
- Accept/reject only `numDraftAccepted` positions. Same metrics surface
  as the existing speculative iterator (`specDecodeProposed/Accepted`).
- Wire into `MLXLMCommon.generate(...)` auto-routing predicate after
  n-gram and DFlash, before plain AR. Routing decision: `mtp-eligible`
  when target carries an `MTPInjector` conformance and the iterator was
  loaded with MTP weights present.
- DeepSeek-V4 first (full-attention layers). Qwen3.5/3.6 deferred to
  phase 4 (waits on spec 020).

### Phase 3 — Stochastic accept/reject (variant A)

- Lift `temperature == 0` requirement using probability-ratio sampling
  (the same algorithm spec 023 already uses for n-gram). MTPLX's
  `mtplx/sampling.py` is the reference: top-p applied **before** top-k,
  `acceptance_probability = min(1, p/q)`,
  `residual = renormalise(max(p - q, 0))`.
- This is a copy-port of the spec-023 path with one change — the draft
  distribution `q` here is the MTP head's logits (not the target's
  argmax), so `q` is a real distribution rather than a one-hot.
- Marginal-correctness oracle test (mirrors MTPLX's
  `speculative_output_marginal`): over 1e5 samples, the iterator's
  empirical token distribution must equal the target's distribution
  within 3σ. Catches sampler bugs that greedy mode can't.

### Phase 4 — Hybrid GDN target (Qwen3.5 / 3.6, Qwen3-Next)

Hard dependency: spec 020 phase 2 + 3 must have landed. Once tape-replay
trim works for `MambaCache`, this phase is mostly mechanical:

- `Qwen35Model`, `Qwen3NextModel`, `MiMoV2Flash` gain `MTPInjector`
  conformance.
- Their sanitize functions stop stripping `mtp.*` (gated on
  `GenerateParameters.mtpEnabled` to keep parity with non-MTP loads).
- Bench harness gains `--mtp` flag mirroring `--ngram`.
- Per model: confirm acceptance rate at temp=0 ≥ 70% on the realistic
  prompt set; below that, log a warning and consider tuning the contract
  defaults.

### Phase 5 — Companion assistant draft loader (variant B)

- `AssistantDraftRegistry` in
  `Libraries/MLXLMCommon/AssistantDraftRegistry.swift`. Hard-coded
  initial map:

| Target | Assistant draft |
|---|---|
| `mlx-community/gemma-4-E2B-it` | `mlx-community/gemma-4-E2B-it-assistant-bf16` |
| `mlx-community/gemma-4-E4B-it` | `mlx-community/gemma-4-E4B-it-assistant-bf16` |
| `mlx-community/gemma-4-26B-A4B-it` | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` |
| `mlx-community/gemma-4-31B-it` | `mlx-community/gemma-4-31B-it-assistant-bf16` |

  Prefix-match + org-strip resolver mirroring SwiftLM's
  `DFlashDraftRegistry`.
- Wire into `MLXLMCommon.generate(...)` with a single new branch: when
  `parameters.mtpEnabled` and target id matches the registry, construct
  `SpeculativeTokenIterator` with the assistant draft auto-loaded.
- Default `numDraftTokens` per the model card: 6 single, 3 batched. CLI
  override via `--draft-block-size`.
- Vocab parity check at registry-load time: SHA256 of the tokenizer
  vocab on both target and draft. Refuse with a clear error rather than
  silently producing garbage.

### Phase 6 — Adaptive depth and benchmark integration

- Adaptive draft depth (variant A): `mtp1 / mtp2 / mtpk / mtpa` modes
  mirroring MTPLX. `mtpa` is expected-value-driven (depth `k` chosen per
  cycle by maximising expected accepted tokens given the rolling
  acceptance rate).
- Bench harness adds `--mtp auto` flag: auto-selects depth by family
  (DeepSeek-V4: k=1; Qwen3-Next: k=2; MiMo: k=2; Gemma-4 assistant: k=6
  greedy / 3 batched).
- Sweep mode rows include MTP acceptance rate alongside n-gram and
  DFlash.

### Phase 7 — Sampling-correctness generalisation (cross-cutting)

- Refactor the spec-023 accept/reject sampler to a shared helper that
  serves n-gram, MTP variant A, and DFlash. Single source of truth for
  top-p-before-top-k ordering and residual renormalisation.
- Marginal-correctness oracle test runs across all three iterators in CI.

## What we should NOT do

- **Don't auto-enable MTP by default.** Loading MTP heads costs
  steady-state memory (DeepSeek-V3 layer-61 is ~14B params at the source;
  in 4-bit MLX still ~3.5 GB). Default off; opt-in via env or parameter.
  Document the memory cost in `documentation/mtp.md`.
- **Don't add upstream `mlx_lm.convert` flag PRs to this spec.** Path 1
  in the conversion section is the *cleanest* upstream remedy but lives
  in another repo and another community. Treat it as a side-quest.
- **Don't unify variant A and variant B iterators prematurely.** The
  abstractions diverge enough (single cache vs. two caches; in-trunk vs.
  external draft model; tokenizer-equality guarantee vs. registry-time
  validation) that forcing a common subclass leaks both. Keep them as
  two iterators that share `MTPSelfSpeculativeTokenIterator` (variant A)
  vs. a small `AssistantDraftRouter` wrapper around
  `SpeculativeTokenIterator` (variant B).
- **Don't ship phase 4 before spec 020 phase 2/3 lands.** Tape-replay is
  the hard prerequisite; without it, hybrid-GDN MTP cannot trim the
  trunk cache safely on rejection. Plain AR fall-back is the right
  behaviour during the gap.
- **Don't conflate "MTP" with "DFlash" in the public API surface or
  bench output.** Even though MTPLX-the-server and SwiftLM's harness use
  overlapping naming, they're distinct mechanisms with distinct
  trade-offs. Separate flags (`--mtp` vs. `--dflash`), separate metrics
  rows.

## Files touched (estimate)

| File | What | Lines (est.) |
|---|---|---|
| `Libraries/MLXLMCommon/MTPSelfSpeculativeDecoding.swift` (new) | Variant-A iterator. | ~700 |
| `Libraries/MLXLMCommon/MTPInjector.swift` (new) | Protocol + `MTPContract`. | ~120 |
| `Libraries/MLXLMCommon/MTPLoader.swift` (new) | Weight discovery + sanitize gating. | ~200 |
| `Libraries/MLXLMCommon/AssistantDraftRegistry.swift` (new) | Variant-B registry + resolver. | ~120 |
| `Libraries/MLXLMCommon/Evaluate.swift` (extension) | Auto-routing predicate. | ~60 |
| `Libraries/MLXLLM/Models/DeepseekV4.swift` (extension) | `MTPInjector` conformance + `DeepseekV4MTP` module + sanitize gating. | ~250 |
| `Libraries/MLXLLM/Models/Qwen35.swift` (extension) | Same. Phase 4. | ~200 |
| `Libraries/MLXLLM/Models/Qwen3Next.swift` (extension) | Same. Phase 4. | ~200 |
| `Libraries/MLXLLM/Models/MiMoV2Flash.swift` (extension) | Same. Phase 4. | ~150 |
| `Libraries/MLXLLM/Models/GLM4MOELite.swift` (extension) | Same. Phase 4. | ~150 |
| `scripts/mtp_convert.py` (new) | Custom converter (path 2). | ~250 |
| `Tests/MLXLMTests/MTPSelfSpeculativeTests.swift` (new) | Round-trip + marginal oracle. | ~400 |
| `Tests/Benchmarks/InferenceBenchmark.swift` | `--mtp` flag plumbing. | ~80 |
| `documentation/mtp.md` (new) | User docs: memory cost, flag matrix, model coverage table. | n/a |

Total ~2900 lines of new Swift + Python conversion script, plus per-target
extensions across the in-trunk family.

## Open questions

1. **Path 2 vs. path 1 for in-trunk reconversion.** Path 2 (custom
   converter) is the lower-risk choice for the initial port — full
   control, no upstream dependency. But long-term path 1 (upstream
   `mlx_lm.convert` carrying MTP) is cleaner. Open: do we contribute the
   upstream PR ourselves once our path-2 implementation has shaken out,
   or wait for the community? Probably the former, but not blocking.
2. **Vocab/tokenizer fingerprinting.** SHA256 of the vocab JSON catches
   most mismatches but not all (e.g. tokenizers with the same vocab but
   different special-token mappings can still produce subtly wrong
   output). Consider adding a chat-template SHA too.
3. **Adaptive depth policy.** MTPLX's
   `ExpectedValueDepthPolicy` uses a discount-rate model to pick `k` per
   cycle. Worth porting verbatim, or design our own simpler EWMA-based
   policy? The MTPLX policy is published with a correctness proof, so
   start with the verbatim port.
4. **MTP head storage layout.** DeepSeek-V3 puts the MTP head at
   `model.layers.{numHiddenLayers}.*` (one extra layer beyond the
   declared count) and our DeepseekV4 sanitize hard-codes that
   convention ([DeepseekV4.swift:909](../Libraries/MLXLLM/Models/DeepseekV4.swift#L909)).
   Other families key under `mtp.*` or `model.mtp.*`. The `MTPLoader`
   should probe all conventions; this is mechanical but the failure mode
   if we miss one is a silently-disabled MTP path. Add an explicit
   "loaded MTP heads: yes/no" log line at init.
5. **Vocab-shifted assistant drafts.** Some Gemma-4 assistant repos
   reserve a draft-only special token range. Confirm the four
   `*-it-assistant-bf16` mirrors don't have this; if they do, the vocab
   parity check needs to be a *prefix* match, not a full match.
6. **Memory pressure on Apple Silicon.** Loading the MTP head adds
   notably to wired-memory pressure on quantised DeepSeek-V3 (already
   ~50 GB at 4-bit). Profile against `WiredMemoryPolicies.swift` to make
   sure the head doesn't push us past the recommended budget on M2/M3.

## References

- MTPLX (Python MLX self-speculative decoder, primary reference):
  https://github.com/youssofal/MTPLX
- MTPLX sampling math: https://github.com/youssofal/MTPLX/blob/main/mtplx/sampling.py
- MTPLX runtime + injection: https://github.com/youssofal/MTPLX/blob/main/mtplx/runtime.py , https://github.com/youssofal/MTPLX/blob/main/mtplx/mtp_patch.py
- MTPLX backend registry (verified vs. compatible-unverified tiers):
  https://github.com/youssofal/MTPLX/blob/main/mtplx/backends/registry.py
- DeepSeek-V3 MTP weight layout (`README_WEIGHTS.md`):
  https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md
- Gemma-4 EAGLE-style assistant drafters (mlx-community mirrors):
  https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-assistant-bf16
- Multi-token prediction PR for mlx-lm (related):
  https://github.com/Blaizzy/mlx-lm/pull/15
- Spec 015 (companion-draft DFlash): [015-dflash-diffusion-speculative-decoding.md](015-dflash-diffusion-speculative-decoding.md)
- Spec 020 (tape-replay rollback — hybrid prerequisite):
  [020-tape-replay-rollback-generalised.md](020-tape-replay-rollback-generalised.md)
- Spec 023 (Leviathan accept/reject — sampling source):
  [023-leviathan-accept-reject-sampling.md](023-leviathan-accept-reject-sampling.md)
