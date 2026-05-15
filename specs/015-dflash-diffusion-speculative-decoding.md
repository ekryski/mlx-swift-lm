# 015 — DFlash diffusion speculative decoding (Swift port)

**Status:** 🚧 Phase 1 scaffold landed ([PR #141](https://github.com/ekryski/mlx-swift-lm/pull/141) — protocol surface + iterator scaffold). Phase 2 (real draft model from `z-lab/Qwen3.5-*-DFlash` / `gemma-4-*-DFlash`) and Phase 3 (hybrid GDN state-replay refactor onto [spec 020](020-tape-replay-rollback-generalised.md)'s `StateReplayCache` — prerequisite ✅ landed) open. Phases 4–6 (DFlash-on-ANE — Tier 4 row 13) blocked on Tier 3 row 11 + spec 021. **Phase 2 + 3 not started.**
**Branch:** Phase 1 merged via PR #141; later phases get fresh branches off alpha.
**Depends on:** spec 013 (n-gram path) only as a fallback target. DFlash
itself does not require n-gram speculation; the two are independent
draft sources.

> **Update 2026-05-08 — scope expanded to Gemma 4.** Z-lab now publishes
> DFlash drafts for Gemma 4 (`gemma-4-31B-it-DFlash`,
> `gemma-4-26B-A4B-it-DFlash`), and `bstnxbt/dflash-mlx` `main` branch
> carries a working Gemma 4 target adapter
> (`dflash_mlx/engine/target_gemma4.py`). The Gemma 4 path is materially
> simpler than Qwen 3.5/3.6 because Gemma 4 is full attention + sliding
> window — **no GatedDeltaNet recurrent state, no tape-replay rollback
> needed** — so it should ship before the hybrid Qwen path. See "Gemma 4
> DFlash path" section below for the per-family details.

## Problem and goal

DFlash (block diffusion for flash speculative decoding,
[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)) is a draft-model-based
speculative decoding method originally designed for the Qwen 3.5 / 3.6
architecture family (hybrid full-attention + GatedDeltaNet recurrent
layers) and now extended to Gemma 4 (full attention + sliding window).
Unlike a classic autoregressive draft model that proposes K tokens via K
sequential forward passes, the DFlash draft model is trained as a **block
diffusion model**: it produces 16 candidate tokens in **one** forward pass
conditioned on a captured hidden state from the target. The target then
verifies all 16 in one forward pass and accepts the longest matching
prefix.

The Python reference implementation is
[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx). **Use the
`main` branch as source of truth, not `engine-v2`** — `engine-v2`
collapsed the per-family `target_*.py` adapters into a single generic
`target_verifier.py` and lost Gemma 4 fidelity (no `embed_scale`, no
`per_layer_inputs`, no `final_logit_softcapping`, no SWA cache awareness).
The Swift port already started in
[`SharpAI/SwiftLM`](https://github.com/SharpAI/SwiftLM) on the
`feat/add-dflash` branch (and was extended on `feat/mtp-harness-updates`
for additional model coverage).

The published numbers on Apple M5 Max from the reference implementation:

| Model | Tokens | Baseline | DFlash | Speedup | Acceptance |
|---|---|---|---|---|---|
| Qwen3.5-4B | 1024 | 53.8 tok/s | 182.9 tok/s | **3.40×** | 86.4% |
| Qwen3.5-9B | 1024 | 30.95 tok/s | 135.3 tok/s | **4.37×** | 89.6% |
| Qwen3.5-27B-4bit | 1024 | 33.55 tok/s | 79.0 tok/s | **2.37×** | 90.0% |
| Qwen3.5-35B-A3B-4bit | 1024 | 143.0 tok/s | 248.9 tok/s | **1.76×** | 89.3% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.3 tok/s | 300.3 tok/s | **2.20×** | 91.0% |
| Gemma-4-26B-A4B-4bit (z-lab) | — | — | — | up to **3.7×** | — |
| Gemma-4-31B-4bit (z-lab) | — | — | — | up to **5.8×** at concurrency 1, 15-16 spec tokens | — |

Gemma 4 numbers are from the z-lab model cards (no A/B baseline published
on the same hardware as the Qwen rows; treat as upper bound until we
re-bench on Apple Silicon).

These dwarf what's achievable with prompt-lookup PLD on the same models, so
**DFlash is the highest-priority addition for Qwen 3.5 / 3.6 and Gemma 4**
in this project. The catch is that it needs a custom-trained draft model
per target, plus several non-trivial runtime pieces.

## Architecture overview

DFlash decode is a state machine repeating the cycle
`prefill → draft block → target verify → acceptance match → commit + rollback`,
keyed off captured hidden states from the target. The reference Python
implementation lives in
`dflash_mlx/engine/spec_epoch.py` (`stream_dflash_generate_impl`); the Swift
counterpart should mirror that structure.

### 1. Draft model

A small (~1B params) Qwen-family transformer with cross-attention to a
**target hidden state** captured at one or more selected layers of the
target. The draft model has its own KV cache for self-attention and a
context-only cache for the cross-attention to target hidden states.

In SwiftLM the architecture is in
`Sources/DFlash/DFlashDraftModel.swift`. The configuration declares
`numTargetLayers` (the target's layer count) and `targetLayerIDs` (which
layers' hidden states are captured) plus `blockSize` (16, the diffusion
window).

### 2. Block diffusion draft step

Standard autoregressive drafting runs the draft model `K` times to produce
`K` tokens. DFlash's draft model is *non-autoregressive*: one forward pass
produces 16 candidate tokens in parallel, denoised from a starting point
that's a sequence of `[MASK]` tokens conditioned on the target hidden
state. The "block" is the 16-token window; the diffusion is the iterative
refinement that produces all 16 tokens.

In practice the draft model's forward pass takes ~2-3× the cost of a
single-token target decode but emits 16 candidates instead of 1, so even at
50% acceptance the throughput math is favourable.

### 3. Target verify with hidden-state capture

The target runs `[1, 17]` (last committed token + 16 drafts) in one forward
pass, returning logits at all 17 positions AND captured hidden states from
the configured target-layer-IDs. The captured hidden states feed the next
draft cycle's cross-attention.

`DFlashTargetModel` protocol in `Sources/DFlash/`:

```swift
protocol DFlashTargetModel {
    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray
    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray
    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray])
    var dflashIsHybridGDN: Bool { get }
}
```

This is mechanical to add to each Qwen target; SwiftLM already does it for
`Qwen35TextModel` in `Sources/SwiftLM/Qwen35+DFlash.swift`.

### 4. Acceptance match

Vectorized longest-common-prefix between draft tokens and target argmax over
the verify positions. dflash-mlx uses `cumprod` over the equality mask:
the first zero in the cumulative product marks the first mismatch. Same
shape as our linear `proposeDraft` accept loop, just without the
out-of-Python loop overhead.

### 5. Rollback for hybrid GatedDeltaNet

This is the technically hardest piece. Standard speculative decoding trims
the cache by `(numDraft - accepted)` — works for full-attention layers
because their KV cache is positional and deterministic. **GatedDeltaNet
layers don't have positional state — they have a recurrent state vector
that mixes information from all input positions.** Trimming "the last K
positions" doesn't make sense for a recurrent layer.

dflash-mlx uses **tape-replay rollback**: during verify it records an
"innovation tape" (per-step recurrent updates), and on partial acceptance
it replays only the accepted steps through a custom Metal kernel. This
keeps GDN state consistent with the committed prefix without snapshotting
the full state vector per cycle (which would be too expensive).

Reference: `dflash_mlx/recurrent_rollback_cache.py`,
`dflash_mlx/engine/rollback.py`.

### 5b. Gemma 4 DFlash path (full attention + sliding window)

Gemma 4 is **full attention with a sliding-window pattern** — no
GatedDeltaNet, no recurrent state, no tape-replay required. The DFlash
mechanics on Gemma 4 are accordingly simpler than on Qwen 3.5/3.6, but
Gemma 4's *target* forward has more model-specific dressing than Qwen and
that dressing has to be threaded through DFlash's hidden-state-capture
forward.

Reference implementation:
[`dflash_mlx/engine/target_gemma4.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/engine/target_gemma4.py)
on the **`main` branch** (NOT `engine-v2` — that branch dropped Gemma 4
fidelity).

What the Gemma 4 target adapter must do, beyond the Qwen-flavoured
forward:

1. **Embed scale** — multiply token embeddings by `inner.embed_scale`
   before the trunk (Gemma's residual scaling convention).
2. **Per-layer inputs** — Gemma 4 carries `per_layer_inputs` (PLE
   features) injected at each transformer layer. The capture forward
   must thread `(hidden, per_layer_inputs)` through the layer stack
   intact. Reference helpers: `_get_per_layer_inputs` and
   `_project_per_layer_inputs` in the model file.
3. **SWA mask** — call the trunk's existing `_make_masks` to produce the
   sliding-window + full-attention pattern. Don't re-derive it; Gemma 4
   layer types alternate `(sliding_attention, sliding_attention, ...,
   full_attention)` and we already have the right helper at
   [Gemma4.swift](../Libraries/MLXLLM/Models/Gemma4.swift).
4. **Logit softcap** — apply `tanh(logits / cap) * cap` after `lm_head`
   when `final_logit_softcapping` is set in the config (it usually is).
5. **Shared KV + offset** — propagate `(shared_kv, offset)` through
   layers. Gemma 4's tied K/V across the SWA group is part of the cache
   contract.
6. **Cache rollback** — Gemma 4 ships a `RotatingKVCache` for the SWA
   layers. Trim is `_trim_recent_cache` style: reorder via
   `_temporal_order` and then truncate. Plain `cache.trim()` works for
   the full-attention layers. `Gemma4TargetOps.capabilities_for` returns
   `supports_recurrent_rollback=False, supports_kv_trim=True` —
   straight KV trim is sufficient.
7. **Cache constraints** — reject `quantize_kv_cache=True` and any
   override of `target_fa_window`. Gemma 4 owns its SWA cache; users
   asking for affine-quantised KV on a SWA model get a clear error
   rather than silent corruption.

The capture-feature extraction is the standard
`mx.concatenate([captured_dict[layer_id + 1] for layer_id in
target_layer_ids], axis=-1)` — note the `+ 1` because layer index `0`
captures the (post-embed-scale, post-PLE-projection) embedding output,
not a transformer layer output.

#### Gemma 4 draft model

The z-lab Gemma 4 DFlash drafts are published at:
- `z-lab/gemma-4-31B-it-DFlash` (2B params, BF16 safetensors)
- `z-lab/gemma-4-26B-A4B-it-DFlash` (0.4B params, BF16 safetensors)

These are **not** generic small transformers. Per the z-lab project page,
DFlash drafts (a) extract hidden features from selected target layers via
*Feature Fusion*, (b) inject those features into every draft layer's K/V
projections via *KV Injection* (distinct from EAGLE-3's first-layer-only
feed), (c) draft blocks in parallel. The drafter reuses the target's
embedding + LM head; only intermediate layers carry trained weights. The
draft layer-type pattern follows the target's SWA layout — see
`_default_draft_layer_types` in `dflash_mlx/model.py`. **Reproduce that
emitter in our Swift port for the Gemma 4 path** so the draft attention
mask matches the target's expectations.

Capture-layer selection is checkpoint-baked. Use
`build_target_layer_ids(num_target_layers, num_draft_layers)` from
`dflash_mlx/model.py` verbatim:
`start = 1, end = N - 3, draft_i picks round(start + i * (end - start) / (D - 1))`.
The Swift port should call the same arithmetic, not pick its own values.

#### Why Gemma 4 should ship first

- No tape-replay kernel — phase 3's hardest piece (custom Metal kernel
  for innovation-tape replay through GDN state) is not on the critical
  path for Gemma 4.
- No `MambaCache` subclassing — the rollback story is `RotatingKVCache`
  trim, which we already have.
- Gemma 4 PLE projection is already wired in our existing target
  ([Gemma4.swift](../Libraries/MLXLLM/Models/Gemma4.swift)), so the
  capture forward only adds the layer-output collection plumbing.
- Acceptance is empirically high (the model card claims up to 5.8× at
  concurrency 1 with a 15-16-token speculation block, well above the
  Qwen 3.5 numbers at the same block size).

The existing phase 1 (bootstrapping) and phase 2 (real draft) can be
re-targeted at Gemma 4-31B and Gemma 4-26B-A4B before the hybrid Qwen
work, with the GDN-specific phases (3, parts of 5) deferred. See updated
implementation phases below.

### 6. Prefix cache (cross-request)

DFlash ships a cross-request prefix cache:
`DFlashPrefixCache` in `dflash_mlx/cache/prefix_l1.py`. At end of request,
snapshot the target's KV state keyed by the *stable* prefix tokens (chat
template-aware: trims the trailing `<|im_start|>assistant` boilerplate so
the next turn's matchable prefix is the same). Next request: lookup the
longest matching cached snapshot and skip prefill for that prefix.

This is orthogonal to DFlash itself but is in the same package because
their server (`dflash-serve`) needs both. **For our Swift port, the prefix
cache should be its own spec (017?) since it's useful even without DFlash.**

### 7. Verify-specialized int4 qmm

The target verify forward has shape `[1, 17]` going through quantized
matmuls. Stock `mx.quantized_matmul` is optimised for `M=1` decode shapes;
at `M=17` it leaves performance on the table. dflash-mlx ships a custom
Metal SIMDgroup-MMA kernel (`verify_qmm`) with two variants:
- `mma2big` — fast on smaller K dims.
- `mma2big_pipe` — K-split + double-buffered staging for larger K (MoE,
  ≥40-layer dense).

This is "spec 015 phase 6" — large effort, large speedup at long contexts.
Skip until phases 1-5 land.

## Implementation phases

> **Reordered 2026-05-08.** Gemma 4 (full attention + SWA) ships before
> the hybrid Qwen 3.5/3.6 path. Tape-replay rollback (the hardest
> piece) moves out of the critical path for the initial DFlash launch.

### Phase 1 — Bootstrapping: stub draft model + linear verify

Goal: get the cycle working end-to-end with a *placeholder* draft model
that just samples the target's argmax (i.e. zero speedup). This proves the
plumbing — protocol conformances, hidden-state capture, verify forward,
acceptance match, full-attention rollback — without depending on a trained
draft model.

- Create `Libraries/MLXLMCommon/DFlashSpeculativeDecoding.swift` defining
  `DFlashSpeculativeTokenIterator: TokenIteratorProtocol`.
- Add `DFlashTargetModel` protocol (mirroring SwiftLM's surface). Keep
  the protocol family-agnostic — the Gemma 4 and Qwen 3.5 implementations
  will diverge in the body but expose the same shape.
- Add a stub `DFlashDraftModel` that takes the target's last hidden state
  and emits the next 16 target argmax tokens (i.e. uses the target itself
  as draft). This is obviously slower than baseline, but it exercises the
  cycle.
- Add hidden-state capture to **`Gemma4TextModel`** as the bootstrap
  target — full-attention + SWA, simplest cache rollback story, and our
  per_layer_inputs / embed_scale / softcap handling already lives there.
  (This is a change from the original spec — `Qwen3TextModel` is no
  longer the bootstrap target.)
- Acceptance + commit + rollback for full-attention + SWA only (no GDN).
  Use `RotatingKVCache.trim` via the `_temporal_order` reorder helper
  for the SWA layers and plain `cache.trim()` for full-attention layers.
- Wire into `MLXLMCommon.generate(...)` behind a separate
  `parameters.dflashEnabled` flag, distinct from the n-gram path.

### Phase 2 — Real draft model on Gemma 4

- Pull `z-lab/gemma-4-26B-A4B-it-DFlash` (smaller, faster iteration) and
  `z-lab/gemma-4-31B-it-DFlash` (the bigger headline number).
- Implement the draft model's forward in Swift: GLU MLP,
  cross-attention with `ContextOnlyDraftKVCache` (sink + sliding
  window), block-diffusion forward over 16 mask tokens. Reuse the
  target's `embedTokens` + `lmHead` (DFlash convention — the draft is
  weight-tied to the target on those modules).
- Implement `_default_draft_layer_types` for Gemma 4 in Swift — emit the
  SWA pattern matching the target's layer-type config so the draft
  attention mask is consistent with the target's expectations.
- Implement `build_target_layer_ids` verbatim. Capture-layer choice is
  checkpoint-baked.
- Verify against the Python reference implementation's outputs (token-by-
  token equality on a fixed seed) before turning on speedup measurement.
  Reference tests in `dflash-mlx`: `tests/test_target_gemma4_real.py`,
  `tests/test_target_gemma4_cache.py`, `tests/test_gemma4_draft.py` —
  all on the **`main` branch**.

### Phase 3 — Hybrid GDN rollback (Qwen 3.5 / 3.6)

> Hard dependency: spec 020 phase 2 + 3 (tape-replay rollback at the
> cache layer). This phase is what gates DFlash on the hybrid Qwen
> models. Gemma 4 (phase 2) does not need any of this.

- Port `RecurrentRollbackCache` to Swift. The innovation-tape buffer is a
  per-layer ring of MLXArray "delta" entries. Recording happens during
  verify forward; replay happens after acceptance match resolves
  `accepted_count`.
- Custom Metal kernel for replay: takes the saved deltas and accepted
  count, applies the deltas in order to the layer's `recState` /
  `convState`. SwiftLM's
  [`Sources/DFlash/DFlashKernels.swift`](https://github.com/SharpAI/SwiftLM/blob/feat/add-dflash/Sources/DFlash/DFlashKernels.swift)
  on `feat/add-dflash` is the reference — two kernels (`dflash_tape_replay`
  + `dflash_gated_delta_tape`), each with `_vec` and `_mask` variants,
  using `metal::select` for branchless conditional state update. Subclass
  `MambaCache` so existing `cache as? MambaCache` checks in the GDN
  forward still work.
- **Empirical question to resolve before locking this in:** SwiftLM's
  Qwen3Next adapter explicitly disables tape rollback (sets
  `dflashIsHybridGDN: false`) with the comment that "any rollback
  scheme degrades acceptance rate by leaving recurrent state stale, and
  rejected-token contamination is empirically negligible (<1 reject per
  accepted cycle at long context)." Measure both paths (tape-replay vs.
  no rollback) on Qwen 3.5 / 3.6 and pick the winner per model. If the
  no-rollback path holds within 2-3% of the tape-replay path on
  acceptance, it's the right default — much less code, no Metal kernel
  to maintain.
- Apply the `_ExactSmallProjPad` trick from
  `dflash_mlx/engine/target_qwen_gdn.py` — pad M to 16 for `in_proj_b`
  and `in_proj_a` projections, since matmul accuracy degrades for tiny
  M.
- Extend `DFlashTargetModel` conformance for the hybrid models
  (`Qwen35TextModel`, the 35B-A3B MoE, optionally `Qwen3NextModel`).

### Phase 4 — Auto draft resolution + registry

- `DFlashDraftRegistry`: target HF id → draft HF id mapping. SwiftLM has
  this in
  [`Sources/DFlash/DFlashDraftRegistry.swift`](https://github.com/SharpAI/SwiftLM/blob/feat/add-dflash/Sources/DFlash/DFlashDraftRegistry.swift)
  — adopt the prefix-match + org-strip resolver. Initial map:

  | Target | Draft |
  |---|---|
  | `mlx-community/gemma-4-31B-it-*` | `z-lab/gemma-4-31B-it-DFlash` |
  | `mlx-community/gemma-4-26B-A4B-it-*` | `z-lab/gemma-4-26B-A4B-it-DFlash` |
  | `mlx-community/Qwen3.5-{4B,9B,27B}*` | `z-lab/Qwen3.5-{4B,9B,27B}-DFlash` |
  | `mlx-community/Qwen3.5-35B-A3B*` | `z-lab/Qwen3.5-35B-A3B-DFlash` |
  | `mlx-community/Qwen3.6-27B*` | `z-lab/Qwen3.6-27B-DFlash` |
  | `mlx-community/Qwen3.6-35B-A3B*` | `z-lab/Qwen3.6-35B-A3B-DFlash` |

- Bench harness `--dflash auto` flag that resolves the draft from the
  registry. `--dflash-draft <id>` for explicit override.
- Handle missing draft → fall back to `TokenIterator` (or n-gram path if
  configured) gracefully, like the n-gram iterator falls back on hybrid
  cache.
- z-lab's `Qwen3.6-27B-DFlash` model card flags it as "still under
  training" with engine-support gaps. Gate it behind a `tier:
  experimental` flag in the registry; auto-mode skips experimental
  drafts, opt-in via `--dflash auto:experimental`.

### Phase 5 — Length budget + auto-fallback

DFlash has a context-length sweet spot per model. From the reference
benchmarks, speedup tapers from 4.37× at 1024 tokens to 2.22× at 8192 on
Qwen3.5-9B. Beyond ~16K, the draft model's cross-attention starts costing
more than it saves.

- Implement `DFLASH_MAX_CTX` environment threshold (default per model
  family from a small registry).
- Auto-fall-back to TokenIterator for prompts above the threshold or when
  `parameters.maxTokens <= 256` (per dflash-mlx, the per-request fixed
  costs aren't worth it for short generations).

### Phase 6 — Verify-specialized int4 qmm kernel (deferred)

Out of scope for the initial port. Track separately as 015b. Prerequisite
is the `make-metal` pipeline already in this repo; the kernel itself
follows the existing SIMDgroup-MMA pattern in our metallib.

## What we should NOT do

- Don't try to share infrastructure between DFlash and the n-gram iterator
  beyond `TokenIteratorProtocol`. The two paths have different cache
  semantics (DFlash needs hidden-state capture and tape-replay rollback;
  n-gram doesn't), different trim semantics (DFlash trims by acceptance
  match in 16-block, n-gram trims by accepted draft count in linear-K),
  and different fallback strategies. Forcing them through a common
  abstraction will leak both into each other.
- Don't ship the prefix cache and DFlash in the same PR. Prefix cache is
  itself worth ~1.5-2× on multi-turn agentic workloads regardless of which
  decode path runs underneath. Spec 017.
- Don't attempt VLM support. DFlash is for text-only Qwen. VLM needs
  separate spec work for the cross-attention path.

## Dependency on the n-gram work

DFlash is not a successor to or replacement for the n-gram path:

- N-gram covers GPT-OSS / Llama / Phi / Qwen 2-3 dense — pure attention
  models with no DFlash draft. ~+25% on input-grounded tasks.
- DFlash covers **Gemma 4** (full attention + SWA) and **Qwen 3.5 / 3.6**
  (hybrid GDN + attention). ~2-6× depending on model. Requires a
  DFlash draft registered for the target.
- For Gemma 4 specifically, both n-gram (PLD) and DFlash now apply.
  Order of precedence: DFlash > n-gram > AR. Auto-routing should pick
  DFlash when a draft is registered and fall through to n-gram
  otherwise.

The two should coexist behind separate auto-routing predicates in
`MLXLMCommon.generate(...)`. Roughly:

```
if dflash-eligible (target has DFlash protocol conformance + draft available):
    use DFlashSpeculativeTokenIterator
elif mtp-eligible (target has MTPInjector + MTP weights present; spec 030):
    use MTPSelfSpeculativeTokenIterator
elif ngram-eligible (params.ngramSize >= 1, temp == 0, fully trimmable cache):
    use NGramSpeculativeTokenIterator
else:
    use TokenIterator
```

Caller-supplied configuration can disable any path explicitly.

## Files touched (Swift port estimate)

| File | What | Lines (est.) | Phase |
|---|---|---|---|
| `Libraries/MLXLMCommon/DFlashSpeculativeDecoding.swift` (new) | Iterator + cycle. | ~600 | 1 |
| `Libraries/MLXLMCommon/DFlashTargetModel.swift` (new) | Protocol + helpers. | ~80 | 1 |
| `Libraries/MLXLMCommon/DFlashDraftModel.swift` (new) | Draft transformer (block diffusion + KV injection). | ~450 | 2 |
| `Libraries/MLXLMCommon/DFlashDraftLayerTypes.swift` (new) | `_default_draft_layer_types` emitter for SWA + full patterns. | ~80 | 2 |
| `Libraries/MLXLMCommon/DFlashDraftRegistry.swift` (new) | Target → draft mapping (Gemma 4 + Qwen 3.5/3.6). | ~120 | 4 |
| `Libraries/MLXLLM/Models/Gemma4.swift` (extension) | Capture forward, embed_scale, per_layer_inputs threading, softcap, protocol conformance. | ~150 | 1-2 |
| `Libraries/MLXLMCommon/RecurrentRollbackCache.swift` (new) | Tape-replay cache wrapping `MambaCache`. | ~250 | 3 |
| `Libraries/MLXLLM/Models/Qwen35.swift` (extension) | Capture + protocol conformance. | ~80 | 3 |
| `Libraries/MLXLLM/Models/Qwen3.swift` (extension) | Same for dense Qwen 3. | ~50 | 3 |
| `Libraries/MLXLLM/Models/Qwen35MoE.swift` (extension) | Same for 35B-A3B. | ~80 | 3 |
| `Libraries/MLXLLM/Models/Qwen3Next.swift` (extension) | Same for Qwen 3-Next-80B (optional). | ~80 | 3 |
| `Sources/Cmlx/mlx-generated/metal/dflash_replay.metal` (new) | Innovation-tape replay kernel. | ~200 | 3 |
| `Tests/MLXLMTests/DFlashSpeculativeTests.swift` (new) | Unit + integration tests (Gemma 4 first, Qwen later). | ~400 | 1-3 |
| `Tests/Benchmarks/InferenceBenchmark.swift` | DFlash bench mode plumbing. | ~80 | 4 |

Total ~2700 lines of new Swift + Metal code, plus per-target model
extensions across Gemma 4 and the Qwen family. Gemma 4 path alone (no
GDN, no Metal kernel) is ~1700 lines and could ship as a self-contained
PR.

## Open questions

1. **Draft model availability and format.** All current DFlash drafts are
   on HF as `z-lab/{family}-DFlash`. They ship as BF16 safetensors. We
   need to confirm key-naming compatibility with our existing module
   trees (DFlash drafts have a peculiar weight-tying convention with the
   target on `embedTokens` + `lmHead`). SwiftLM's
   [`Sources/DFlash/DFlashDraftBackend.swift`](https://github.com/SharpAI/SwiftLM/blob/feat/add-dflash/Sources/DFlash/DFlashDraftBackend.swift)
   handles this on their side; check whether they already publish
   MLX-converted variants on HF. If not, do a one-shot
   `mlx_lm.convert`-style pass and upload to `mlx-community`.
2. **Innovation-tape memory** (phase 3 only). Per layer per cycle, we
   hold ~16 deltas. For Qwen 3.5 35B-A3B that's 60 layers × 16 ×
   `state_dim` MLXArrays per cycle. Profile the watermark; this might
   force a smaller block size on long contexts. Doesn't apply to Gemma 4
   (no GDN).
3. **Prefix cache coupling.** Should the iterator have a hook for
   pre-warming with a snapshot, or should the prefix cache be a layer
   above the iterator that hands the iterator a pre-filled `[KVCache]`?
   The latter is cleaner if we can keep iterator stateless w.r.t. cache
   provenance. dflash-mlx's `DFlashPrefixKey` includes
   `target_model_id`, `draft_model_id`, `capture_layer_ids`,
   `draft_sink_size`, `draft_window_size`, `target_fa_window`, and a
   `format_version`. Adopt that key shape verbatim — invalidates the
   cache when any axis changes.
4. **Block size.** 16 is dflash-mlx's default. The Gemma-4-31B model
   card recommends 15-16 for its highest-acceleration regime. Adaptive
   block size based on rolling acceptance rate (à la spec 013 §2
   adaptive draft length) could help long-context decode where
   acceptance drops. Probably a phase-5 follow-up.
5. **Tape-replay vs no-rollback** (phase 3). SwiftLM's Qwen3Next adapter
   skips tape-replay entirely on the empirical claim that "rejected-token
   contamination is empirically negligible (<1 reject per accepted cycle
   at long context)." dflash-mlx maintains tape-replay on principle.
   Resolve this by measuring, not by argument: build both paths in
   phase 3, A/B them at long context, default to whichever wins by 3% or
   more on the realistic prompt set. If the gap is below 3%, default to
   no-rollback (less Metal kernel maintenance).
6. **Gemma 4 verify_qmm**. The `verify_qmm` Metal kernel
   (phase 6 / 015b) is currently spec'd for the M=17 verify shape on
   quantised int4 matmuls. Gemma 4's quantised path uses our
   TurboQuant kernels at decode (`M=1`), but for verify (`M=17`) we'd
   want the same custom kernel. Confirm Gemma 4 quantisation layout
   matches the assumptions in dflash-mlx's `verify_qmm` (`BM=16`,
   `BN=32`, `BK=32`, `BK_SUB=8`) before scoping that work.

## References

- DFlash paper: https://arxiv.org/abs/2602.06036
- dflash-mlx (Python reference, **`main` branch — not `engine-v2`**):
  https://github.com/bstnxbt/dflash-mlx
- dflash-mlx Gemma 4 target adapter:
  https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/engine/target_gemma4.py
- dflash-mlx Qwen GDN target adapter:
  https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/engine/target_qwen_gdn.py
- dflash-mlx draft model + layer-type emitter:
  https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/model.py
- dflash-mlx verify_qmm Metal source:
  https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/verify_qmm.py
- dflash-mlx prefix cache key shape:
  https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/cache/prefix_l1.py
- SwiftLM DFlash port (canonical branch is `feat/add-dflash`):
  https://github.com/SharpAI/SwiftLM/tree/feat/add-dflash/Sources/DFlash
- SwiftLM extended target coverage (DeepSeek-V3 / Kimi-K2):
  https://github.com/SharpAI/SwiftLM/tree/feat/mtp-harness-updates
- z-lab DFlash drafts (Gemma 4 + Qwen 3.5/3.6):
  https://huggingface.co/z-lab
- z-lab DFlash project page (architecture overview):
  https://z-lab.ai/projects/dflash
- Multi-token prediction (sibling spec, native MTP heads):
  [030-multi-token-prediction.md](030-multi-token-prediction.md)
- Multi-token prediction PR for mlx-lm (related work):
  https://github.com/Blaizzy/mlx-lm/pull/15
