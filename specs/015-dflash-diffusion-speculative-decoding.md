# 015 — DFlash diffusion speculative decoding (Swift port)

**Status:** spec, deliberately separate workstream from spec 013/014
**Branch:** new feature branch (probably `ek/dflash-mlx-swift`)
**Depends on:** spec 013 (n-gram path) only as a fallback target. DFlash
itself does not require n-gram speculation; the two are independent
draft sources.

## Problem and goal

DFlash (block diffusion for flash speculative decoding,
[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)) is a draft-model-based
speculative decoding method designed for the Qwen 3.5 / 3.6 architecture
family (hybrid full-attention + GatedDeltaNet recurrent layers). Unlike a
classic autoregressive draft model that proposes K tokens via K sequential
forward passes, the DFlash draft model is trained as a **block diffusion
model**: it produces 16 candidate tokens in **one** forward pass conditioned
on a captured hidden state from the target. The target then verifies all 16
in one forward pass and accepts the longest matching prefix.

The Python reference implementation is
[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2),
which targets stock MLX from pip. The Swift port already started in
[`SharpAI/SwiftLM`](https://github.com/SharpAI/SwiftLM) under
`Sources/DFlash/` and `Sources/SwiftLM/Qwen3*+DFlash.swift`.

The published numbers on Apple M5 Max from the reference implementation:

| Model | Tokens | Baseline | DFlash | Speedup | Acceptance |
|---|---|---|---|---|---|
| Qwen3.5-4B | 1024 | 53.8 tok/s | 182.9 tok/s | **3.40×** | 86.4% |
| Qwen3.5-9B | 1024 | 30.95 tok/s | 135.3 tok/s | **4.37×** | 89.6% |
| Qwen3.5-27B-4bit | 1024 | 33.55 tok/s | 79.0 tok/s | **2.37×** | 90.0% |
| Qwen3.5-35B-A3B-4bit | 1024 | 143.0 tok/s | 248.9 tok/s | **1.76×** | 89.3% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.3 tok/s | 300.3 tok/s | **2.20×** | 91.0% |

These dwarf what's achievable with prompt-lookup PLD on the same models, so
**DFlash is the highest-priority addition for Qwen 3.5 / 3.6** in this
project. The catch is that it needs a custom-trained draft model per target,
plus several non-trivial runtime pieces.

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

### Phase 1 — Bootstrapping: stub draft model + linear verify

Goal: get the cycle working end-to-end with a *placeholder* draft model
that just samples the target's argmax (i.e. zero speedup). This proves the
plumbing — protocol conformances, hidden-state capture, verify forward,
acceptance match, full-attention rollback — without depending on a trained
draft model.

- Create `Libraries/MLXLMCommon/DFlashSpeculativeDecoding.swift` defining
  `DFlashSpeculativeTokenIterator: TokenIteratorProtocol`.
- Add `DFlashTargetModel` protocol (mirroring SwiftLM's surface).
- Add a stub `DFlashDraftModel` that takes the target's last hidden state
  and emits the next 16 target argmax tokens (i.e. uses the target itself
  as draft). This is obviously slower than baseline, but it exercises the
  cycle.
- Add hidden-state capture to `Qwen3TextModel` (the dense Qwen 3 from spec
  013-supported set) — this is the simplest target for bootstrapping.
- Acceptance + commit + rollback for full-attention only (no GDN).
- Wire into `MLXLMCommon.generate(...)` behind a separate
  `parameters.dflashEnabled` flag, distinct from the n-gram path.

### Phase 2 — Real draft model

- Pull a trained DFlash draft model from HuggingFace (the
  `z-lab/Qwen3.5-9B-DFlash` family).
- Implement the draft model's forward in Swift: GLU MLP, cross-attention
  with `ContextOnlyDraftKVCache`, block diffusion forward over 16 mask
  tokens.
- Verify against the Python reference implementation's outputs (token-by-
  token equality on a fixed seed) before turning on speedup measurement.

### Phase 3 — Hybrid GDN rollback

- Port `RecurrentRollbackCache` to Swift. The innovation-tape buffer is a
  per-layer ring of MLXArray "delta" entries. Recording happens during
  verify forward; replay happens after acceptance match resolves
  `accepted_count`.
- Custom Metal kernel for replay: takes the saved deltas and accepted
  count, applies the deltas in order to the layer's `recState` /
  `convState`. SwiftLM's `Sources/DFlash/DFlashSWAMask.swift` is a
  reference for the kernel surgery style.
- Extend `DFlashTargetModel` conformance for the hybrid models
  (`Qwen35TextModel` and the 35B-A3B MoE).

### Phase 4 — Auto draft resolution + registry

- `DFlashDraftRegistry`: target HF id → draft HF id mapping. SwiftLM has
  this in `Sources/DFlash/DFlashDraftRegistry.swift`.
- Bench harness `--dflash auto` flag that resolves the draft from the
  registry. `--dflash-draft <id>` for explicit override.
- Handle missing draft → fall back to `TokenIterator` (or n-gram path if
  configured) gracefully, like the n-gram iterator falls back on hybrid
  cache.

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

- N-gram covers Gemma 4 / GPT-OSS / Llama / Phi / Qwen 2-3 dense — pure
  attention models with no DFlash draft. ~+25% on input-grounded tasks.
- DFlash covers Qwen 3.5 / 3.6 — hybrid GDN + attention. ~2-4× on the same
  workload class (and many others), but only if a DFlash draft is
  registered.

The two should coexist behind separate auto-routing predicates in
`MLXLMCommon.generate(...)`. Roughly:

```
if dflash-eligible (target has DFlash protocol conformance + draft available):
    use DFlashSpeculativeTokenIterator
elif ngram-eligible (params.ngramSize >= 1, temp == 0, fully trimmable cache):
    use NGramSpeculativeTokenIterator
else:
    use TokenIterator
```

Caller-supplied configuration can disable either path explicitly.

## Files touched (Swift port estimate)

| File | What | Lines (est.) |
|---|---|---|
| `Libraries/MLXLMCommon/DFlashSpeculativeDecoding.swift` (new) | Iterator + cycle. | ~600 |
| `Libraries/MLXLMCommon/DFlashTargetModel.swift` (new) | Protocol + helpers. | ~80 |
| `Libraries/MLXLMCommon/DFlashDraftModel.swift` (new) | Draft transformer. | ~400 |
| `Libraries/MLXLMCommon/RecurrentRollbackCache.swift` (new) | Tape-replay cache wrapping `MambaCache`. | ~250 |
| `Libraries/MLXLMCommon/DFlashDraftRegistry.swift` (new) | Target → draft mapping. | ~80 |
| `Libraries/MLXLLM/Models/Qwen35.swift` (extension) | Capture + protocol conformance. | ~80 |
| `Libraries/MLXLLM/Models/Qwen3.swift` (extension) | Same for dense Qwen 3. | ~50 |
| `Libraries/MLXLLM/Models/Qwen35MoE.swift` (extension) | Same for 35B-A3B. | ~80 |
| `Sources/Cmlx/mlx-generated/metal/dflash_replay.metal` (new) | Innovation-tape replay kernel. | ~200 |
| `Tests/MLXLMTests/DFlashSpeculativeTests.swift` (new) | Unit + integration tests. | ~300 |
| `Tests/Benchmarks/InferenceBenchmark.swift` | DFlash bench mode plumbing. | ~80 |

Total ~2300 lines of new Swift + Metal code, plus per-target model
extensions across the Qwen family.

## Open questions

1. **Draft model availability.** All current DFlash drafts are on HF as
   `z-lab/Qwen*-DFlash`. Are these MLX-format-compatible (mlx_lm style
   safetensors with the right key naming)? If not, we need a converter or
   a one-shot upload. SwiftLM's `Sources/DFlash/DFlashDraftBackend.swift`
   handles this on their side; check whether they already publish
   converted weights.
2. **Innovation-tape memory.** Per layer per cycle, we hold ~16 deltas. For
   Qwen 3.5 35B-A3B that's 60 layers × 16 × `state_dim` MLXArrays per
   cycle. Profile the watermark; this might force a smaller block size on
   long contexts.
3. **Prefix cache coupling.** Should the iterator have a hook for
   pre-warming with a snapshot, or should the prefix cache be a layer
   above the iterator that hands the iterator a pre-filled `[KVCache]`?
   The latter is cleaner if we can keep iterator stateless w.r.t. cache
   provenance.
4. **Block size.** 16 is dflash-mlx's default. Adaptive block size based
   on rolling acceptance rate (à la spec 013 §2 adaptive draft length)
   could help long-context decode where acceptance drops. Probably a
   phase-5 follow-up.

## References

- DFlash paper: https://arxiv.org/abs/2602.06036
- dflash-mlx engine-v2 (Python reference):
  https://github.com/bstnxbt/dflash-mlx/tree/engine-v2
- SwiftLM DFlash port (in-progress):
  https://github.com/SharpAI/SwiftLM/tree/main/Sources/DFlash
- Qwen3.5 / 3.6 published DFlash drafts:
  https://huggingface.co/z-lab
- Multi-token prediction (related family — DeepSeek-V4 native MTP head):
  https://github.com/Blaizzy/mlx-lm/pull/15
