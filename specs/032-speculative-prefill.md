# 032 — Speculative prefill (drafter-scored span selection)

**Status:** spec, ready (high-leverage, no kernel work)
**Branch:** new branch off alpha
**Depends on:** [015 phase 2](015-dflash-diffusion-speculative-decoding.md) (draft model integration), [017](017-prefix-kv-cache.md) (composes for multi-turn)
**Origin:** Research review 2026-05-08; PFlash ([Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/tree/main/pflash), [HN](https://news.ycombinator.com/item?id=47975259)) demonstrated 10× prefill speedup at 128K on RTX 3090 by combining drafter span-scoring with dense target prefill on the compressed prompt
**Related:** [031](031-vertical-slash-sparse-prefill.md) (composes), [033](033-block-sparse-sdpa-metal.md) (drafter's own prefill benefits)

## The insight

For long-context prompts, most input tokens contribute negligibly to the model's output. PFlash's measurement on 128K Qwen3.6-27B finds that a small drafter (Qwen3-0.6B) can identify which spans matter using attention-weight aggregation, then the target prefills only on the kept spans. At `keep_ratio=0.05`, 128K → 6.4K — the target's prefill cost drops by 20×, and end-to-end TTFT by ~10× (drafter cost included).

The technique is **algorithmically separable from the block-sparse kernel work**. The drafter's *own* prefill can use any sparse method (dense, vertical-slash, block-sparse — whatever's available); the target sees a normal short prompt and uses its existing dense prefill path. So this spec lands independently of spec 033 and benefits from spec 031 / 033 multiplicatively if either ships.

## Why this is the highest-leverage prefill spec

1. **No new Metal kernel.** Target prefill is unchanged dense SDPA on a shorter input.
2. **Composes with everything.** Vertical-slash on the kept 6.4K (spec 031); prefix cache on the kept spans (spec 017); TurboQuant compression on the resulting KV (already shipped); n-gram / DFlash speculative decode after prefill (specs 013, 015) — all unaffected by this spec.
3. **Reuses the drafter we already need for spec 015.** DFlash spec already plans `z-lab/Qwen3.5-*-DFlash` drafters. One drafter serves both span-scoring (this spec) and speculative decoding (spec 015). Sharing is strictly cheaper than two drafters.
4. **Apple Silicon unified-memory advantage.** PFlash on a 24GB 3090 has to park-and-restore drafter weights to fit the target. On Apple Silicon's unified memory there's no eviction — drafter stays resident, drafter KV from the scoring pass stays resident and is reused for speculative decoding (saves a re-prefill on the first decode token).

## Design

### Pipeline

```
Input: prompt P of length T (long, e.g. 64K-128K)
Output: compressed prompt P' of length T' = ceil(T · keep_ratio)

1. Drafter prefill:    drafter.forward(P) → drafter_attn_weights, drafter_kv_cache
2. Span scoring:       scores = aggregate(drafter_attn_weights)        # [T]
3. Selection:          keep_idx = top-k(scores, T') ∪ sink_indices ∪ tail_indices
4. Compression:        P' = P[keep_idx]   (token ids kept in original order)
5. Target prefill:     target.forward(P')  →  target_kv_cache
6. Generation:         iterator over target with drafter as spec-decode draft
                        — drafter_kv_cache is reused (rebased onto kept spans)
```

### Score aggregation

PFlash's "tail attention" scoring: for each token position `t`, score is the **max over (layer, head), mean over the last `tail_window` queries** of the drafter's attention weights at column `t`.

```
scores[t] = mean_{q ∈ [T - tail_window, T)} max_{l, h} attn[l, h, q, t]
```

Tail window default 256. Captures "which tokens does the most-recent context attend to most."

Alternatives to evaluate in phase 1:

- **PFlash's threshold-based selection** (`alpha · mean(scores)` threshold instead of top-k) — preserves variable keep ratios per chunk, more robust to score-distribution shape
- **MInference-style vertical-stripe detection** — pick top-k columns by global attention mass
- **Hidden-state cosine** (spec 019's selector protocol, reusable) — cheaper than attention scoring; should be benchmarked as a baseline

Default config: hybrid threshold `alpha=0.85` with floor `keep_ratio_min=0.03` and ceiling `keep_ratio_max=0.15`.

### Mandatory keep regions

Selection always includes:

- **Sink tokens:** first `s` tokens (default 4) — match attention-sink convention
- **Tail tokens:** last `tail_keep` tokens (default 256) — preserve the immediate context the user just typed
- **System / instruction span:** if the chat template marks a system prompt boundary, keep all of it

Span selection on the *user content body* only.

### Drafter KV cache reuse

The drafter does a full prefill on `P` to score. Two options post-scoring:

- **(A) Discard drafter KV** — matches PFlash exactly. Saves memory; on first decode, drafter must re-prefill on `P'`.
- **(B) Rebase drafter KV onto `P'`** — gather the drafter's KV at `keep_idx`, treat as the drafter's prefill output for `P'`. Saves ~one drafter forward at first decode. Numerically equivalent up to the drafter's own attention sparsity.

Default: **(B) on Apple Silicon** (unified memory makes the trade obvious); fall back to (A) if memory pressure detected.

### Position-id handling

After compression, target sees tokens `P[keep_idx]` but their original positions were sparse. Two strategies:

- **Renumber positions 0..T'-1** — what PFlash does. Loses absolute-position info but works on RoPE'd models because RoPE encodes relative position, not absolute.
- **Preserve original positions** — gather position embeddings at `keep_idx`. Strictly correct but breaks any model with absolute position embeddings.

Default: renumber. RoPE-only safety check at integration time. Models with absolute embeddings (rare in current zoo — none of Gemma4/Qwen3.5/Qwen3.6/GPT-OSS/Mistral/Nemotron-H) error out with a routing message.

### Drafter requirements

- Same tokenizer as target (or vocab-compatible — see spec 021's vocab gate)
- Attention-weight extraction supported (most models discard these in the SDPA fast path; needs a "score-mode" forward)
- Small enough to load alongside target — Qwen3-0.6B (~600MB Q4) or Qwen3-1.7B (~1.5GB Q4) for typical 27B-class targets

## Implementation phases

1. **Phase 1 — Score-mode drafter forward.** Add a `ScoringForward` protocol that returns attention weights aggregated at a configurable layer subset. Implement on Qwen3 model class. ~1 week. Goal: drafter produces per-token scores matching reference Python implementation within fp16 tolerance.

2. **Phase 2 — Span selector + compressed-prompt iterator.** Build `SpeculativePrefillSelector` (selects `keep_idx`), wire into a new `CompressedPromptTokenIterator` that runs drafter scoring → target prefill on selected spans → standard generation. ~1 week. Goal: end-to-end TTFT speedup measured at ≥ 32K context.

3. **Phase 3 — Drafter KV reuse.** Strategy (B) above. ~3 days. Goal: first decode token does not re-prefill the drafter.

4. **Phase 4 — Calibration & accuracy harness.** NIAH benchmark + `wikitext-perplexity-on-compressed-prompt` test. Sweep `(alpha, keep_ratio_min, keep_ratio_max)` per model pair. Emit per-pair config sidecar. ~1 week. Goal: shipped configs hit NIAH ≥ 95% retention at 128K.

5. **Phase 5 — Composition with spec 015 (DFlash decode).** Single drafter for both phases. ~3 days. Goal: zero-overhead drafter sharing.

6. **Phase 6 — Composition with spec 031 (vertical-slash) on the drafter's own prefill.** Drafter still has to prefill the long prompt; vertical-slash on the drafter cuts that cost too. ~3 days.

## Expected impact

| Context | Drafter cost | Target prefill (compressed) | TTFT speedup vs dense |
|---------|--------------|------------------------------|------------------------|
| 8K      | ~0.1s        | ~0.4s                        | 1.5–2× (overhead dominated) |
| 32K     | ~0.4s        | ~1s                          | 3–4× |
| 128K    | ~3s          | ~3s                          | 8–12× (matches PFlash claim) |

Composes multiplicatively with spec 031 (vertical-slash on the kept 6.4K shaves another 1.3–1.8×) and additively-then-multiplicatively with spec 033 (drafter's own prefill uses block-sparse).

## Risk register

1. **Quality regression on retrieval-heavy prompts.** PFlash's NIAH validation gates this — required acceptance criterion. Per-model `(alpha, keep_ratio)` calibration is mandatory.

2. **Drafter and target disagree on token importance.** Mitigation: oracle benchmark — compare drafter scores against target's own first-pass attention to measure score correlation. If correlation is low for a target / drafter pair, refuse to enable speculative prefill on that pair (route to dense).

3. **Score-mode forward breaks fused-attention paths.** Most production attention paths discard attention weights. Adding a score-mode adds complexity. Mitigation: only the drafter needs score-mode; target keeps its existing fast paths. Drafter is small, so the extra-allocation cost (storing T·H·L weights at fp16) is bounded — at 128K with H=14, L=28 on Qwen3-0.6B, that's ~6GB. Mitigation: stream-aggregate during the forward pass (running max + tail-window mean), never materialize the full tensor. Reduces to T·D'-sized scores buffer.

4. **Multi-turn prefix cache invalidation.** If the kept span set changes between turns, prefix cache (spec 017) keys may not match. Mitigation: hash the *uncompressed* prompt prefix as the cache key, not the compressed one. Selection is deterministic given drafter weights + prompt, so the same prefix produces the same compression.

## Acceptance criteria

- `MLX_SPECULATIVE_PREFILL=1` env var routing in `MLXLMCommon.generate(...)`
- ≥ 1 validated drafter/target pair (Qwen3-0.6B + Qwen3.5-9B canonical)
- TTFT speedup measured at 4K / 16K / 64K / 128K
- NIAH retention ≥ 95% at 128K
- PPL regression ≤ +0.5% on `wikitext-2`
- Per-prompt sweep harness extended with `--method spec-prefill`
- Documentation: `documentation/SPECULATIVE-PREFILL.md` covering pair selection, calibration, composition with spec 015

## What this spec deliberately does NOT do

- **No new Metal kernel.** Target prefill uses existing dense SDPA on a short input.
- **No replacement for spec 015.** That's the decode-side speculative iterator. This spec is prefill-side. They share a drafter.
- **No "draft-then-verify" prefill.** Compressed prompt is committed; we don't verify that the target would have produced the same output on the uncompressed prompt. Quality is gated by NIAH + PPL, not per-token verification. (A future "verified speculative prefill" spec could add per-block verification, but it would add cost roughly equal to dense prefill — defeats the purpose.)
