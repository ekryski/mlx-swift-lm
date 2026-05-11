# 035 — Quest: page-bounded top-k attention over the KV cache

**Status:** spec, parked behind spec [034](034-decode-side-kv-selection.md) (positioned as a refinement of 034's selector framework; ship 034 V1 first, evaluate this against the measured NIAH curve)
**Branch:** new branch off alpha
**Depends on:** [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) Metal paged-attention kernel for `PagedKVCache`, [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) wire `PagedKVCache` into model factories. Composes with [#129](https://github.com/ekryski/mlx-swift-lm/issues/129) TurboQuant + paged integration and PR #186 windowed eviction. **Selector framework reuses** [spec 034](034-decode-side-kv-selection.md) phases 1–3.
**Origin:** [`papers/beyond-quadratic-attention-on-apple-silicon.md`](../papers/beyond-quadratic-attention-on-apple-silicon.md) §3.2; [Quest, MIT, ICML 2024 (arXiv 2406.10774)](https://arxiv.org/abs/2406.10774); [reference implementation](https://github.com/mit-han-lab/Quest)

## Relationship to spec 034

Spec [034](034-decode-side-kv-selection.md) (landed 2026-05-09) is the umbrella K-side top-k decode framework, with three candidate selectors: (1) **block-mean LSH** (mean(K) + max(|K|) per block), (2) heavy-hitter retention (H2O-style), (3) recency + sinks baseline.

This spec is the **original Quest paper's selector formulation** — the elementwise **K_max / K_min upper-bound** — which is a *fourth* selector for 034's framework, distinct from 034's "block-mean LSH (Quest-style)" variant. The two are related but mathematically different:

| Selector | Per-page metadata | Score | What it estimates |
|---|---|---|---|
| 034 block-mean LSH | `mean(K)`, `max(\|K\|)` | `Q · mean(K_page)` | Approximate *average* attention score per page |
| **035 K_max/K_min** | `K_max`, `K_min` (elementwise over the page) | `max(q·K_max, q·K_min)` | **Provable upper bound** on `max_t (q · K[p,t])` |

K_max/K_min gives a tighter bound at the cost of 2× more page metadata (~6% of K-cache footprint vs ~3%). The expected payoff is **better NIAH retention at smaller k** — i.e. you can keep fewer pages active per query for the same retrieval fidelity. This matters for very-long-context workloads where the K-side compute budget is the binding constraint.

**Ship order:** 034 V1 (block-mean LSH) first, measure NIAH curves, then evaluate whether K_max/K_min's tighter bound justifies the storage + metadata maintenance overhead. If 034 already hits the NIAH target at the operating point we care about, this spec stays parked. The two specs share 034 phases 1–3 (block-summary infrastructure, selector dispatch, NIAH harness); 035 only differs in the selector kernel itself.

## The insight

Decode-time attention is bandwidth-bound on Apple Silicon: at 32 K context the KV-cache reads compete with the weight stream, dropping Qwen 3.5-9B-4bit from ~54 to ~41 tok/s ([speculative-decoding-on-apple-silicon.md §1](../papers/speculative-decoding-on-apple-silicon.md)). Past 64 K it's the dominant cost.

But for any given query, **most KV pages contribute negligibly to the attention output.** If we knew per-page upper bounds on the attention score, we could load only the top-k pages and run exact attention over them. That's Quest:

1. Per page `p` of length `P` (typically 16–64 tokens), maintain two small vectors alongside the K matrix: `K_max[p]` and `K_min[p]`, the elementwise max and min of the page's keys across the token axis.
2. At decode time, compute an upper bound on the maximum attention score for each page from the query `q`: `s_max[p] = max(q · K_max[p], q · K_min[p])` (taking the elementwise max of `q*K_max` and `q*K_min`, then summing along head-dim). This is provably an upper bound on `max_t (q · K[p,t])` for any token in the page.
3. Sort pages by `s_max[p]` descending; pick top-k.
4. Run exact attention over the selected pages only.

This is **lossless attention over a learned-at-runtime subset**: no quantization noise, no eviction, no retraining. The full KV cache stays in unified memory; we just don't *read* most of it per query.

The published number is ~2.2× attention speedup with ~7× end-to-end at 32 K context on a server GPU. On M-series the win should be larger relative to baseline because we're already further off the bandwidth ceiling at long context.

## Why this fits Apple Silicon specifically

Three reasons Quest is a near-perfect fit for the M-series:

1. **Pure bandwidth saving.** No new FLOPs, no new dispatches per page; just fewer page reads per query. Apple GPUs are bandwidth-bound at decode — bandwidth saved is throughput gained, almost 1:1.
2. **Block-aligned.** Page sizes of 16–64 tokens match Metal threadgroup tiles and `simdgroup_async_copy` granularity. Unlike token-level sparsity (which kills coalesced loads), page-level top-k preserves the contiguous-read pattern Metal kernels are tuned for.
3. **Metadata cost is tiny.** `K_max[p]` and `K_min[p]` are each `[H, D]` per page — for a page size of 32 tokens, they add ~6% to the K-cache footprint. Trivially absorbable in unified memory.

## What this composes with

- **PR #186 windowed eviction.** Quest is orthogonal: windowed eviction caps the cache size; Quest reduces reads-per-query *within* the (capped) cache. They stack.
- **TurboQuant Int4/Int8 (`AffineQuantizedKVCache`, `TurboQuantizedKVCache`).** `K_max`/`K_min` are computed in fp16 over the dequantized page on first write; selection runs in fp16; final attention runs over the (still-quantized) selected pages. Composes cleanly.
- **`PagedKVCache`.** This spec assumes paged is the cache layout Quest plugs into. Implementing Quest on top of a non-paged contiguous cache is possible (synthetic page boundaries) but defeats the cache-line-aligned read advantage.
- **Spec 020 tape-replay rollback.** Quest is per-query state — the `K_max`/`K_min` metadata is a write-time artifact, not a read-time one — so spec-decode rollback doesn't need to touch it.
- **Reasoning models** (the LazyEviction/ForesightKV failure mode of greedy eviction). Quest avoids the failure mode by construction: nothing is evicted, late re-attention to early CoT tokens still finds them in the top-k for the right query.

## What this does NOT compose with (yet)

- **DuoAttention (spec 036).** DuoAttention's "streaming" heads use a sink+window cache, not a paged full cache. Quest only applies to the "retrieval" heads. The two specs partition the attention layers.
- **Native sparse attention (NSA / DSA / MoBA).** Models that already learned a sparse-attention pattern at pretraining time don't benefit — the indexer or routing gate is already doing per-layer Quest.
- **Pure-Mamba / pure-GDN layers.** Quest is for softmax attention over an explicit KV cache. SSM/linear-attention layers don't have one to score against.

## Design

### Phase 1 — `K_max`/`K_min` page metadata on `PagedKVCache`

Add two MLXArrays alongside each page's K:

```swift
public class PagedKVCache: BaseKVCache {
    public var pages: [Page]
    public struct Page {
        public var keys: MLXArray       // [H, P, D]
        public var values: MLXArray     // [H, P, D]
        public var keyMax: MLXArray     // [H, D] — per-head per-dim max over P tokens
        public var keyMin: MLXArray     // [H, D] — per-head per-dim min over P tokens
        public var occupancy: Int       // 0..P
    }
}
```

`keyMax`/`keyMin` are updated incrementally on every K write (cheap reduction over `P`). On a full page they're frozen until the page is reused (e.g., after a windowed eviction).

**Footprint:** for `P=32`, `H=32`, `D=128` (Qwen 3.5-9B-class), `keyMax+keyMin` is `2 * 32 * 128 * 2 bytes = 16 KB` per page. A 32 K-token cache at `P=32` is 1024 pages → 16 MB metadata. Cache itself is ~537 MB (per the M1 Max bench). Metadata is ~3% of cache.

### Phase 2 — Page scoring kernel

Per-decode-query computation:

```
score[p] = sum_d max(q[d] * keyMax[p, h, d], q[d] * keyMin[p, h, d])  for each head h
score[p] = max_h score[p, h]    // worst-case across heads
```

Implementation options:

- **(a) MLX-only**, using `maximum(q * keyMax, q * keyMin).sum(axis: -1).max(axis: 1)`. Cheap to land; one reduction; trades dispatches for kernel work.
- **(b) Custom Metal kernel**, fused with the page-selection top-k. Single dispatch, cache-line coalesced.

Phase 2 ships (a). Phase 5 (optional) replaces with (b) if measurement shows the dispatch overhead matters.

### Phase 3 — Top-k page selection + sparse attention dispatch

Decode-time path:

```swift
// 1. Score all pages
let scores = scorePages(query: q, cache: paged)   // [numPages]
// 2. Pick top-k indices
let selected = topK(scores, k: questBudget)       // [k]
// 3. Gather selected pages into contiguous K, V tensors
let (sparseK, sparseV) = gatherPages(paged, indices: selected)
// 4. Run exact attention over the gather
let out = MLXFast.scaledDotProductAttention(
    queries: q, keys: sparseK, values: sparseV,
    scale: scale, mask: .none)
```

`questBudget` is configurable per-model. Typical: `k = max(64, min(numPages, ceil(0.1 * numPages)))` — cap at 10% of pages plus a floor of 64 pages (≈2 K tokens) to preserve recall on short-distance dependencies.

The "always-include" set: **always include the most recent N pages** (sliding window) and **the first M pages** (attention sink), regardless of score. This matches the empirical finding that recent + sink tokens are load-bearing for any query.

### Phase 4 — Per-model integration + budget tuning

Wire Quest into the attention call site of the priority-1 models (Qwen 3.5, Gemma 4 26B-A4B, GPT-OSS-20B). Budget tuning per model on RULER + LongBench:

| Model | Default `k` (% of pages) | Floor pages | Sink pages | Window pages |
|---|---|---|---|---|
| Qwen 3.5-9B | 10% | 64 | 4 | 32 |
| Qwen 3.5-26B | 8% | 96 | 4 | 64 |
| Gemma 4 26B-A4B | 12% | 64 | 4 | 32 |
| GPT-OSS-20B | 10% | 64 | 4 | 32 |

These are starting points for the bench sweep — final values come from Phase 4's RULER/needle-in-haystack matrix.

### Phase 5 — Optional fused page-score + top-k Metal kernel

If Phase 4 shows the page-scoring + gather overhead exceeds ~1.5% of decode-token cost, write a fused Metal kernel that computes scores, runs top-k via partial bitonic sort, and emits the gather indices in a single dispatch. Otherwise skip.

## Implementation phases

1. **Phase 1 — Page metadata.** `PagedKVCache.Page.keyMax/keyMin` + incremental update on K write. Composes with TurboQuant by computing on dequantized fp16 K. Land behind `MLX_QUEST_METADATA=1` so it's free until enabled. ~120 lines in `PagedKVCache.swift` + tests.

2. **Phase 2 — MLX-only page scorer + selector.** `Quest.swift` in `Libraries/MLXLMCommon/` with `scorePages(query:cache:)` and `selectPages(scores:budget:sinks:window:)`. Pure MLX ops. ~150 lines + unit tests against a brute-force oracle (compute true page-max attention scores; check Quest's ranking is a superset of the true top-k for at least 95% of queries on a synthetic test).

3. **Phase 3 — Sparse attention dispatch.** Gather + SDPA wiring in a new `MLXFast` extension or `attention/QuestAttention.swift`. ~80 lines. Unit test: compare full-attention output vs Quest output at `budget = 100%`; must be bit-exact.

4. **Phase 4 — Per-model integration + budget tuning.** Wire into Qwen 3.5, Gemma 4 26B-A4B, GPT-OSS-20B attention call sites behind a `--quest-budget` CLI flag. Run RULER + LongBench at budgets 5/10/20/50/100% for each model. Document in `benchmarks/notes/spec-035-quest-2026-MM-DD.md`. ~300 lines across model files + bench harness.

5. **Phase 5 (optional) — Fused Metal kernel.** Only if Phase 4's profile says it's worth it. ~400 lines of Metal + ~50 lines of MLXFast wiring.

## Expected impact

- **Decode tok/s at long context.** At 32 K with Quest budget 10%, projected **+30–50% decode** over baseline (we go from reading 100% of pages to reading ~10% of pages; the per-query bandwidth saving dominates). At 64 K, projected **+50–80%**. At 8 K, near-zero (cache reads are <10% of decode cost).
- **Quality.** Lossless in expectation when budget is well-tuned. Per the Quest paper, RULER 32 K perplexity matches full attention at budget 10%; at budget 5% degrades on multi-key needle-in-haystack.
- **Memory.** +3% on KV-cache footprint for the metadata. No reduction (this is a *read* sparsifier, not an eviction).
- **No prefill win.** Quest is decode-only — at prefill we don't have the per-query selection signal yet, and FlashAttention's sequential read is already O(N) bandwidth-optimal. Prefill stays on the standard SDPA path.

## Risks

1. **Page-scoring quality vs truly random K distributions.** The `K_max/K_min` upper bound is tight when K values within a page are correlated (which empirically holds — adjacent tokens have similar K), but loose when not. If a model's K layer mixes across positions in unusual ways (some Gemma 4 sliding-window layers do), the selector may miss the true top-k. Mitigation: Phase 2's superset test catches this per model; Phase 4's RULER sweep is the deployment gate.

2. **Page-size sensitivity.** Smaller pages (`P=16`) give tighter bounds but more metadata + more selection overhead; larger (`P=64`) the reverse. Default `P=32` matches the existing `PagedKVCache` page size from #127.

3. **Reasoning chains and the always-include-recent window.** Reasoning models re-attend to far-back CoT tokens that aren't in the recent window. The always-include set doesn't help; we rely on the score-based selection to surface them. If a CoT token's K signature isn't a strong match for the current query (early-exposition tokens often aren't), Quest may miss it. Mitigation: in Phase 4 specifically run reasoning prompts (math, multi-hop QA) at the candidate budgets; budget for reasoning models may need to be 15–20% rather than 10%.

4. **Composition with windowed eviction.** PR #186's windowed eviction means pages get rotated/overwritten. Page metadata must be reset on eviction — straightforward, but easy to forget and expensive to debug (silent quality regression).

5. **TurboQuant interaction.** `K_max/K_min` need to live in fp16 for selection precision; computing them from quantized Int4 K loses bound tightness. Solution: cache the fp16 reduction at write time (when K is still fp16 pre-quantization), not at read time. ~no compute cost.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/PagedKVCache.swift` | `Page.keyMax/keyMin` + incremental update on `update(...)` |
| `Libraries/MLXLMCommon/Quest.swift` (new) | `scorePages`, `selectPages`, `gatherPages` |
| `Libraries/MLXLMCommon/QuestConfig.swift` (new) | Per-model defaults, sink/window/budget |
| `Libraries/MLXLMCommon/MLXFast+Quest.swift` (new) | `scaledDotProductAttentionQuest(...)` extension |
| `Libraries/MLXLLM/Models/Qwen35.swift` | Attention call site → Quest path when `paged + quest` |
| `Libraries/MLXLLM/Models/Gemma4.swift` | Same |
| `Libraries/MLXLLM/Models/GPTOSS.swift` | Same |
| `scripts/benchmark.sh` | `--quest-budget <fraction>` flag |
| `Tests/MLXLMCommonTests/QuestTests.swift` (new) | Bit-exact at budget=100%; superset oracle test |
| `benchmarks/notes/spec-035-quest-2026-MM-DD.md` (new) | Phase 4 sweep results |

## Why this is Tier 3-adjacent (not Tier 4)

Two reasons it ranks above 027/028 in the IMPLEMENTATION-PLAN.md ordering:

1. **Universal at long context.** Every model with softmax attention benefits, not per-model effort. The per-model wiring is one call-site change; the bulk of the work is the kernel/selector and lives in `MLXLMCommon`.
2. **Scales with the workload that hurts most.** The win is largest exactly where current decode degrades: 32 K – 128 K context. That's the regime where TTFT and tok/s both fall off, and the regime where document QA / agent loops / RAG actually live.

It sits behind Tier 1 (#5 tape-replay, #6 prefix cache) and #127–#129 paged KV because it depends on paged KV being the cache layout. Once paged KV ships, Quest is a 1–2 week project for the headline phases (1–4) and a clear measurable win.

The `PagedKVCache` infrastructure work (#127–#129) is the long pole; Quest is "what you do with paged KV once you have it."
