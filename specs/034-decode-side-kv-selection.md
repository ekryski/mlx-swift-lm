# 034 — Decode-side K-side top-k attention (Quest / RetrievalAttention)

- **Status:** spec, ready (depends on spec 033's kernel for the fused fast path)
- **Branch:** new branch off alpha
- **Depends on:** [033](033-block-sparse-sdpa-metal.md) phase 5 (kernel + Swift wrapper landed)
- **Origin:** Research review 2026-05-08; addresses gap surfaced during sparse-attention audit — TurboQuant Path B's sparse-V kernel covers V-side waste but K-side score compute is still O(T_kv) per decode step
- **Related:** Quest (MIT'24), RetrievalAttention (MSR'24), H2O (NeurIPS'23), Scissorhands, MagicPIG

## The insight

At decode time, attention is `Q ∈ [B, H, 1, D]` against the full stored cache `K, V ∈ [B, H, T_kv, D]`. We pay O(T_kv·D) per layer per token to compute scores, even though the post-softmax distribution is highly concentrated — typically <5% of slots get >95% of the attention mass. TurboQuant Path B's sparse-V kernel ([TurboQuantKernels.swift:1140](Libraries/MLXLMCommon/TurboQuantKernels.swift:1140)) exploits this **after** softmax to skip V-aggregation, but the score compute itself is still dense.

Quest, RetrievalAttention, H2O, and MagicPIG all attack the K-side: estimate which cache slots are likely to be heavy hitters using a cheap proxy (block summaries, LSH, recency+attention-history), select top-k slots, do exact SDPA against just those. At 128K context with k=2048, that's a 64× compute reduction on the score-and-aggregate, on top of the V-side savings already shipped.

## Why this matters specifically for our stack

- **At long context, decode tok/s is K-side-bound.** TurboQuant compresses the cache (memory bandwidth win), but the FLOP count of `Q · K^T` over 128K slots still dominates per-layer time. Compression doesn't help that — only sparsification does.
- **Cache-agnostic.** The technique applies to plain bf16 `StandardKVCache`, `AffineQuantizedKVCache`, and `TurboQuantizedKVCache` alike — only the block-summary computation differs per backend. V1 ships on `StandardKVCache`; TurboQuant Path B variant in phase 7 is an additional fast path, not a prerequisite.
- **Issue [#114](https://github.com/ekryski/mlx-swift-lm/issues/114) (long-context fallback to TurboFlash for large models)** is a related but coarser approach. This spec is the per-step-adaptive version.
- **Composes with TurboQuant Path B.** The selection picks `top_k` slot indices; TurboQuant's compressed-domain scoring runs over the gather of just those slots; the V-side sparse skipping kernel runs naturally on the smaller selected set. K-side compute + V-side aggregation + compressed bandwidth all stack.
- **Composes with spec 031/032.** Sparse-prefill keeps fewer KV slots; this spec keeps fewer of those at decode. Multiplicative.

## Design

### Selection protocol

```
selectTopK(query: Q, cache: KV, budget: k) -> [Int]   // slot indices, length k
```

Three implementations to evaluate, in order:

#### (1) Block-mean LSH (Quest-style)

Maintain per-block summaries: for each contiguous block of `B_k` slots (default 64), store `mean(K)` and `max(|K|)` per dimension. At decode: compute `Q · mean(K_block)` for all blocks, select the top-`k/B_k` blocks, then return all slots in those blocks. O(T_kv/B_k · D) per layer — **64× cheaper than dense scoring.**

Block summaries are computed once at prefill end (or incrementally during prefill), updated lazily as the cache grows during decode. Storage: `[B, H, T_kv/B_k, D]` per cache — for 128K context with B_k=64, that's 2K block summaries vs 128K full cache, ~1.6% overhead.

#### (2) Heavy-hitter retention (H2O-style)

Maintain a running "attention history" `score_history[t]` per slot, updated each decode step. Top-`k_h` heavy hitters by history are always retained; the remaining budget `k - k_h` goes to the most-recent `k - k_h` slots. No scoring kernel needed — pure bookkeeping.

This is cheaper than (1) but quality-regresses on tasks requiring rare-but-important context (e.g. NIAH). Use as an opportunistic prefilter to (1).

#### (3) Recency + attention-sinks baseline

Always keep first `s` sink slots + last `w` window slots. Equivalent to a static window cache but applied at compute time, not eviction time. Free baseline; everything must beat this on accuracy.

Default routing: (3) for context < 8K (no benefit); (1) for context ≥ 8K; (1) ⊕ (2) for context ≥ 32K with k_h adaptive.

### Sparse-SDPA call

Once `top_k_idx` is selected, call spec 033's `block_sparse_attention` with adjacency that selects exactly those slots' blocks, or — for the simpler V1 — call dense SDPA on the gathered tensors:

```
K_sel = K[..., top_k_idx, :]
V_sel = V[..., top_k_idx, :]
out = MLXFast.scaledDotProductAttention(queries=Q, keys=K_sel, values=V_sel, scale=...)
```

The gather costs O(k·D) memory bandwidth; SDPA costs O(k·D) FLOPs. Both dominated by the saved O(T_kv) work for `k << T_kv`.

V1 ships gather + dense; V2 uses spec 033's sparse kernel for fused execution.

### Block-summary maintenance

Three options for when block summaries are computed:

- **(A) At prefill end:** one-shot pass over the full cache. Adds ~0.5% to prefill time. Simple.
- **(B) Streamed during prefill:** update summaries as each chunk's KV is written. Zero extra prefill time but requires a hook into the prefill write path.
- **(C) Lazy at first decode:** compute summaries before the first decode step. Hidden in TTFT but adds visible latency to the first token.

Default: (A). (B) once spec 024 (KV-cache write fusion) lands and gives a clean hook point.

For decode steps: each new token appends one slot. Update the *latest* block's summary in-place; once the block fills (every B_k tokens), seal it and start a new block. Constant-time per decode token.

### Causal correctness at decode

At decode, `Q` is one query at the latest position. Causal mask is implicit (all stored slots are <= current position). Selection respects causality automatically.

### Quantized-cache compatibility

#### Affine-quantized cache

Compute block summaries on the dequantized K. Costs one per-block dequant per prefill end. Stored summaries are fp16, not quantized.

#### TurboQuant cache

Block summaries computed on the WHT-rotated, packed K. Direct from packed bytes — no dequant. Algorithmic detail: after WHT rotation, `mean(K_block)` is a meaningful summary because the rotation preserves dot products. Validated experimentally during phase 4.

### Per-head `k` budgeting

Different heads need different budgets — some are diffuse (need wide k), some are concentrated (k=64 suffices). Phase 4 adds per-head `k_h` based on entropy of the block-summary score distribution at calibration time. Default flat budget `k = max(2048, T_kv * 0.05)`.

## Implementation phases

1. **Phase 1 — Block-summary infrastructure.** Add `BlockSummaryCache` protocol; implement on `StandardKVCache` and `TurboQuantizedKVCache`. Compute at prefill end (option A). Test that summaries match brute-force reference within fp16 tolerance. ~1 week.

2. **Phase 2 — Selection + gather + dense SDPA decode path.** Add `TopKDecodeSelector` (block-mean LSH); wire into the attention call site as an alternative to dense SDPA, behind `MLX_TOPK_DECODE=1` env var. ~1 week. Goal: end-to-end decode working at 32K+ context with measurable speedup vs dense.

3. **Phase 3 — Quality gate + accuracy harness.** NIAH benchmark + perplexity sweep + per-step `top_k_idx` overlap with brute-force ground truth. Tune `(B_k, k)` defaults per model. ~1 week. Goal: NIAH ≥ 95% retention at all calibrated context lengths.

4. **Phase 4 — Per-head adaptive budget.** Calibration pass measuring per-head score entropy → per-head `k_h`. Stored in same sidecar as spec 031 patterns. ~1 week.

5. **Phase 5 — Heavy-hitter prefilter (H2O hybrid).** Maintain `score_history`; reserve fraction of budget for HH retention. ~1 week.

6. **Phase 6 — Spec 033 fused-sparse path.** Once spec 033 phase 5 lands, replace gather+dense with `block_sparse_attention`. ~3 days. Goal: eliminate gather memory bandwidth.

7. **Phase 7 — TurboQuant Path B integration.** Run selection on packed K; route selected slots through compressed-domain scoring. ~2 weeks.

8. **Phase 8 — Cross-model rollout + per-model calibration.** Gemma4 / Qwen3.5 / Qwen3.6 / GPT-OSS / Mistral / Nemotron-H. Per-model `(B_k, k, k_h)` recipes. ~1 week per model.

## Expected impact

On `StandardKVCache` (bf16, no TurboQuant):

| Context | k    | Score compute reduction | Decode tok/s lift |
|---------|------|-------------------------|--------------------|
| 8K      | 2048 | 4×                      | 1.1–1.3× (overhead-dominated)     |
| 32K     | 2048 | 16×                     | 1.5–2.5×                          |
| 128K    | 4096 | 32×                     | 2.5–4×                            |
| 256K    | 4096 | 64×                     | 3–5×                              |

Compositionally on top of TurboQuant Path B: bandwidth + compute both sparsified, projected combined long-context decode lift of **5–8× at 128K** vs current dense Path B baseline.

## Risk register

1. **Block-mean is a poor proxy for some heads.** Mitigation: per-head fallback to dense (calibration gate; if entropy too low, that head can't be sparsified). Same per-head dispatch infrastructure as spec 031.

2. **WHT rotation preservation of dot products is approximate, not exact.** TurboQuant variant (phase 7) needs validation. Mitigation: phase 7 includes explicit numerical correctness check vs unrotated reference; if drift exceeds threshold, fall back to dequant-summary path.

3. **Selection latency exceeds saved compute at small T_kv.** Mitigation: dispatch threshold; use dense SDPA below `T_kv < 4096`. Threshold is per-shape, calibrated.

4. **Quality cliff on retrieval-heavy tasks.** Same NIAH gate as specs 030/031. Per-model k tuning. If a model can't pass NIAH at k=4096 by 128K, that model is routed to dense decode at long context.

5. **Decode-time per-step selection adds CPU↔GPU sync points.** Selection runs on GPU (one matmul + topk); the resulting indices feed gather. No CPU sync needed if topk indices stay on-device. Mitigation: ensure topk runs as a Metal kernel returning a GPU tensor, not via `.asArray()`.

## Acceptance criteria

- `MLX_TOPK_DECODE=1` env var routing
- Block-mean selection working on `StandardKVCache` and `TurboQuantizedKVCache`
- Decode tok/s lift measured at 8K / 32K / 128K
- NIAH retention ≥ 95% at 128K with default config
- PPL regression ≤ +0.5% on `wikitext-2`
- TurboQuant Path B variant landed and tested
- Documentation: `documentation/TOPK-DECODE.md` covering selection algorithms, per-model recipes, composition with TurboQuant

## What this spec deliberately does NOT do

- **No replacement for cache eviction.** Window-cache and prefix-cache are orthogonal — both keep fewer slots in the cache; this spec keeps fewer of the slots that *are* in the cache from being attended to.
- **No replacement for TurboQuant.** Compression and selection are complementary axes.
- **No prefill-time top-k.** Prefill is multi-query; spec 031 / 033 handle that.
