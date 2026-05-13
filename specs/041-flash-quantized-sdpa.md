# 041 — Flash-style quantized SDPA

**Status:** Spec drafted 2026-05-12. **Not started.**
**Branch:** TBD (`ek/041-flash-quantized-sdpa-phase1` once implementation begins)
**Depends on:** none (independent kernel work). Touches the four-repo chain (`mlx` kernel + C++ Primitive → `mlx-c` ABI → `mlx-swift` wrapper → `mlx-swift-lm` callsite).

## Problem

`AffineQuantizedKVCache`'s SDPA path (`quantizedScaledDotProductAttention` in `Libraries/MLXLMCommon/KVCache.swift`) materialises the full `[B, H, L, T]` attention-score tensor in GPU memory:

```swift
var scores = quantizedMM(
    scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
    transpose: true, ...)
// ...mask application via MLX.where...
let attentionWeights = softmax(scores, axis: -1)
var output = quantizedMM(
    attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
    transpose: false, ...)
```

For a prefill chunk of `L = 1024` against a `T = 32k` cache the scores matrix is `1 × 16 × 1024 × 32768` bf16 ≈ **1 GiB per layer**. Across ~30 attention layers this is the dominant peak-GPU contributor on `--kv affine4` long-context summarization, even after the fp32-upcast and `expandQuant`-eval fixes shipped in the same PR as this spec landed. Measured on M1 Max 64 GB (post-fix):

| Model | ctx | no-quant peak | affine4 peak | gap |
|---|---|---|---|---|
| Qwen3.5-0.8B | 32k | 1.97 GiB | 4.03 GiB | +2.06 GiB |
| Qwen3.5-2B   | 32k | 2.52 GiB | 3.79 GiB | +1.27 GiB |
| Qwen3.5-9B   | 32k | 6.91 GiB | 8.81 GiB | +1.90 GiB |

The no-quant path uses `MLXFast.scaledDotProductAttention` — Apple's Flash-style fused kernel that tiles `K`/`V` and runs an online softmax, never materialising the full `[L, T]` score tensor. The affine path falls back to discrete `quantizedMM → softmax → quantizedMM` because today's `MLXFast.scaledDotProductAttention` doesn't accept quantized K/V tuples.

**Goal:** match Flash's tiling + online-softmax pattern for quantized K/V. Drop the `+1–2 GiB` gap on affine prefill at long context without sacrificing decode correctness or numeric quality.

## Scope: which quantization schemes does this apply to?

**Affine quantization is the only path that needs new kernel work.** Inventory:

| Path | L = 1 (decode) | L > 1 (prefill) | Needs flash-quant kernel? |
|---|---|---|---|
| `--kv none` (`StandardKVCache`) | `MLXFast.scaledDotProductAttention(... sinks:)` | same | No — already Flash |
| `--kv turbo*` raw-key + compressed-V (path A) | `MLXFast.scaledDotProductAttention(... sinks:)` after dequant-to-FP16 | same | No — same Flash kernel |
| `--kv turbo*` compressed K + V (path B, `useCompressedAttention=true`) | `TurboQuantizedKVCache.compressedAttention` — TurboFlash fused (single Metal dispatch, online softmax) | `compressedAttention(L>1)` — discrete score → softmax → value, **same shape** as the affine path | **Partly** — turbo L>1 currently materialises scores too, but its dispatcher routes L>1 prefill to the raw path (`update + standard SDPA`), so it never materialises scores in practice. Compressed-domain L>1 only fires when a hydrated compressed snapshot enters (spec 039), at which point the same online-softmax tile pattern this spec adds would apply. |
| `--kv affine*` (`AffineQuantizedKVCache`) | discrete `quantizedMM → softmax → quantizedMM` (the failure mode this spec targets) | same | **Yes** — primary target |

So the new kernel covers two cases under one design:

1. **AffineQuantizedKVCache (any L).** Replaces today's discrete-pass `quantizedScaledDotProductAttention`.
2. **TurboQuantizedKVCache compressed-domain L > 1 (post-spec-039 only).** Replaces the `compressedAttention(L>1)` discrete fallback when a compressed snapshot was loaded.

Both consume per-group `(packedIndices, scales, biases?)` triples for K and V; only the codec (affine vs. TurboQuant's MSE) and the packing layout differ. The kernel takes the dequant step as a callback / kernel-time template parameter and tiles the same fused-softmax loop downstream.

## Why it's a bigger change

`MLXFast.scaledDotProductAttention`'s implementation lives in the upstream `mlx-explore/mlx` repo as a Metal kernel that assumes raw FP16 / bf16 K and V tensors. To support quantized K/V we need:

- A new kernel (or kernel variant) that dequants K and V on-the-fly inside the tiled SDPA loop, computing scores against the dequanted K tile and accumulating against the dequanted V tile.
- A C++ `Primitive` to expose it through MLX's eval graph.
- A C ABI in `mlx-c` to call it from Swift.
- A Swift wrapper in `mlx-swift` (`MLXFast.quantizedScaledDotProductAttention` or similar).
- A Swift-side callsite swap in `mlx-swift-lm` (`Libraries/MLXLMCommon/KVCache.swift` `quantizedScaledDotProductAttention(...)` body + the post-spec-039 turbo L > 1 dispatcher branch).

Apple's official Metal-Performance-Shaders SDPA + the in-tree fused SDPA both already have a tiled-softmax skeleton; the addition is the dequant-fused tile load. Reference for the existing MSP-SDPA tile-load shape lives in `mlx-explore/mlx/mlx/backend/metal/kernels/scaled_dot_product_attention.metal`. Reference for a dequant-fused quantized matmul tile lives in the same repo's `quantized.metal` (`qmm_t_*` / `qmv_t_*` kernels) — those already do per-group dequant inside the tile and accumulate against an FP register.

The merge of the two patterns is the load-bearing kernel work. Roughly:

```metal
// Per query tile, for each KV block:
//   1. Dequant K block from (packed, scales, biases) → FP16 register tile.
//   2. Q · K^T into score tile (FP register accumulator).
//   3. Subtract running max + exp into a softmax-update step.
//   4. Dequant V block from its packed/scales/biases.
//   5. (softmax_weights · V) into the output accumulator.
//   6. Normalise at end of KV traversal.
```

Group-size + bit-width-specific kernel instantiations (the same template fan-out the existing `qmv_t_64_4` / `qmv_t_64_8` / etc. emit) cover the affine-cache shapes that matter: `groupSize ∈ {32, 64}`, `bits ∈ {2, 3, 4, 6, 8}`. For mxfp4 / nvfp4, add a parallel `qmv_t_64_4_mxfp4` instantiation that consumes packed nibbles directly (no scales / biases).

## Phasing

### Phase 1 — minimum-viable kernel (affine 4-bit / 8-bit, groupSize=64, GQA, sliding-window mask)

**Status:** Phase 1 stop-gap shipped 2026-05-13 via the **dequant-then-MLXFastSDPA** path (see `flashQuantizedScaledDotProductAttention` in `Libraries/MLXLMCommon/KVCache.swift`). Phases 1, 2, and 4 all pass through the same call:

- **Phase 1** (4-bit / 8-bit): MLX's `dequantized()` + `MLXFast.scaledDotProductAttention(...)` cover this shape.
- **Phase 2** (2-/3-/6-bit / mxfp4): comes free — `dequantized()` already supports `bits ∈ {2,3,4,5,6,8}` and mxfp4.
- **Phase 4** (sinks): `MLXFast.scaledDotProductAttention(... sinks:)` folds them natively. No extra fold needed in the affine path under the flash strategy.

**Auto-strategy selection**: `quantizedScaledDotProductAttention` routes `L > 1` (prefill) through the flash path and `L = 1` (decode) through the discrete `quantizedMM → softmax → quantizedMM` path. Empirical: flash dominates prefill (no [B, H, L, T] score materialisation = -42% peak GPU on GPT-OSS-20B 8k, -60% on Gemma 4 31B 8k); discrete dominates decode at long context because dequanting K/V per step costs ~15% decode tok/s. Auto-strategy yields the best of both. Override via `MLX_AFFINE_SDPA={auto,flash,discrete}`.

**Phase 1 acceptance gate measurements** (M1 Max 64 GB, today):

| Model | ctx | no-quant peak | affine4 peak (before) | affine4 peak (after) | Δ |
|---|---:|---:|---:|---:|---:|
| Qwen 3.5-0.8B | 32k | 1.97 GiB | 4.03 GiB | **1.68 GiB** | **-58%** (below no-quant) |
| Qwen 3.5-9B | 32k | 6.91 GiB | 8.81 GiB | **6.21 GiB** | **-30%** (below no-quant) |
| GPT-OSS-20B | 8k | 11.94 GiB | 20.41 GiB | **11.85 GiB** | **-42%** (below no-quant) |
| Gemma 4-31B | 8k | 28.10 GiB | 33.43 GiB (gibberish) | **27.52 GiB coherent** | **-18%** (below no-quant) |
| Gemma 4-E2B | 8k | 4.51 GiB | 5.05 GiB | 5.47 GiB | acceptable |

Decode tok/s (auto-strategy → uses discrete for decode): within 5% of baseline on the smoke matrix; GPT-OSS-20B 8k 60.7 → 62.2 tok/s, Gemma 4-31B 8k 7.0 → 12.9 tok/s.

**Fused-kernel optimisation (deferred to Phase 1.1)**: a tile-wise dequant Metal kernel that never materialises the full K_fp / V_fp transient would save the per-call ~K_fp + V_fp ≈ 100-400 MiB at long context. Profitable specifically on small-model long-context shapes where the dequant transient becomes a meaningful fraction of total. The kernel-level optimisations below (`simdgroup_matrix_multiply_accumulate`, threadgroup codebook, INT8 score) all apply to that future path.

Original Phase 1 design notes follow — still relevant for the fused-kernel follow-up:

- mlx: new `flash_quantized_sdpa.metal` template + C++ `Primitive` for `bits ∈ {4, 8}`, `groupSize = 64`, causal + sliding-window mask modes, GQA + MQA.
- mlx-c: ABI: `mlx_flash_quantized_sdpa(queries, kq, ks, kb, vq, vs, vb, scale, mask_mode, window, group_size, bits, out)`.
- mlx-swift: `MLXFast.quantizedScaledDotProductAttention(queries:quantizedKeys:quantizedValues:scale:mask:groupSize:bits:)`.
- mlx-swift-lm: route `quantizedScaledDotProductAttention` (`KVCache.swift`) through the new wrapper when shapes are in-range; fall back to the discrete path otherwise.

**Kernel-level optimisations to land in Phase 1** (lessons from `bulkDequantRotated` + `turbo_flash_attention.metal` perf today):

- **`simdgroup_matrix_multiply_accumulate` for Q·K^T and weights·V.** Apple's `MLXFast.scaledDotProductAttention` uses the matrix-engine 8×8 tile MMA intrinsics; today's `turbo_flash_attention.metal` uses plain `float4` SIMD ops, costing ~8× compute density per cycle. The dequant prologue eats register pressure, so the tile size will likely shrink from Apple's default (`Bq = 32, Bk = 32`) to e.g. `Bq = 16, Bk = 16` — the matrix-engine pipeline survives.
- **Codebook in threadgroup memory.** The 4-bit codebook is 16 entries × 2 bytes = 32 bytes; the 8-bit is 256 × 2 = 512 bytes. Hoist into `threadgroup` storage at kernel start so each thread reads from L1 instead of device memory. Cuts a 32k × `codebook_bytes` device-memory read per layer down to one threadgroup load.
- **Mixed-precision Q·K scoring (INT8 accumulator).** Apple's GPU has fast 8-bit MMA paths. Softmax compresses dynamic range to ~6 bits effectively, so Q · codebook[K_index] computed at INT8 → FP16 accumulator gives 1.5–2× compute density vs FP16 scoring without measurable PPL impact. Add a `scoreDtype` template parameter (`.int8` vs `.fp16`) and benchmark per-shape.
- **Pre-folded V codec rotation.** `valueMSECodec.rotation` is constant per (dim, bits, seed). Pre-multiply it into the model's `o_proj` weights at `sanitize(...)` time so the kernel's output is already in original V-space — saves one matmul per layer per token. Tracked in [issue #118](https://github.com/ekryski/mlx-swift-lm/issues/118).
- **Pre-warmed pipeline state.** All `(bits, groupSize, headDim)` kernel instantiations get pre-compiled at model load (extending today's `getOrCreateCodec` warm-up). Shifts ~50–100 ms of JIT compile cost out of TTFT.
- **Speculative-decode amortisation.** When n-gram speculative + flash-quantised SDPA combine, the K candidate tokens process in parallel through one SDPA call (instead of K serial calls). Per-token dequant work amortises across the batch; this is the same matrix-engine path Apple uses for `MLXFast.scaledDotProductAttention(L>1)`.

**Acceptance gate:** identical PPL/KLD to the discrete path (within fp16 rounding noise — < 1% relative on WikiText-2 word-level PPL), peak GPU at 32k drops to within 200 MiB of `--kv none` baseline across the smoke matrix below, decode tok/s within 10% of `--kv none` on `gemma4-31b` × `--kv turbo4v2` × ctx 32k (today's compromise route via TurboFlash hits ~5.6 tok/s vs no-quant 11.2 tok/s).

### Phase 2 — broaden bit-widths + mxfp4

**Status:** Shipped 2026-05-13 (for free, as a consequence of Phase 1's dequant-then-SDPA strategy). MLX's `dequantized(...)` already supports `bits ∈ {2,3,4,5,6,8}` and mxfp4, so the flash path inherits all of them without any kernel work. Affine2 / affine6 / mxfp4 routes through the same `flashQuantizedScaledDotProductAttention` as affine4 / affine8.

When the fused-kernel optimisation (Phase 1.1) lands, Phase 2's bit-width work transfers there as kernel instantiations.

### Phase 3 — TurboQuant compressed-domain L > 1 hookup (spec 039 follow-up)

**Status:** Deferred until spec 039 lands. Today's `attentionWithCacheUpdate` (`Libraries/MLXLMCommon/AttentionUtils.swift:71-81`) routes `L > 1` on a TurboQuant cache through the raw FP16 prefix prefill path (`turboCache.update(...) → MLXFast.scaledDotProductAttention`) — `compressedAttention(L>1)` is never called in production today. Spec 039 introduces compressed prefix-cache hydration which would land the cache in compressed mode at the start of a warm-turn suffix prefill; only then does the L>1 compressed dispatch fire.

When that callsite exists, it can route through `flashQuantizedScaledDotProductAttention` via the same dequant-then-SDPA shape Phase 1 uses (TurboQuant's MSE codec dequant — `bulkDequantRotated` — is already a single Metal op). The fused-kernel optimisation tracked as Phase 1.1 would replace the dequant-then-SDPA shape there too.

### Phase 4 — attention-sinks support (spec 6c crossover)

**Status:** Shipped 2026-05-13 (for free under Phase 1's flash strategy). `MLXFast.scaledDotProductAttention(... sinks:)` natively folds per-head sink logits into the online softmax — the flash path passes `sinks` through transparently. The explicit augmented-softmax sinks fold on the discrete path (added in PR #210) is retained as the fallback for `L = 1` decode where the auto-strategy picks discrete. GPT-OSS-20B `--kv affine4` works coherently end-to-end as a result; see the Phase 1 acceptance table above.

When Phase 1.1's fused-kernel optimisation lands, the in-kernel softmax loop will fold sinks the same way — see the spec's original Phase 4 description below for the math reference.

### Phase 5 — KV-sharing reader path (Gemma 4 E2B / E4B fallback recovery)

**Status:** Phase 5 (Gemma 4 LLM) shipped 2026-05-13. Discrete-path
`quantizedScaledDotProductAttention` is used as the stop-gap; the flash
kernel from Phase 1 will replace it without changing the reader API.
Gemma 3n and Gemma 4 VLM still pass `forceRawKV: true` and fall back to
`StandardKVCache` on affine donors — extending the reader path to those
files is straightforward (mechanical mirror of the Gemma 4 LLM diff)
and tracked alongside Phase 1.

Closes the `--kv affine*` fallback that `makeAttentionCache` currently routes through `StandardKVCache` whenever a layer is flagged `architecturalSlidingWindow` or `forceRawKV` (KV-sharing donor). Bug history + current fallback semantics tracked in [#202](https://github.com/ekryski/mlx-swift-lm/issues/202).

The fallback exists because today's shared-layer attention path in `Gemma4.swift` (and similarly in `Gemma3nText.swift`) consumes `sharedKVArrays: (MLXArray, MLXArray)` — raw FP16 K/V — and routes them straight to `MLXFast.scaledDotProductAttention`. Donor caches built as `AffineQuantizedKVCache` store `(packedIndices, scales, biases)` triples, which the shared reader has no way to feed into the FP16 SDPA call. Three options were considered:

| Option | Approach | Trade-off |
|---|---|---|
| **A. StandardKVCache fallback (current)** | Drop affine on donor + sliding layers; store FP16. | Loses ~14 MB compression on E2B/E4B donors; bigger losses possible on future KV-sharing models. No decode-time cost. |
| **B. Lazy dequant on `lastReturnedKeys` read** | Dequant the donor's active prefix every decode step. | Keeps compression; pays ~1–5 ms/decode-step/layer at T=32 k (5–15% decode regression on E2B/E4B). |
| **C. Incremental FP16 mirror** | Maintain a parallel FP16 buffer on donor caches, write both on each update. | Constant per-step cost, but mirror is full FP16 cost *on top of* quantized cache — worse than Option A. Rejected on memory grounds. |
| **D. Reader runs flash quantized SDPA directly (this phase)** | Shared layer accepts `sharedKVArrays` as a quantized tuple and invokes `MLXFast.quantizedScaledDotProductAttention(...)` from Phase 1. | No dequant. No materialised scores. **Best architectural answer**, blocked on Phase 1 (no flash-quant kernel = no tile-based reader path). |

Phase 5 work, once Phase 1 lands:

1. Extend `Gemma4TextAttention.callAsFunction(useSharedKV:sharedKVArrays:...)` (and its Gemma 3n analogue) to accept an `Either<(MLXArray, MLXArray), AffineSharedKV>` shape — the `AffineSharedKV` carries the donor's `(wq, scales, biases)` tuples plus `groupSize` / `bits` / `mode`. When the affine variant is set, route the SDPA call through `MLXFast.quantizedScaledDotProductAttention` (Phase 1's new wrapper). When the raw variant is set, current `MLXFast.scaledDotProductAttention` path is unchanged.
2. Extend `Gemma4TextModel.newCache(...)` (and `Gemma4VLM` analogue + `Gemma3nLanguageModel`) to *stop* setting `forceRawKV: isDonor` for the affine path once the reader supports the quantized tuple. The `architecturalSlidingWindow` fallback can also unwind for affine on Gemma 4 dense sliding layers — Phase 1's kernel handles windowed semantics via the cache's own mask.
3. Add a new `intermediateKVs` storage variant in `Gemma4ModelInner` that carries the affine tuple alongside the existing FP16 variant. Setter checks `cache[i] as? AffineQuantizedKVCache` and grabs the donor's `getQuantizedState()` (existing accessor) instead of `lastReturnedKeys` (which doesn't and won't exist).
4. The shared-reader test in `KVCacheTests.swift` gains a matching `testAffineSharedReader_matchesFP16Reference()` — donor + reader run with identical seeds on quantized vs. FP16 cache, expected `allClose` modulo affine rounding.

Result: Gemma 4 E2B / E4B run `--kv affine4` with **full quantization across all layers** (donors included), no dequant overhead on decode, no workspace cost on the shared-reader SDPA. Closes the GH issue tracking the current `StandardKVCache` fallback.

Estimated ~3 days on top of Phase 1 — pure plumbing (no new kernels), mostly model-side wiring + tests. Can also ship ahead of Phase 2/3/4 if the affine compression on KV-sharing models becomes a higher priority than the bit-width broaden-out.

## Test plan

### Correctness

- `KVCacheTests.swift`: add `testFlashQuantizedSDPA_matchesDiscretePath()` and `…_matchesDiscretePathWithSlidingWindow()` — run the new kernel and the existing discrete path on identical `(Q, qK, qV)` triples; assert `allClose(out_flash, out_discrete, rtol: 1e-2, atol: 1e-3)` (rounding from different summation orders is the only expected delta).
- Re-run the existing `quantized.metal` ref-vs-Metal tests in `TurboQuantKernelTests.swift` to confirm no regression in the underlying dequant primitives.
- Per-quant-scheme PPL on WikiText-2 ctx=2048 / 4096 / 8192 across `qwen35-{0.8,9}b` and `gemma4-{e2b,26b-a4b}` — assert ≤ 1% relative drift vs. discrete path.

### Performance

- `--method summarization --quick --kv affine4` on M1 Max 64 GB. Smoke matrix:
  - `qwen35-{0.8b,2b,4b,9b,35b-a3b}` × `{1024, 4096, 8192, 32768}`.
  - `gemma4-{e2b,e4b,26b-a4b,31b}` × `{1024, 8192, 32768}`.
  - `nemotron-30b-a3b` × `{4096, 32768}`.
- Acceptance gates per row:
  1. Peak GPU within 200 MiB of `--kv none` peak.
  2. Prefill tok/s ≥ 95% of discrete-path baseline (no regression).
  3. Decode tok/s ≥ 95% of discrete-path baseline.
  4. PPL within 1% of discrete-path baseline.

### Numerical sanity

- NaN guard: assert no NaN in scores across the full WikiText-2 forced-decode pass at 32k.
- Sink-token edge case (Phase 4 only): assert the per-head sink logit is folded into the softmax denominator exactly the same way as MLX's pass2 kernel.

## Risk / open questions

1. **Mask-mode plumbing.** MLX's `MLXFast.ScaledDotProductAttentionMaskMode` covers `.none`, `.causal`, `.array`, `.slidingWindow`, `.arrays`. The Metal kernel needs all of them. Sliding-window is the trickiest because per-query `windowSize` + offset interact with the tile loop. Recommended approach: lift the existing tile-mask logic from `quantizedScaledDotProductAttention`'s `case .causal, .slidingWindow` branch into the kernel-side scoreboard.

2. **GQA expansion timing.** Today the affine path tiles GQA by `expandedDimensions + tile`; doing the same inside the kernel saves the `expandedDimensions` allocation. Investigate whether to broadcast on-the-fly inside the kernel (likely yes — that's what upstream MLX SDPA does).

3. **Numerical drift from different summation orders.** The discrete path computes the full softmax in fp32; flash-style online softmax accumulates in fp32 register but tiles in fp16/bf16. Expect sub-1% PPL drift on long-context evals; gate on this in the smoke matrix.

4. **No new dependencies, but four-repo PR ordering.** Same shape as spec 020 / spec 6c chain — mlx kernel → mlx-c ABI → mlx-swift wrapper → mlx-swift-lm callsite. Each PR is independent up to the previous one's submodule bump.

## Estimated scope

| Phase | Effort | Calendar |
|---|---|---|
| 1 (4 + 8-bit + GQA + sliding window) | ~3 weeks single-engineer | After Tier 1–3 stabilise |
| 2 (2 + 3 + 6 bit + mxfp4) | ~1 week | Immediately after Phase 1 |
| 3 (TurboQuant compressed-domain L > 1) | ~1 week | After spec 039 |
| 4 (attention sinks) | ~1 week | After Tier 1 row 6c lands |
| 5 (KV-sharing reader path — Gemma 4 E2B / E4B fallback recovery) | ~3 days | Can ship ahead of Phase 2/3/4 if affine compression on KV-sharing models is prioritised; otherwise after Phase 1 + 4 |

Total: ~6.5 weeks for the full quantized-flash surface. Phase 1 alone closes the affine peak-GPU gap and clears the path for spec 039 (compressed prefix cache) to land without re-introducing the discrete-pass workspace cost on warm-turn prefill. Phase 5 specifically closes the Gemma 4 KV-sharing `StandardKVCache` fallback currently shipped in `makeAttentionCache`.
