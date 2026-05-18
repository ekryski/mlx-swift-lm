# 044 — Batched sparse attention decode

- **Status:** Proposed 2026-05-17 (this PR). Implementation in flight on `pr/alpha-batched-sparse-attention` ([PR #224](https://github.com/ekryski/mlx-swift-lm/pull/224)).
- **Branch:** `pr/alpha-batched-sparse-attention`.
- **Depends on:** none structurally. Composes orthogonally with [spec 041](041-flash-quantized-sdpa.md) (the affine + turbo quantized cache layouts) — the sparse wrapper composes via a held `BatchedKVCache` and routes through the same MLXFast SDPA primitives.

## Problem

Multi-client serving on a single `mlx-swift-lm` engine wants both batched decode (multiple concurrent requests sharing one GPU pass) AND long-context attention (32K+ KV per request). The two pull in opposite directions:

- **Batched decode**: one `[B, 1]` token pass through the whole model. Existing `BatchedKVCache` already supports this on dense attention — the bandwidth cost per step is `O(B × nKVH × T × D)` (the rectangular cache fanned over `B` requests).
- **Long-context sparse decode** (Quest / NSA / SeerAttention-R style): per-request retrieval index narrows attention from `T` to `K_topk ≪ T` candidate blocks. Existing sparse path (`RetrievalAttentionKVCache`) is per-request, B = 1 only — every callsite asserts dim(0) == 1.

The product wants both: B concurrent requests, each at long context, each saving bandwidth via sparse selection. Bandwidth scales as `B × nKVH × T × D` for dense and `B × nKVH × K_topk × D` for sparse — a 4–8× saving at long context is exactly the cell where multi-client serving wants to spend its capacity.

Today the dispatcher sees `B > 1` on a sparse request → falls back to a serial per-request decode loop. The sparse selector AND the SDPA both lose their batching, and the engine effectively serves the cohort at single-stream speed.

- **Goal:** wire a per-`(B, KV-head)` selector index + a batched sparse SDPA path through every supported model family, so concurrent long-context decode keeps both the sparse bandwidth saving AND the single-batched-call dispatch shape.

## Design

### Protocols + cache

Three new public surfaces in `Libraries/MLXLMCommon`:

- `BatchedSparseLLM` (protocol) — pure-attention families. Single method `fullyBatchedSparseDecode(inputs:raCaches:)` over a `[BatchedRetrievalAttentionKVCache]` (one per layer).
- `BatchedHybridSparseLLM: BatchedHybridLLM` — hybrid families (interleaved attention + SSM/GDN). Adds `fullyBatchedSparseDecode(inputs:caches:)` and `newBatchedHybridSparseCache(maxBatch:parameters:raConfig:)` over a `BatchedHybridCache`. Inherits dense `fullyBatchedDecode` from the parent.
- `BatchedRetrievalAttentionKVCache` — composes an `inner: BatchedKVCache` (rectangular fp16 K/V) with a `BatchedRetrievalAttentionIndexB` (per-(B, KV-head) selector index, built lazily). Public API is `update(newKeys:newValues:)` → `updateIndex(newKeys:)` → `sparseAttend(queries:scale:)`.

The composition is orthogonal to the quantization layout: `BatchedKVCache` is still the foundation — `BatchedRetrievalAttentionKVCache` just sits in front of it. A quantization-aware sparse path drops in by swapping the inner cache type without touching the selector or per-family forwards.

### Kernels

Three Metal kernels back the sparse step. All live in `Libraries/MLXLMCommon/RetrievalAttentionKernels/`:

1. **Batched mask kernel** (default — `VSM_SPARSE_BATCHED_KERNEL=mask`). Builds a per-slot `[B, 1, 1, T]` fp16 additive mask from the per-(B, KV-head) top-K block selections, then calls `MLXFast.scaledDotProductAttention(mask: .array(...))`. This hits the tuned `sdpa_vector_2pass` Metal kernel on Apple Silicon (the same one dense uses) — the win comes from skipping non-selected tokens during the score pass, not from reducing K/V bandwidth.
2. **Compose-gather kernel** (`gather`). Materialises `[B, nKVH, K_padded, D]` slabs via `takeAlong`, then calls dense `MLXFast.scaledDotProductAttention(mask: .none)` on the small slab. Saves actual K/V bandwidth; oversub-bound on Apple at low `K_padded / T` ratios.
3. **Fused sparse group kernel** (`group`). Custom Metal kernel (`retrievalAttentionGroupSparseSDPA`) that gathers + SDPAs in one dispatch, GQA-coloaded per simdgroup. Experimental — wins on CUDA-style hardware but loses on M-series due to threadgroup-occupancy mismatch.

Default is the mask kernel (option 1) — it composes with Apple's tuned SDPA fast path and is the only one that ships positive numbers across the longctx grid (see §"Performance" below).

### Per-family hooks

Every supported model has a `{Family}+Sparse.swift` extension file that adds `fullyBatchedSparseForward` overloads on its attention + transformer-block + model-inner classes. The conformance lands on the top-level model class. Pure-attention families implement `BatchedSparseLLM`; hybrids implement `BatchedHybridSparseLLM`.

Family-specific work (because every attention layer's Q / K / V projection ordering differs):

- **Qwen2 / Qwen3 / Llama / Mistral3 / Phi3** — straight projection → reshape → RoPE → cache update → sparse SDPA. Llama-style attention sits behind a uniform `LlamaAttention.fullyBatchedSparseForward` and is mirrored by Mistral3 / Phi3 (Phi3 has a fused qkv split, Mistral3 has a sliding/global layer mix that dispatches per-`layer.useSliding`).
- **Qwen3 / Qwen3MoE** — Q/K RMSNorm applied pre-RoPE on the per-head reshape. MoE FFN (`Qwen3MoESparseMoeBlock`) is orthogonal — it routes by token (`UnaryLayer` conformance) regardless of dense vs sparse attention.
- **Gemma3** — sliding-only attention layers; embedding scale + final-logit softcap preserved through the sparse output.
- **Gemma4** — sliding + global layer mix with DIFFERENT head dims per layer kind; per-layer `BatchedRetrievalAttentionKVCache` shape must match. MoE blocks + `num_kv_shared_layers > 0` (e2b: 20 trailing layers reuse the donor's K/V) both supported — shared layers point at the donor's `BatchedRetrievalAttentionKVCache` instance (mirrors the dense `previousKVs` map), compute Q only, RoPE Q at the donor's pre-update offset.
- **Qwen3.5 / Qwen3.6 (hybrid)** — attention + GatedDeltaNet interleave. Attention layers route through `.sparseAttention`; GDN layers stay on the existing `.gdn(BatchedMambaCache)` batched path. The MoE Qwen3.6 inherits the conformance from `Qwen35TextModel`.
- **NemotronH (hybrid)** — Mamba2 + Llama-style attention + MLP + MoE pattern (4 block kinds via `hybridOverridePattern`). Attention layers (`*`) route to `.sparseAttention`; Mamba2 layers (`M`) route to a new `NemotronHMamba2Mixer.fullyBatchedForward` against `BatchedMambaCache` (with `recDtype` set to the model's compute dtype — Mamba2 has an input-dtype SSM state contract, unlike GDN's fp32). MLP (`-`) and MoE (`E`) blocks contribute NO cache slot — the per-layer dispatch walks the pattern and only advances the cache index on mamba/attention, mirroring `newCache`'s shape.

### Activation

Consumer side is opt-in: the bridge (downstream consumer, e.g. `vllm-swift`) casts the loaded model to `BatchedSparseLLM` / `BatchedHybridSparseLLM` and dispatches `fullyBatchedSparseDecode` when a `BatchedRetrievalAttentionKVCache` list / sparse-augmented `BatchedHybridCache` is in hand. Dispatch logic itself lives downstream — this repo just ships the model-side surface + kernels.

Kernel choice env var: `VSM_SPARSE_BATCHED_KERNEL ∈ {mask, gather, group, loop}` (default `mask`).

## Per-family coverage

| Family | File | Type | Synthetic smoke |
|---|---|---|---|
| Qwen2 | `Qwen2+Sparse.swift` | `BatchedSparseLLM` | `Qwen2BatchedSparseSmokeTests` |
| Qwen3 | `Qwen3+Sparse.swift` | `BatchedSparseLLM` | `Qwen3BatchedSparseSmokeTests` |
| Qwen3MoE | `Qwen3MoE+Sparse.swift` | `BatchedSparseLLM` | `Qwen3MoEBatchedSparseSmokeTests` |
| Llama | `Llama+Sparse.swift` | `BatchedSparseLLM` | `LlamaBatchedSparseSmokeTests` |
| Mistral3 | `Mistral3+Sparse.swift` | `BatchedSparseLLM` | `Mistral3BatchedSparseSmokeTests` |
| Phi3 | `Phi3+Sparse.swift` | `BatchedSparseLLM` | `Phi3BatchedSparseSmokeTests` |
| Gemma3 | `Gemma3+Sparse.swift` | `BatchedSparseLLM` | `Gemma3BatchedSparseSmokeTests` |
| Gemma4 | `Gemma4+Sparse.swift` | `BatchedSparseLLM` | `Gemma4BatchedSparseSmokeTests` (covers MoE + KV-shared) |
| Qwen3.5 / Qwen3.6 | `Qwen35+Sparse.swift` | `BatchedHybridSparseLLM` | `Qwen35BatchedSparseSmokeTests` |
| NemotronH | `NemotronH+Sparse.swift` | `BatchedHybridSparseLLM` | `NemotronHBatchedSparseSmokeTests` |

Eight pure-attention families + two hybrid families. The 22 sparse tests across 12 suites cover protocol surface, cache layout, slot lifecycle, and end-to-end synthetic decode.

## Coexistence with spec 041

Spec 041 (flash quantized SDPA) reshapes the cache: `AffineQuantizedKVCache` / `TurboQuantizedKVCache` replace `StandardKVCache` for quantized configs. The sparse work composes orthogonally — `BatchedRetrievalAttentionKVCache` wraps a `BatchedKVCache` via the public `inner` member; a future quantization-aware sparse path drops in by swapping the inner type without touching the per-family `+Sparse` forwards. The kernel-selection env var (`VSM_SPARSE_BATCHED_KERNEL`) is independent of the quantization-strategy env var (`MLX_AFFINE_SDPA`).

## Apple Silicon SDPA notes

The mask-kernel default depends on Apple's `sdpa_vector_2pass` Metal kernel running with the right block size for the long-context regime. The default `blocks` heuristic in upstream MLX (`mlx-explore/mlx`) was tuned on the `_nomask` kernel and picks `blocks=256` at `N ∈ (8K, 32K]` on M5 Max, which oversubscribes the GPU for the `_floatmask` variant the sparse path uses. The sister PR [ekryski/mlx#34](https://github.com/ekryski/mlx/pull/34) backports the env-var override (`MLX_SDPA_BLOCKS`) from upstream PR 3455 + adds a per-shape blocks formula (`blocks ≈ ceil(960 × 1.1 / (B × nKVH × gqa))`). With it, the mask-kernel sparse path lands +24–42% over the upstream default on long-context batched cells; without it the sparse path can fall behind dense at `ctx ∈ (8K, 32K]` for `B ≥ 8`.

The bridge (downstream) sets `MLX_SDPA_BLOCKS=128` at engine-create time. Setting the env BEFORE `python` invokes is preferable to setting it from Swift (PSO JIT timing) but both work; the difference is the first-call warmup window.

## Legacy retirement

The fork carried a single-stream sparse decode path (`RetrievalAttentionKVCache`,
B=1 wrapper) plus a B≥1 batched cache (`BatchedRetrievalAttentionKVCache`,
this PR). Both share the same selector + mask kernel. With the batched cache
handling B=1 correctly (see `Qwen2BatchedSparseB1Tests`), the single-stream
wrapper is redundant and is not proposed for upstreaming. Memory cost at
B=1 long-context single-stream is `O(maxSeq)` rectangular preallocation —
equivalent to the wrapper's dynamic-concat cost when `maxSeq` is sized to
actual context. One cache, one selector, one mask kernel covers all batch
sizes.

Failed exploration paths (F-71b custom Metal sparse SDPA kernel at B=8 was
14× slower than dense; F-76 implicit sparse was slower than dense at 128K;
F-84 blockGather lost 1.3× at B=1) are documented in the project's research
notes and not proposed for upstreaming.

## Open

- **vllm-swift bridge dispatch** — wiring up `BatchedSparseLLM` / `BatchedHybridSparseLLM` casting in the engine's `decode_step` lives in the downstream `vllm-swift` consumer, NOT this repo.
- **`.attention` slot on the sparse-decode path** — `BatchedHybridSparseLLM`'s `newBatchedHybridSparseCache` emits only `.sparseAttention` slots for attention layers. The per-block dispatch supports `.attention` slots as a defensive path (the parent `BatchedHybridLLM`'s dense `fullyBatchedDecode` reuses the same per-block forward), but it's not exercised by the sparse-decode dispatcher.
- **Compressed-domain sparse decode** — when [spec 039](039-compressed-prefix-kv-cache.md) lands, swapping `BatchedKVCache` → quantized variant inside `BatchedRetrievalAttentionKVCache.inner` will let sparse decode read directly from compressed K/V. Out of scope for this spec.

## References

- **Prior art**: NSA ("Native Sparse Attention", DeepSeek-AI 2025), SeerAttention-R (2024), Quest (page-bounded top-K, Du et al. 2024), DuoAttention (retrieval/streaming head split, spec 036).
- **Sister kernel PR**: [ekryski/mlx#34](https://github.com/ekryski/mlx/pull/34) — `MLX_SDPA_BLOCKS` env override + per-shape blocks heuristic.
- **Composition spec**: [041 — Flash quantized SDPA](041-flash-quantized-sdpa.md) (cache composition compatibility).
- **Adjacent specs**: [034 — Decode-side KV selection](034-decode-side-kv-selection.md), [035 — Quest page-bounded top-K](035-quest-page-bounded-topk-attention.md), [036 — DuoAttention](036-duoattention-retrieval-streaming-head-split.md).
