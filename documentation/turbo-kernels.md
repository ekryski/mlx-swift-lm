# TurboQuant custom Metal kernels

`mlx-swift-lm` ships eleven custom Metal kernels in the upstream [`ekryski/mlx`](https://github.com/ekryski/mlx) fork. They power the `TurboQuantizedKVCache` path (`--kv turbo*`). This doc gives a brief explainer of each: what it does, when it runs, where its source lives, and how it's wired in Swift.

Apple's first-party MLX kernels (`scaled_dot_product_attention`, `quantized_matmul`, `rope`, `rms_norm`, etc.) are not covered here — they are well-documented upstream. We only override / supplement them where compressed-domain attention requires it.

## Big-picture map

Every decode step on `--kv turbo*` runs three logical stages: **encode** (writes new K/V into compressed form), **score** (computes attention weights), and **aggregate** (computes the value-weighted sum). The custom kernels fall into three families matching those stages:

```
                                  ┌──────────────────────────────────────┐
                  prefill         │ encode family                        │
                  + first decode  │ ─ turbo_fused_encode                 │
                  ───────────────►│ ─ turbo_fused_encode_wht             │
                                  └────────────────┬─────────────────────┘
                                                   │ produces packed K/V indices + per-vector norms
                                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                       compressed K/V storage                          │
   │           uint32 packed indices  +  float32 per-vector norms          │
   └────┬──────────────────────────────────┬───────────────────┬───────────┘
        │ A path                           │ A path             │ B path (opt-in)
        ▼                                  ▼                    ▼
┌──────────────────┐  ┌─────────────────────────┐  ┌──────────────────────────┐
│ TurboFlash       │  │ separated fallback      │  │ bulk dequant + Apple SDPA │
│ (two-pass)       │  │ (mse_score + softmax    │  │ ─ turbo_dequant_rotated  │
│ ─ turbo_flash_p1 │  │  + mse_weighted_sum)    │  │ ─ MLXFast.SDPA (Apple)   │
│ ─ turbo_flash_p2 │  │ ─ turbo_score           │  │                          │
│                  │  │ ─ turbo_value           │  │                          │
└──────────────────┘  └─────────────────────────┘  └──────────────────────────┘

  single-pass (sinks)
  ─ turbo_flash_sdpa_v
```

## Kernel-by-kernel

### Encode family

#### `turbo_fused_encode` ([`turbo_quant.metal`](../../mlx/mlx/backend/metal/kernels/turbo_quant.metal))

**What it does.** Takes new K (or V) of shape `[B, H, S, D]` and emits a packed compressed representation in one Metal dispatch:

1. Per-vector L2 norm (used to scale the codebook lookup back to original magnitude on decode)
2. Dense O(D²) matrix multiply by the codec rotation `Π` (Wadamard / random orthogonal, fixed at calibration time) — moves K into a coordinate frame where the optimal MSE codebook lives
3. Quantise each element to the nearest codebook entry (Lloyd-Max-optimal codebook for the rotated coordinate distribution)
4. Pack the codebook indices into uint32 words (32 / bits indices per word — e.g., 8 for 4-bit, 16 for 2-bit)

**When it runs.** Once per layer per decode-time write (after the lazy compression on first L=1 decode call), and once per layer during `compressRawCache(...)` when the prefill→decode boundary is crossed.

**Template params.** `Bits` (2 / 3 / 4 / 8), `Dim` (head dim — 64 / 96 / 128 / 256), `PackedWidth` (= 32/Bits — derived from Bits).

**Swift entry point.** `TurboQuantKernelOps.fusedEncode(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1270).

#### `turbo_fused_encode_wht`

**What it does.** Same as `turbo_fused_encode` but replaces the dense O(D²) rotation with a Fast Walsh-Hadamard butterfly — O(D log D) bit-shuffle, no matmul. Only valid when `D` is a power of two (which most production models are: 64, 128, 256).

**When it runs.** Preferred over `turbo_fused_encode` when `D` is a power of two. Same callsite, dispatched by the wrapper.

**Template params.** `Bits`, `Dim`, `PackedWidth`, `LogDim` (= log₂(Dim)).

**Swift entry point.** `TurboQuantKernelOps.fusedEncodeWHT(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1301).

### A-path attention family (compressed-domain, no FP16 materialisation)

#### `turbo_flash_p1` (decode, non-causal)

**What it does.** Pass 1 of the two-pass TurboFlash attention algorithm. Each threadgroup processes `BlockSize` tokens and emits, per query row:
- The partial softmax denominator (running max + running exp-sum) over its block
- The partial weighted-V output (uint32-indexed codebook lookup → float multiply → accumulate)

Reads packed K/V directly — no dequant intermediate. Stores partial results in a per-block scratch buffer for Pass 2 to merge.

**When it runs.** Default A-path decode (`L=1`, no sinks). Dispatched by `turboFlashAttention(...)`.

**Template params.** `KeyBits`, `ValueBits`, `Dim`, `KeyPackedWidth`, `ValuePackedWidth`.

**Tile geometry.** `flashBlockSize` (default 64, adaptive on `tokenCount / 32`, env override `TURBO_FLASH_BLOCK_SIZE`). `flashNR0` (default 2, env override `TURBO_FLASH_NR0`) — number of query rows handled per simdgroup.

**Swift entry point.** `TurboQuantKernelOps.turboFlashAttention(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1635).

#### `turbo_flash_p1_causal` (prefill, causal)

**What it does.** Same as `turbo_flash_p1` but for `L>1` prefill cells. Adds causal-mask handling inline in the inner loop (a token at position `i` only attends to positions `≤ i`).

**When it runs.** When the cache prefill path is routed through compressed attention (rare — most callers hit raw-update path for prefill and only compress on first decode call).

**Swift entry point.** `TurboQuantKernelOps.turboFlashAttentionCausal(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1702).

#### `turbo_flash_p1_nr0` / `turbo_flash_p1_nr0_causal`

**What it does.** Variants that handle multiple query rows per simdgroup (the "NR0" amortisation pattern, ported from llama.cpp). For NR0=2: 32 SIMD lanes process 2 queries instead of 1, halving the K-dequant cost per query. NR0=4 / NR0=8 are available but unvalidated.

**When it runs.** Selected automatically when `flashNR0 > 1` (default 2). Smaller models with `headDim ≤ 64` benefit most; larger ones see register-pressure pushback.

#### `turbo_flash_pass2` (cross-block reduction)

**What it does.** Pass 2 of two-pass TurboFlash. Reads the partial softmax states + partial V outputs that Pass 1 emitted into the scratch buffer, runs the online-softmax merge formula across all blocks, emits the final normalized attention output.

**When it runs.** Once per Pass 1 dispatch. The two passes split into separate kernels because a single-pass version would require cross-threadgroup synchronisation (Metal doesn't expose that primitive).

**Template params.** `Dim`.

#### `turbo_flash_pass2_fused_rot`

**What it does.** Same as `turbo_flash_pass2` but fuses the post-softmax inverse-rotation matmul (`out @ Π_v^T` to move back from codec coordinates to original V space) into the kernel — saves one transient FP16 buffer of shape `[B, H, L, D]`.

**When it runs.** When the caller passes a `valRotation` argument to `turboFlashAttention(...)` (most production calls do — the codec has a non-trivial value rotation).

#### `turbo_flash_sdpa_v` (single-pass, sinks-aware)

**What it does.** Single-pass alternative to the `p1 + pass2` chain for sinks-using models (GPT-OSS family). Does score + online softmax + sinks fold + value aggregation in one kernel dispatch. Sidesteps the cross-block-softmax fusion bug that the older pass1/pass2 sinks fold hit on GPT-OSS-20B.

Trade-off: no Pass 1/Pass 2 parallelism across blocks → slower than the two-pass version on long contexts, but correct on sinks where the two-pass version was incoherent.

**When it runs.** When `sinks != nil` and `L == 1`. Replaces the `p1+pass2` chain for GPT-OSS-20B-style attention.

**Template params.** `KeyBits`, `ValueBits`, `Dim` (same instantiation matrix as `turbo_flash_p1`).

**Swift entry point.** `TurboQuantKernelOps.turboFlashSDPAv(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1806).

### Separated fallback family (compressed-domain, used when TurboFlash doesn't have an instantiation)

#### `turbo_score`

**What it does.** Q × K dot product, where K is read in packed-codebook form. One kernel per `(bits, dim, packedWidth)` combo. Outputs raw attention scores `[B, H, L, T]`.

**When it runs.** When the `(keyBits, valueBits)` combo lacks a TurboFlash kernel instantiation (e.g., uncommon configurations from research code). In production this is rarely hit.

**Swift entry point.** `TurboQuantKernelOps.mseScore(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1856).

#### `turbo_value`

**What it does.** Value aggregation — weighted sum of codebook-indexed V vectors using a precomputed softmax-weight tensor. Output shape `[B, H, L, D]` in codec coordinates (caller applies the inverse value rotation).

**When it runs.** Pairs with `turbo_score` in the separated fallback path. Caller chain: `turbo_score` → softmax (in MLX, not in kernel) → `turbo_value`.

**Swift entry point.** `TurboQuantKernelOps.mseWeightedSum(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1878).

### B-path support kernel

#### `turbo_dequant_rotated`

**What it does.** Bulk decompresses `[B, H, T, PackedWidth] uint32 + norms[B, H, T]` back to `[B, H, T, Dim]` FP16 / BF16, **still in the codec-rotated coordinate frame** (Π is not undone). One thread per packed word; bit-extract → codebook lookup → norm-multiply, fused.

**When it runs.** B path opt-in — when `useDequantSDPA: true` or `useBias: true` (the bias-correction path doesn't have a compressed-domain implementation yet, so it routes through B). After this kernel, the caller invokes `MLXFast.scaledDotProductAttention` on the dequanted tensors. The rotation Π cancels inside SDPA because Q and K both live in rotated space.

**Why the FP16 buffer is the cost.** This kernel materialises a transient `[B, nKV, T, D] FP16` tensor per layer per decode step. At ctx=8k, D=128, nKV=8, B=1, that's 8 × 8192 × 128 × 2 = 16 MiB per layer per step. For a 40-layer model that's ~640 MiB of working memory per decode step (lazily eval'd, so peak is bounded by streaming and the MLX scheduler doesn't actually hold all 40 simultaneously — but the cycles are real).

**Template params.** `Bits`, `Dim`, `PackedWidth`, `T` (output dtype: half / bfloat).

**Swift entry point.** `TurboQuantKernelOps.bulkDequantRotated(...)` ([`TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift):1458).

## Two routing decisions

### A path (default) vs B path (opt-in)

Selected by `TurboQuantizedKVCache.useDequantSDPA` (or env `TURBO_DEQUANT_SDPA=1`):

| | A path (TurboFlash) | B path (dequant + Apple SDPA) |
|---|---|---|
| Operates on quantized K/V | yes — kernel reads packed indices directly | no — dequants to FP16 first |
| Per-step FP16 working buffer | no | yes (`B × nKV × T × D × 2 bytes`) |
| Kernel set | `turbo_flash_p1` + `turbo_flash_pass2` (or `turbo_flash_sdpa_v` for sinks) | `turbo_dequant_rotated` + Apple's `scaled_dot_product_attention` |
| Speed vs `--kv none` | 60-99% depending on cell | 91-99% on most cells (matches matmul-engine throughput) |
| Memory vs `--kv none` | KV ~3-4× smaller; peak same as `--kv none` during prefill (see below) | KV ~3-4× smaller plus a transient FP16 buffer during decode |

### TurboFlash variant (within A path)

| Variant | When |
|---|---|
| `turbo_flash_sdpa_v` (single-pass with sinks) | Sinks-using models (GPT-OSS family) — bias forces this branch today via `useBias: true` |
| `turbo_flash_p1` + `turbo_flash_pass2` (two-pass) | All other A-path decode cells |
| `turbo_flash_p1_causal` + `turbo_flash_pass2` | When compressed attention serves a `L>1` prefill cell (rare) |
| Separated `turbo_score` → softmax → `turbo_value` | When the `(keyBits, valueBits)` combo lacks a TurboFlash kernel instantiation |

## Why peak GPU on `--kv turbo*` matches `--kv none` and `--kv affine*` is lower

The 156-cell sweep surfaces this pattern cleanly. Example from Qwen 3.5 9B 4bit at ctx=1024:

| KV | Decode tok/s | KV cache | GPU peak |
|---|---:|---:|---:|
| none | 49.3 | 57 MB | 6.00 GB |
| affine-8 | 48.1 | 59 MB | 5.92 GB |
| affine-4 | 48.6 | 43 MB | 5.91 GB |
| turbo4 | 47.8 (B) | 36 MB | 6.00 GB |
| turbo4v2 | 47.9 (B) | 34 MB | 6.00 GB |

Two different cache architectures explain the difference:

- **`AffineQuantizedKVCache` quantises at write time.** [`updateQuantized(...)`](../Libraries/MLXLMCommon/KVCache.swift) (line 1109) runs `quantized(keys, ...)` on every prefill chunk and stores `(wq, scales, biases)` directly. The cache never holds the full FP16 K/V in memory — even during prefill. So `--kv affine*` peak is bounded by `quantised K/V + transient compute`, ~80-90 MB lower than `--kv none` on this model.

- **`TurboQuantizedKVCache` is a two-phase architecture: raw FP16 during prefill, lazy compression on first decode.** See the file-header comment ([`TurboQuantKVCache.swift`](../Libraries/MLXLMCommon/TurboQuantKVCache.swift):97). The prefill path appends to `rawKeys` / `rawValues` (same shape as `--kv none`), then `compressRawCache(...)` fires at the prefill→decode boundary and emits the packed buffer. The GPU peak measurement spans the entire run — including prefill — so the peak captures the moment when both raw and compressed buffers transiently coexist. That moment sets the peak; the post-decode steady state is much smaller.

This is an architectural choice. Reasons it's that way:

1. **Prefill speed.** Encoding during prefill would add `~D²` matmul + quantise cost per token (the cost of `turbo_fused_encode`). On a 4k-token prefill that's measurable (~5-10% prefill regression). Lazy compression amortises the encode work over a single batched dispatch at the boundary.
2. **Prefill correctness on sliding-window layers.** Some sliding-window-eviction quirks are simpler to reason about in raw form. Eviction happens against the raw buffer; compression sees a clean post-eviction state.
3. **Decode-time memory is what matters in practice.** Long-running streaming sessions are bounded by decode-time working set. Steady-state memory on `--kv turbo*` is ~3-4× smaller than `--kv none` — the peak measurement is an artefact of the bench harness measuring max-over-the-whole-run, not what the user sees at minute 30.

The peak could be brought down by encoding during prefill (option to add a `eagerEncode: true` flag on the cache constructor). That work is unspecced — see [Future work](#future-work) below.

## Tunables

Behaviour controlled by environment variables and module-level statics:

| Env var | Default | Range | Effect |
|---|---|---|---|
| `TURBO_DEQUANT_SDPA` | unset (= 0) | 0 / 1 | Force B path (1) or A path (0) globally |
| `TURBO_BIAS` | unset | 0 / 1 | Force DC-bias correction on (1) / off (0) globally; default-on for GPT-OSS-20B |
| `TURBO_FLASH_BLOCK_SIZE` | adaptive (max(16, tokenCount/32), clamped to [16, 256], rounded to power of 2) | any power of 2 | Pin Pass 1 block size; disables adaptive sizing |
| `TURBO_FLASH_NR0` | 2 | 1, 2, 4, 8 | Number of query rows per simdgroup in NR0 amortised kernels |
| `TURBO_SPARSE_V_THRESHOLD` | 0 | float | Skip V aggregation for attention weights below this threshold |
| `TURBO_DEQUANT_JIT` | unset | 0 / 1 | Force the JIT'd MLXFast.metalKernel dequant path (debugging only — the precompiled `turbo_dequant_rotated` is the production path) |

See [`kv-cache.md`](kv-cache.md) for the full behaviour-knob table including cross-class flags.

## Future work

Optimisations identified but not yet specced:

### From the 2026-05-14 A-vs-B bench (specced in [#043](../specs/043-turboflash-kernel-uplift.md))

1. **Per-simdgroup bit-unpack reuse** — cache the unpacked K block in threadgroup memory once per tile. Highest leverage on long-context regression. (Spec 043 Phase 1.)
2. **bf16 V accumulator + fp32 softmax m/l** — drop the per-tile value-aggregation tensor from `float` to `half`. (Spec 043 Phase 2.)
3. **headDim-aware tile autotune** — `(headDim) → (NR0, blockSize)` static table at dispatch time. Current `flashBlockSize` adapts on tokenCount but not on headDim. (Spec 043 Phase 3.)

### From the broader audit (specced in [#042](../specs/042-metal-kernel-simd-audit.md))

4. **`simdgroup_matrix_multiply_accumulate` (MMA) conversion** — replace `float4` scalar SIMD with matrix-engine intrinsics across all hand-rolled kernels. ~8× FP16 throughput on M2+. (Spec 042 Phase 1.)
5. **Threadgroup-shared codebooks** — small (16-256 entries) but read by every SIMD lane per token. Currently in device memory; move to `threadgroup` storage. (Spec 042 Phase 1.)
6. **`(headDim, bits, groupSize)`-specialised instantiations** — emit the full template fan-out, same pattern Apple's `scaled_dot_product_attention.metal` uses for its 12 specialisations.

### Not yet specced (candidates for future work)

7. **Fused K+V dequant kernel.** B path currently calls `bulkDequantRotated` twice — once for K, once for V. A single fused kernel would halve the dispatch overhead and improve cache locality. Worth ~3-7% on the B path.
8. **Pass1+Pass2 fusion at small token counts.** For ctx ≤ 1024 the block count is ~4-16, and Pass 2's merge work is a measurable fraction of decode-step latency. A single-pass variant of the two-pass kernel (similar to `turbo_flash_sdpa_v` but without sinks) could shave ~5-10% at short context.
9. **Eager prefill encode.** Encode K/V during prefill instead of lazy at first decode. Trades ~5-10% prefill regression for ~3-4× lower peak GPU on long prefills. Optional via cache constructor flag.
10. **Adaptive precision per layer.** Some layers (boundary / outlier layers per the TurboQuant paper) may need more bits for the same MSE. Currently we use a global `(keyBits, valueBits)` schedule. Per-layer schedules could shave 0.5-1 bit on average without quality loss.
11. **Sparse-V skip auto-tune.** `TURBO_SPARSE_V_THRESHOLD` is currently a flat global value. A per-cell adaptive threshold (based on softmax entropy) could skip 30-50% of V-aggregation work on attention heads with concentrated mass.
12. **Fast Walsh-Hadamard for the value rotation post-matmul.** When `valueMSECodec.rotation` is a Hadamard matrix (most production calibrations), the post-matmul could use the FWHT butterfly instead of a dense matmul. Saves a transient `[B, H, L, D]` buffer.

Items 7-12 are candidates for a future spec 044 once spec 043 lands.

## File index

| File | What it contains |
|---|---|
| [`mlx/mlx/backend/metal/kernels/turbo_quant.metal`](../../mlx/mlx/backend/metal/kernels/turbo_quant.metal) | `turbo_score`, `turbo_value`, `turbo_fused_encode`, `turbo_fused_encode_wht`, `turbo_dequant_rotated`, `turbo_flash_pass2`, `turbo_flash_pass2_fused_rot` |
| [`mlx/mlx/backend/metal/kernels/turbo_flash.metal`](../../mlx/mlx/backend/metal/kernels/turbo_flash.metal) | `turbo_flash_p1`, `turbo_flash_p1_causal`, `turbo_flash_p1_nr0`, `turbo_flash_p1_nr0_causal` |
| [`mlx/mlx/backend/metal/kernels/turbo_flash_sdpa.metal`](../../mlx/mlx/backend/metal/kernels/turbo_flash_sdpa.metal) | `turbo_flash_sdpa_v` |
| [`Libraries/MLXLMCommon/TurboQuantKernels.swift`](../Libraries/MLXLMCommon/TurboQuantKernels.swift) | Swift wrappers dispatching to each kernel via `MLXFast.metalKernel(...)` |
| [`Libraries/MLXLMCommon/TurboQuantKVCache.swift`](../Libraries/MLXLMCommon/TurboQuantKVCache.swift) | Cache class consuming all of the above; routing logic in `compressedAttention(...)` |
| [`Tests/MLXLMTests/TurboQuantKernelTests.swift`](../Tests/MLXLMTests/TurboQuantKernelTests.swift) | Per-kernel correctness tests (kernel vs Swift-reference parity) |

## Related docs

- [`kv-cache.md`](kv-cache.md) — full cache-class reference, env-var table, model-coverage matrix
- [`batched-decoding.md`](batched-decoding.md) — deployment-shape tradeoff table (memory vs quality)
- [specs/041-flash-quantized-sdpa.md](../specs/041-flash-quantized-sdpa.md) — architectural spec for the compressed-domain attention path
- [specs/042-metal-kernel-simd-audit.md](../specs/042-metal-kernel-simd-audit.md) — broad SIMD audit roadmap
- [specs/043-turboflash-kernel-uplift.md](../specs/043-turboflash-kernel-uplift.md) — focused 3-phase TurboFlash uplift driven by the 2026-05-14 bench
