# Spec 042 — Precision audit (M1 Max focus)

**Branch:** `ek/specs-042-043-kernel-uplift`
**Date:** 2026-05-15
**Status:** Audit draft. Per-site decisions pending discussion.
**Subsumes:** [#162](https://github.com/ekryski/mlx-swift-lm/issues/162) (bf16 vs fp16 host-side compute), [#158](https://github.com/ekryski/mlx-swift-lm/issues/158) (fp16 V accumulator)

## Why precision matters on M1

Apple Silicon Metal SIMD hardware natively supports **fp16 + fp32**.
`bfloat16` is implemented in software via the [`bf16.h`](../../mlx-swift/Source/Cmlx/mlx-generated/metal/bf16.h)
shim — every bf16 load and store goes through a per-element conversion
routine. In hot kernels this is a measurable overhead:

| Operation | fp16 | fp32 | bf16 |
|---|---|---|---|
| Per-element load/store | hardware-native | hardware-native | software shim (~3-5x scalar overhead) |
| Per-element compute | hardware-native | hardware-native | promoted to fp32, then truncated |
| Register footprint | 16 bits | 32 bits | 32 bits (stored as fp32 internally) |
| TG memory bank traffic | 2 bytes/lane | 4 bytes/lane | 4 bytes/lane |

The rule of thumb: **on M1 family, prefer fp16 anywhere the dynamic
range fits**; fp32 only where softmax / accumulator stability requires
it; bf16 only when storage compatibility with model weights forces it.

## Audit — kernel side

### turbo_flash_sdpa_v ([`turbo_flash_sdpa.h`](../../mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/turbo_flash_sdpa.h))

| Site | Current dtype | Recommended | Rationale | Effort |
|------|--------------|-------------|-----------|--------|
| `device bfloat* out [[buffer(7)]]` (output) | bf16 | **add `half` variant** alongside | Per-element output store today triggers software bf16 cast on M1. Consumer cast at the model boundary is a one-shot. Same template pattern `turbo_dequant_rotated` already uses (lines 645/655 in turbo_quant.metal). | M — adds template param + new instantiations; cross-repo PR chain to teach Swift to pick dtype |
| `device float* queries [[buffer(0)]]` | fp32 | keep fp32 | Score path needs the dynamic range; fp16 q · k can overflow on long contexts | — |
| `device bfloat* sinks` | bf16 | **convert to fp16 input** | Per-Q-head value; one read per kernel; could be fp16 since `sinks` is bounded ([-5, +5] empirically per spec 6c bench) | S — input dtype on the kernel signature; Swift caller `.bfloat16` → `.float16` for the sinks array |
| `device float* k_codebook` | fp32 | keep fp32 — but TG cache could be `half` | Codebook values fit comfortably in fp16; small (≤256 entries × 2 bytes = 512 B vs 1 KiB in fp32). Already tried in my earlier Phase 2 over-aggressive attempt; regressed perf. Re-test in isolation. | S — typedef swap |
| `thread float q[]`, `thread float k[]` (per-lane registers) | fp32 | keep fp32 | Score `q · k` is the precision-critical inner product. Don't downgrade. | — |
| `thread half o[]` (V accumulator) | half (post Phase 2) | keep half | Already fp16 per Phase 2 | — |
| `threadgroup float outputs[BN * BD]` (cross-simdgroup) | fp32 | could be fp16 | Reused for cross-simdgroup `o[i]` aggregation; same data as `o[]` which is already fp16. Saves 2 KiB TG memory + faster TG access. | S — typedef swap, careful with the simd_sum * factor compute path |
| `threadgroup float max_scores[BN], sum_exp_scores[BN]` | fp32 | **MUST stay fp32** | Softmax m/l dynamic range. Spec 043 Phase 2 explicit call-out. | — |

### turbo_flash.metal (`turbo_flash_p1` family)

| Site | Current dtype | Recommended | Rationale | Effort |
|------|--------------|-------------|-----------|--------|
| `device float* q_rot` | fp32 | keep | Same as above | — |
| `device float* key_codebook`, `val_codebook` | fp32 | TG storage could be fp16 | Codebooks; same trade-off as turbo_flash_sdpa_v | S |
| `device float* o_partials, m_partials, l_partials` (Pass 2 inputs) | fp32 | Keep `m_partials`, `l_partials` fp32; **`o_partials` could be fp16** | o_partials is per-block V partial sums; same precision class as `o[]` itself | M — also need to update the Pass 2 kernel that consumes them |
| `thread float key_cb[KEY_LEVELS], val_cb[VAL_LEVELS]` (per-thread codebook copy) | fp32 | **Hoist to TG memory as half** | Spec 042 §2 audit item, also relevant for Phase 1 unfinished work in `turbo_flash.metal` (Phase 1 only hoisted in `turbo_flash_sdpa.metal`). Per-thread fp32 codebook costs 32 lanes × 16-256 floats = 2-32 KiB register pressure across the simdgroup; fp16 TG single-copy is 32 bytes - 1 KiB. | S — same pattern as Phase 1 applied to non-sdpa_v kernels |
| `thread float q_vals[]`, `k_decoded[]`, `v_decoded[]` | fp32 | keep fp32 score; **`v_decoded[]` could be fp16** | Score path stays fp32; V decoded values feed only into the V aggregation which already uses fp16 in turbo_flash_sdpa_v post Phase 2 | S |

### turbo_quant.metal (turbo_score, turbo_value, turbo_dequant_rotated)

| Site | Current dtype | Recommended | Rationale |
|------|--------------|-------------|-----------|
| `turbo_dequant_rotated` output | bf16 and fp16 instantiations exist | **already done** | Output dtype is already parameterised; dispatcher picks per consumer |
| `turbo_score` output `device float*` | fp32 | keep fp32 | Attention scores feed softmax; need range |
| `turbo_value` output | fp32 | keep fp32 | Feeds inverse rotation; precision-critical |
| `turbo_fused_encode` input vector | bf16 (typically) | input dtype is caller-controlled; output is uint32 (packed indices) + fp32 (norms) | OK |

## Audit — Swift side

### `MSECodec` ([`TurboQuantKVCache.swift`](../Libraries/MLXLMCommon/TurboQuantKVCache.swift):625+)

| Site | Current dtype | Recommended | Rationale |
|------|--------------|-------------|-----------|
| `rotation` matrix (`[dim, dim]`) | bf16 (line 663, 668) | **fp16 alternative** for M1 | Used in Swift-side matmul on every decode step. bf16 × bf16 matmul on M1 = software cast for inputs + outputs. fp16 would be hardware native. |
| `rotationT` (transpose) | bf16 (derived) | fp16 if rotation is fp16 | Same |
| `rotatedOnes` `[1, dim]` (line 677) | bf16 | fp16 | One vector per codec; trivial size; fp16 is fine for the constant |
| `boundaries`, `codebook` | fp32 | fp32 (or fp16) | These feed kernels that read fp32; could be fp16 if kernels were updated too. Not a quick win standalone. |
| Cache buffers (`keyNorms`, `valNorms`, `keyBias`, `valBias`) | fp32 | fp32 — norms have wide dynamic range | Norms can be small (e.g., 1e-4) or large (e.g., 100); fp16 risks underflow. Bias is bounded but fp32 storage is small overhead. Keep fp32. |
| `keyPackedMSE`, `valPackedMSE` | uint32 (packed indices) | unchanged | Bit-exact storage, not dtype |

### Dispatcher casts ([TurboQuantKVCache.swift:1943-1944](../Libraries/MLXLMCommon/TurboQuantKVCache.swift))

```swift
let dt: DType = (originalDtype == .bfloat16 || originalDtype == .float16)
    ? originalDtype : .bfloat16
```

When the model is bf16 (typical), `dt = .bfloat16` → bulk dequant + Apple SDPA path runs entirely in bf16. **On M1 this incurs bf16 emulation across the whole SDPA call**, including Apple's matrix-engine MMA which still pays the bf16 cast cost per lane.

Recommendation: when M1 + model is bf16, consider running the dequant + SDPA chain in fp16 instead, then casting back to bf16 at the model boundary. This is a model-output dtype mismatch (kernel writes fp16, model expects bf16) but the per-batch cast is a single op vs the per-element cost inside the kernel.

**Effort:** M — needs careful dtype tracking across the bulk-dequant → SDPA → inverse-rotation chain. The codec's `rotation` matrix dtype must agree with the SDPA dtype.

## Prioritised changes

In order of expected M1 lift × implementation simplicity:

1. **Hoist codebook in `turbo_flash.metal` to threadgroup memory as fp16** (S).
   Closes the Phase 1 gap that's now turbo_flash_sdpa.metal-only; reduces
   register pressure and bf16 emulation cost. Matches spec 042 §2.

2. **Add `half` output variant to `turbo_flash_sdpa_v`** (M).
   Per-element bf16 store on M1 = software cast per output element. The
   `turbo_dequant_rotated` precedent shows the dual-dtype instantiation
   pattern works.

3. **Swift-side fp16 codec rotation matrix** (S, but caller-touching).
   Provides an `.fp16` build of the codec for M1; default bf16 unchanged.
   Caller (`TurboQuantKVCache`) picks based on model dtype + hardware.

4. **fp16 V accumulator inside `turbo_flash.metal`** (S).
   Phase 2 of spec 043 applied to the non-sdpa_v kernels. Same `typedef
   half ACC_T` swap, same Metal promotion semantics.

5. **Dispatcher: fp16 SDPA path on M1 when model is bf16** (M).
   Bigger architectural change; defer until 1-4 land and re-bench.

## What this won't fix

- **Kernel output dtype must match what the model expects.** Adding fp16
  output variant lets consumers choose, but the model still writes back
  in its native dtype. For bf16 models, the output cast still happens
  somewhere — we're just moving it from per-element to per-batch.

- **MMA (spec 042 Phase 1a+) requires M2+ regardless of precision.**
  M1 family lacks `simdgroup_matrix_multiply_accumulate`. Precision
  audit is M1-relevant; MMA conversion is M2+.

- **The codebook codec calibration is fp32 offline.** Changing the
  per-instance codebook dtype from fp32 to fp16 only saves at-runtime
  bytes; the offline Lloyd-Max optimisation doesn't change.

## Implementation order proposal

Combine 1 + 4 into a single "spec 042 §7b — internal precision audit"
commit chain (no cross-repo plumbing; kernel-internal only).

Then 2 as a separate "spec 042 §7c — fp16 output dtype variant" chain
(4-repo plumbing).

Then 3 as a separate "spec 042 §7d — Swift-side codec dtype option"
(mlx-swift-lm-only, no kernel changes).

Then 5 dependent on the previous three.
