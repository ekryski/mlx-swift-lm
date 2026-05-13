# 042 — Metal kernel SIMD-optimisation audit

**Status:** Spec drafted 2026-05-12. **Not started.**
**Branch:** TBD (`ek/042-metal-kernel-simd-audit-phase1` once implementation begins)
**Depends on:** none structurally. Logically pairs with [spec 041](041-flash-quantized-sdpa.md) since both target compute density on M-series GPUs; can land independently.

## Problem

`mlx-swift-lm` ships several hand-rolled Metal kernels in the upstream `mlx` fork (turbo-quant family) and consumes Apple's first-party MLX kernels for everything else. The hand-rolled set was authored with correctness as the primary goal; SIMD-level tuning (matrix-engine intrinsics, threadgroup-memory tiling, register-pressure budgeting, vec-load coalescing) was patched in opportunistically rather than systematically.

Concrete evidence the gap is real: TurboFlash's decode path on Gemma 4 31B + `--kv turbo4v2` at ctx 32k clocks ~5.6 tok/s vs Apple's `MLXFast.scaledDotProductAttention` on dequanted FP16 K/V at ~9.6 tok/s (same workload, same memory). Both kernels touch the same packed K/V bytes; the difference is Apple's kernel uses `simdgroup_matrix_*` MMA intrinsics + tile geometry tuned for M-series compute units, while TurboFlash uses plain `float4` SIMD. The compute-density ratio is roughly 8× per cycle in Apple's favour.

The bench comment in `TurboQuantKVCache.swift` (~line 1849) captures the same effect from a different angle: dequant-first SDPA (which routes through Apple's kernel post-dequant) beat TurboFlash by 14–52% on Qwen 0.8B / 9B / Nemotron 30B at varying context. That gap is the SIMD-optimisation gap, not an algorithmic gap.

**Goal:** audit every hand-rolled Metal kernel in `mlx-swift-lm`'s upstream `mlx` fork, identify which ones miss matrix-engine / threadgroup-memory / vec-load patterns Apple's first-party kernels use, and bring them up to par. Treat it as a one-time perf-uplift sweep rather than per-kernel one-off PRs.

## Kernels in scope

Audit checklist — verify each kernel uses:

1. **`simdgroup_matrix_multiply_accumulate`** for any matmul-shaped inner loop (Q·K^T, weights·V, dequant × Q, codec rotation). The matrix-engine instructions deliver ~8× FP16 throughput vs scalar SIMD on M2+; M1 has limited support but the kernel template should branch on hardware capability rather than always using scalar.
2. **Threadgroup memory for re-used operands.** Codebooks, rotation matrices, group-shared scales / biases — anything read by more than one SIMD lane per token should sit in `threadgroup` storage, not device memory.
3. **Vectorised loads** (`packed_uint4`, `bfloat4`, `half4`) for memory traffic. Each thread should pull 8–16 elements per device-memory transaction.
4. **No register spills.** Profile each instantiation under Instruments' Metal System Trace; spills show as L1 writes and tank throughput silently.
5. **Tile sizes aligned to GPU warp / threadgroup-memory bank counts.** Apple's GPUs have 32-wide SIMD lanes and 32-bank threadgroup memory; tile widths should be multiples of 32 (or 16 if register pressure forces a halving) and access patterns must avoid bank conflicts.
6. **Specialisation per `(headDim, bits, groupSize)`.** Apple's `scaled_dot_product_attention.metal` template emits ~12 specialisations (different head dims × different precisions). Hand-rolled kernels often have one generic version that loses to the specialised case; emit the full matrix when the kernel ships.
7. **Numeric-format audit (composes with [#162](https://github.com/ekryski/mlx-swift-lm/issues/162) and [#158](https://github.com/ekryski/mlx-swift-lm/issues/158)).** Two sub-passes:
    - **bf16 vs fp16 host-side compute** ([#162](https://github.com/ekryski/mlx-swift-lm/issues/162)) — Apple Silicon Metal SIMD natively supports fp16 + fp32, but bf16 is a software conversion via `bf16.h` that adds per-load/per-store overhead in hot kernels. Audit every bf16 site in the kernel set; where numerical headroom allows, convert compute to fp16 (storage may stay bf16 to match model weights). Gate on `≥ 5% mean decode tok/s improvement with no PPL / KLD regression` per the issue's existing acceptance bar.
    - **fp32 accumulator audit inside kernels** ([#158](https://github.com/ekryski/mlx-swift-lm/issues/158)) — TurboFlash pass2 currently accumulates `o[DIMS_PER_LANE]` + softmax `m, l` in `float`. Per `sam/planning/performance-notes/simd-analysis-and-remaining-gains.md`, dropping to fp16 V accumulator (keeping fp32 m/l for softmax stability) recovers a documented gain. Same pattern likely applies to `mse_score`, `mse_weighted_sum`, `turbo_dequant_rotated` — sweep them all in this audit rather than as separate one-off PRs.

  Important constraint: never downgrade the *softmax* accumulator (m / l in online-softmax algorithms) below fp32 — softmax dynamic range needs the headroom. The audit targets compute paths and final-stage accumulators, not the running-max bookkeeping.

Concrete kernel list (Phase 1 surface):

| Kernel | Where | Current status |
|---|---|---|
| `turbo_flash_attention` (L=1 decode) | `mlx/mlx/backend/metal/kernels/turbo_flash_attention.metal` (in `ekryski/mlx` fork) | Plain `float4` SIMD; no matrix-engine; codebook in device memory. Highest-leverage target. |
| `turbo_flash_attention_causal` (L>1 prefill) | same file | Same issues as L=1; bigger workload so the SIMD lift would compound. |
| `turbo_dequant_rotated` (bulk K/V dequant) | `turbo_quant.metal` | Vectorised `float4` loads ✓; threadgroup codebook ✗; no MMA. Moderate target — invoked once per layer per decode step in the dequant-first path. |
| `fused_encode_dispatch` (prefill K/V encode) | `turbo_quant.metal` (paths: `fused_encode`, `fused_encode_wht`) | WHT butterfly unrolled per dim ✓; matrix-engine ✗ (the butterfly is bit-shuffle-heavy, less MMA-shaped); profile and decide. |
| `mse_weighted_sum` (compressed-domain V × softmax-weights aggregation) | `turbo_quant.metal` | Hot path on the separated turbo decode fallback; no matrix-engine. |
| `mse_score` (Q × packed K scoring, separated fallback) | `turbo_quant.metal` | Used when neither TurboFlash nor dequant-first apply; legacy fallback. Lower priority. |

Kernels that are *already* well-tuned (skip unless audit surfaces a regression):

- Everything in upstream Apple `mlx` (`scaled_dot_product_attention`, `quantized_matmul`, `rms_norm_rope_fused`, etc.) — already use matrix-engine + tile patterns.
- `fused_norm_rope` (our fork) — uses Apple's `rms_norm` + `rope` building blocks; vectorised internally.

## Phasing

### Phase 1 — TurboFlash MMA conversion

Highest-leverage. Convert `turbo_flash_attention` (both L=1 and L>1 variants) to use `simdgroup_matrix_*` MMAs for Q·K^T scoring and weights·V aggregation. Hoist codebook to threadgroup memory. Tile geometry: `Bq = 16, Bk = 16` per the same constraints spec 041 Phase 1 identifies (dequant register pressure caps the tile vs Apple's FP16-K/V baseline `Bq = 32, Bk = 32`).

**Acceptance gate:** TurboFlash decode tok/s ≥ 90% of dequant-first decode tok/s on `qwen35-0.8b` × `--kv turbo4v2` (today's worst gap, where dequant-first wins by 52%). Equivalent or faster on `gemma4-31b` × `--kv turbo4v2` ctx 32k (currently TurboFlash 5.6 vs dequant-first 9.6 — flip the relationship).

Result: dequant-first SDPA can be retired entirely. The `headDim < 256` gate added to `compressedAttention` (workaround for the 31B 32k coherence bug) collapses; all paths route through compressed-attention. Closes the "should we use TurboFlash or dequant-first?" question that has eaten two days of bench time across this PR and prior ones.

### Phase 2 — `turbo_dequant_rotated` MMA conversion

If Phase 1 lands and the dequant-first path is retired, this might be skippable. But if any caller still needs bulk dequant (e.g. for `Affine` cache snapshotting or migration code paths), the matrix-engine conversion + threadgroup codebook are the same templates from Phase 1.

### Phase 3 — `mse_weighted_sum` and `mse_score` (fallback paths)

Lower priority. These only fire when neither TurboFlash nor dequant-first apply (e.g. unsupported `(keyBits, valueBits)` combos). Convert for consistency but accept lower priority since the fallback path is rarely hit.

### Phase 4 — `fused_encode_dispatch` evaluation

Profile the prefill-encode kernels (`fused_encode`, `fused_encode_wht`) under Instruments. The Walsh-Hadamard butterfly is bit-shuffle-heavy and may not benefit from MMA conversion. Decide per-instantiation; if profiling shows ≥ 10% headroom from threadgroup-shared rotation matrices or vec-load coalescing, land that. Otherwise skip.

## Test plan

### Correctness

- For each converted kernel, add a `_simd` variant test in `TurboQuantKernelTests.swift` that runs the new MMA kernel against the existing scalar reference on identical inputs and asserts `allClose(rtol: 1e-2, atol: 1e-3)`. Same harness pattern PR #99 already uses for `turbo_dequant_rotated`'s metal-vs-ref tests.
- WikiText-2 word-level PPL on `qwen35-{0.8b, 9b}` and `gemma4-{e2b, 31b}` × `--kv turbo4v2` ctx ∈ {2048, 8192, 32768}. Assert ≤ 1% relative drift vs the scalar reference.

### Performance

- Per-shape bench sweep: `--method summarization --quick` × `--model {qwen35-0.8b, qwen35-9b, qwen35-35b-a3b, gemma4-{e2b, 26b-a4b, 31b}, gpt-oss-20b, nemotron-30b-a3b}` × `--kv turbo4v2`.
- Acceptance: each converted kernel matches or beats both (a) its scalar predecessor and (b) Apple's dequant-first equivalent. The "match or beat both" gate is what flips the default-routing decision in `compressedAttention`.

### Hardware coverage

- M1 Pro / M1 Max / M2 Max / M3 Max / M4 Max (when available). Apple's MMA hardware support varies: M1 has limited SME-like operations, M2+ has full matrix-engine. The kernel templates should branch on `__METAL_VERSION__` + GPU family at compile time and select scalar fallback on unsupported hardware. Test the fallback path under `MTL_GPU_FAMILY_APPLE_7` simulation.

## Risk / open questions

1. **Register pressure with dequant inline.** Apple's SDPA tile (`Bq = 32`) assumes K is already FP16. With dequant in-kernel (codebook lookup + per-token norm multiply), register pressure goes up. May force tile shrink, which costs some throughput. Mitigation: profile per-tile-size, pick the optimum per `(headDim, bits)` combo, codegen the instantiation matrix.

2. **M1 family compatibility.** M1 lacks SME / full matrix-engine; spec mandates scalar fallback when `__metal_arch__ < 7.1` (or similar gate). Verify with the fallback path tests; the M1 case is acceptable to be slower than dequant-first as long as it's coherent.

3. **Codebook fits in threadgroup memory.** For 4-bit (16 entries × 2 bytes = 32 bytes) and 8-bit (256 × 2 = 512 bytes) the codebook is comfortably small. For future larger codebooks (e.g. 12-bit MSE), 4096 entries × 2 = 8 KiB, still within Apple's 32 KiB threadgroup-memory budget. No issue.

4. **Cross-repo PR ordering.** Same shape as spec 020 / spec 6c chain — `mlx` kernel changes → `mlx-c` ABI bumps (if any) → `mlx-swift` wrapper → `mlx-swift-lm` callsite. Each PR independent up to the previous submodule bump.

## Estimated scope

| Phase | Effort | Calendar |
|---|---|---|
| 1 — TurboFlash MMA | ~2 weeks single-engineer | After spec 041 Phase 1 lands (or in parallel — they touch the same kernel files but spec 041 is the architecture; this one is the SIMD tuning). |
| 2 — `turbo_dequant_rotated` MMA | ~3 days | Likely skippable post-Phase 1 of spec 041. |
| 3 — fallback kernels | ~3 days | Low priority. |
| 4 — encode-path profiling + tuning | ~3 days | Bench-driven, may yield no work. |

Total: ~3 weeks for the full sweep. Phase 1 alone closes the practical gap between hand-rolled turbo kernels and Apple's first-party SDPA. Combined with spec 041 Phase 1, gives us a unified compressed-attention path that's competitive with `--kv none` on speed across every supported model family.
