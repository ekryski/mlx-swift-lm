# 043 — TurboFlash decode-time kernel uplift

- **Status:** Spec drafted 2026-05-14. **Not started.**
- **Branch:** TBD (`ek/spec-043-turboflash-kernel-uplift-phase1` once implementation begins; this branch carries the spec only).
- **Depends on:** none structurally. Composes with [spec 042](042-metal-kernel-simd-audit.md)'s broader Metal-kernel SIMD audit; spec 043 is the focused, bench-data-driven sub-set of optimisations that the 2026-05-13/14 A-vs-B sweep flagged as the highest-leverage TurboFlash wins.
- **Parent decision:** [`benchmarks/m1-max-64gb-2026-05-13.A-path.md`](../benchmarks/m1-max-64gb-2026-05-13.A-path.md) and [`m1-max-64gb-2026-05-13.B-path.md`](../benchmarks/m1-max-64gb-2026-05-13.B-path.md) (committed 2026-05-14 at `f4d13bc`).

## Problem

The 2026-05-14 A-vs-B sweep across 13 models × 6 KV configs × {1k, 8k} contexts validated the architectural decision in commit `85afa9b` to make TurboFlash the unconditional default A path. **It also quantified the speed cost we accepted by retiring the dequant-SDPA fast path as the default.**

Headline turbo* decode tok/s deltas (TurboFlash vs dequant-SDPA, mean across `turbo4` / `turbo4v2` / `turbo8v4` cells):

| Model class                       | 1k mean Δ | 8k mean Δ | 8k worst cell |
|-----------------------------------|----------:|----------:|--------------:|
| Large dense (27B, 31B)            |  -6 / -8% |   -17%    | -29% (turbo8v4)|
| MoE (Nemotron-30B-A3B, 35B-A3B)   |  -5 / -17%|  -15 / -28%| -40% (turbo8v4)|
| Mid (4B, 9B)                      | -11 / -13%|  -21 / -28%| -39% (turbo8v4)|
| Small dense (E2B, E4B, 0.8B, 2B)  | -20 / -28%|  -33 / -41%| **-56%** (Qwen 0.8B turbo8v4 8k)|

Three patterns the data exposes:

1. **Smaller models suffer most.** Matmul-engine SDPA dominates when the model is too small to be FLOP-bound. TurboFlash's per-token bit-unpack overhead is most visible in this regime.
2. **Long context amplifies the regression.** Every model regresses 10-20 pp more at 8k than at 1k because the per-decode-step compressed-domain scan is linear in cached tokens, with no matmul-engine amortisation.
3. **turbo8v4 (8-bit K, 4-bit V) is the worst sub-regime.** The 8-bit K unpack has the most per-attention-step work. turbo8v4 8k is the worst cell on 9 of 13 models.

- **Goal:** close the practical gap between TurboFlash and dequant-SDPA via three concrete Metal-kernel optimisations, in the order their bench evidence suggests will pay off. Acceptance gate: TurboFlash decode tok/s ≥ 90% of dequant-SDPA decode tok/s on the previously-worst cells (Qwen 0.8B turbo8v4 8k, Gemma 4 31B turbo8v4 8k).

  This spec complements [spec 042](042-metal-kernel-simd-audit.md) — that one is the broad audit; this one is the focused, bench-driven follow-up that lands the three highest-leverage items immediately.

## Three phases

### Phase 1 — Per-simdgroup bit-unpack reuse (highest leverage)

**Current state.** `turbo_flash_sdpa.metal` and `turbo_flash.metal` unpack the bit-packed K (and V) on every thread that touches a token's data. Each SIMD lane re-runs the same bit-extract + codebook lookup + norm multiply for the same packed K word. At ctx=8192, headDim=128, 4-bit K, that's ~262K redundant unpack operations per attention step per layer.

**Change.** Cache the unpacked K block in threadgroup memory once per tile, then have all SIMD lanes within the tile read from the cache instead of re-unpacking. Same pattern Apple's `quantized_matmul.metal` uses for its codebook lookup — one thread per simdgroup pulls + unpacks, broadcast via threadgroup memory, all 32 lanes consume.

Math: per tile of `Bk` tokens × `headDim` dims, current scheme does `Bk × headDim × 32` unpack ops (every lane × every element); cached scheme does `Bk × headDim` unpack ops (one lane per element) + `Bk × headDim × 31` threadgroup loads. Threadgroup loads are ~10× cheaper than bit-extract + codebook lookup, so net cost drops ~30×.

**Why this is Phase 1.** This is the only optimisation that directly attacks the "per-token bit-unpack overhead grows linearly with context" pattern that the 8k regression highlights. Every other optimisation helps a constant factor; this one bends the slope.

**Files (in `ekryski/mlx` fork):**
- `mlx/mlx/backend/metal/kernels/turbo_flash_sdpa.metal` — main `turbo_flash_sdpa_v` kernel (sinks path)
- `mlx/mlx/backend/metal/kernels/turbo_flash.metal` — `turbo_flash_attention` and `turbo_flash_attention_causal`

**Cross-repo PR shape:** mlx kernel + C++ Primitive → mlx-c bridge (no ABI bump expected; same callsite) → mlx-swift submodule bump → mlx-swift-lm consumes via existing `MLXFast.turboFlashSDPAv` / `turboFlashAttention` wrappers (no Swift API changes).

- **Expected lift:** 20-40% on 8k turbo* cells where bit-unpack dominates. Worst-case Qwen 0.8B turbo8v4 8k (-56% today) should land at -25 to -30%.

- **Acceptance gate:**
  - Correctness: `testTurboFlashSDPAvSinksSlidingWindow`, `testTurboFlashSDPAvNoSinksMatchesTurboFlashAttention`, `testTurboQuantCompressedAttentionSinksMatchesReference` all pass with `rtol: 1e-2, atol: 1e-3` against scalar reference (existing harness).
  - Performance: Qwen 9B turbo4v2 8k decode tok/s ≥ 38 (today: 36.4, B path: 43.0 → 90% gate = 38.7).
  - Hardware coverage: M1 Pro / M1 Max / M2 Max minimum; gate on `__metal_arch__ ≥ 7.0` so older M1-family hardware falls back to current scalar path.

### Phase 2 — bf16 V accumulator (with fp32 softmax m/l)

**Current state.** `turbo_flash_sdpa.metal` accumulates the per-tile `o[DIMS_PER_LANE]` value-aggregation tensor in `float`. Per [planning note `simd-analysis-and-remaining-gains.md`](../sam/planning/performance-notes/simd-analysis-and-remaining-gains.md), this is over-precise: V values are bf16-storage post-codec, and a fp16 accumulator (with fp32 softmax m/l for stability) is documented to give a 5-8% throughput lift without measurable PPL drift.

**Change.** Convert the per-lane V accumulator + the `weights × V` inner product from `float` to `half` (FP16). **Crucially leave softmax `m` (running max) and `l` (denominator) in `float`** — the online-softmax dynamic range needs fp32 headroom or the long-context cases drift.

Same pattern spec 042 §7b calls out. The reason this gets its own phase here (rather than waiting for the broad audit) is the 2026-05-13/14 sweep shows the V-accumulator path is hottest precisely on the small-model long-context cells where TurboFlash regresses worst. Direct hit on the regression curve.

- **Files:** same kernels as Phase 1. Change is a `typedef float ACC_T;` → `typedef half ACC_T;` swap (plus careful audit that softmax `m`, `l` are explicitly `float` and not derived from `ACC_T`).

- **Expected lift:** 5-10% across the board on turbo* cells. Compounds with Phase 1's lift on long-context cells (where the V accumulator gets more work per token).

- **Acceptance gate:**
  - Correctness: all Phase 1 tests pass with **tighter** tolerance after the conversion — `rtol: 5e-3, atol: 5e-4` because reducing accumulator precision should produce identical-up-to-FP-error outputs. WikiText-2 PPL on `qwen35-{0.8b, 9b}` and `gemma4-31b` × `--kv turbo4v2` ctx ∈ {2048, 8192} must drift ≤ 0.5% relative vs Phase 1 baseline.
  - Performance: ≥ 5% mean decode tok/s improvement over Phase 1 (the issue [#158](https://github.com/ekryski/mlx-swift-lm/issues/158) gate).

### Phase 3 — headDim-aware tile autotune

**Current state.** TurboFlash's `flashBlockSize` adapts on `tokenCount` (`TurboQuantKernels.swift:1372` — `tokenCount/32`, clamped to `[16, 256]`, rounded to power of two). It does **not** adapt on `headDim`. Defaults are tuned for `headDim=128` (Qwen, Gemma 4, Nemotron).

The bench shows two consequences:
- **Qwen 0.8B / 2B (headDim=64)** are dramatically over-tiled. Lanes go idle for ~half the tile because the kernel was written assuming `headDim=128`. This is the leading explanation for the -56% / -43% worst-cells.
- **Qwen 27B / 36 27B (headDim=256)** are under-tiled. Tile geometry doesn't take advantage of the extra register space.

**Change.** Add a `(headDim) → (NR0, blockSize)` static table picked at kernel-dispatch time. Each `headDim` value gets its own pair tuned via a per-shape micro-sweep on M1 Max + M2 Max. Concrete starting points (to be confirmed empirically):

| headDim | NR0 | blockSize at ctx ≤ 4k | blockSize at ctx > 4k | Rationale |
|--------:|----:|----------------------:|----------------------:|-----------|
| 64      | 4   | 32                    | 64                    | Halve dim, double rows-per-SIMD, smaller blocks because each tile is "half size". |
| 128     | 2   | 64                    | 128                   | Current defaults — already validated by today's sweep on Qwen 4B-9B. |
| 256     | 2   | 128                   | 256                   | Bigger headDim has more elements per dim — bigger tiles amortise the K-load cost. |

Plus a `--kv turbo8v4`-specific override: 8-bit K means 2× the unpack work, so prefer smaller blocks even at long context (gives Phase 1's threadgroup unpack cache more re-use per cycle).

- **Files:**
  - `mlx/mlx/backend/metal/kernels/turbo_flash_sdpa.metal` (the kernel side — read the new dispatch params)
  - `Libraries/MLXLMCommon/TurboQuantKernels.swift:1372` (the Swift dispatch side — emit the new `(NR0, blockSize)` based on `headDim`)

- **Expected lift:** 10-25% on small-model turbo* cells (where current over-tiling is the bottleneck). Smaller effect on large models; net effect on Qwen 27B is probably flat.

- **Acceptance gate:**
  - Correctness: same kernel-equivalence tests as Phase 1 pass.
  - Performance: Qwen 0.8B / 2B turbo8v4 8k decode tok/s improves by ≥ 25% over Phase 2 baseline (the worst-case cells we're specifically targeting).
  - No regressions on the cells where current defaults already work: Qwen 4B / 9B turbo* must stay within ±3% of Phase 2 baseline.

### Phase 4 — Bias-aware TurboFlash kernel (unlock GPT-OSS-20B on the A path)

**Current state.** GPT-OSS-20B sets `useBias: true` by default ([`GPTOSSModel.newCache(...)`](../Libraries/MLXLLM/Models/GPTOSS.swift), shipped at commit `93f191b`) because the zero-mean Lloyd-Max codebook can't represent the structured DC offset that K/V from `RMSNorm → Linear(bias=True)` projections carry. Without bias correction the output is incoherent (the original GPT-OSS turbo failure mode tracked in [#171](https://github.com/ekryski/mlx-swift-lm/issues/171) / [#130](https://github.com/ekryski/mlx-swift-lm/issues/130)).

The bias correction adds `b[t] * rotatedOnes[d]` to the rotated K / V reconstruction:

```
K_rotated[t, d] = codebook[idx[t, d]] * norm[t] + bias[t] * rotatedOnes[d]
```

This add is currently Swift-side, inside the **B path** (dequant-then-SDPA), at [`TurboQuantKVCache.swift`](../Libraries/MLXLMCommon/TurboQuantKVCache.swift):1955-1965. The A-path kernels (`turbo_flash_sdpa_v`, `turbo_flash_p1`, `turbo_flash_p1_causal`) don't accept the bias inputs, so anything with `useBias: true` is forced to route through B regardless of `TURBO_DEQUANT_SDPA`. Bench confirms: GPT-OSS-20B A=68.5 vs B=68.0 tok/s on turbo4v2 1k (within 1% — they're running the same B path).

**Change.** Teach the TurboFlash kernels to consume the bias term. Add four new kernel inputs per cache (two for K, two for V):

- `key_bias [B, nKV, T]` and `val_bias [B, nKV, T]` — per-vector DC offsets, fp32.
- `key_rotated_ones [Dim]` and `val_rotated_ones [Dim]` — precomputed `1 @ rotation^T` per codec, constant per `(dim, seed)`. Small enough to live in threadgroup memory.

Inside the reconstruction step, replace `codebook[idx] * norm` with `codebook[idx] * norm + bias[t] * rotated_ones[d]`. Two extra ops per element; both are fp32 mul-add, fold naturally into the existing dot-product accumulator.

**Why this is Phase 4 (not Phase 1).** GPT-OSS-20B currently has a working path (B). The principled win — moving it to the truly compressed-domain A path — is real but smaller than the Phase 1-3 wins for the rest of the model matrix (GPT-OSS is one of 13 models). Ordering it after the broader kernel uplift means GPT-OSS picks up Phase 1-3's wins *and* moves to A path in one swoop.

**Files (in `ekryski/mlx` fork):**
- `mlx/mlx/backend/metal/kernels/turbo_flash_sdpa.metal` — primary target (GPT-OSS uses the sinks variant)
- `mlx/mlx/backend/metal/kernels/turbo_flash.metal` — `turbo_flash_p1` / `_causal` / `_nr0` / `_nr0_causal` (for non-sinks bias users; rare today but the codepath should be symmetric)
- New kernel instantiation suffix: `turbo_flash_sdpa_v_bias_<kb>_<vb>_<dim>` so the no-bias kernel stays cheap (skips the extra mul-adds).

**Cross-repo PR shape:** mlx kernel + C++ Primitive (new with-bias variant) → mlx-c bridge → mlx-swift wrapper (`MLXFast.turboFlashSDPAv(..., keyBias:, valBias:, keyRotatedOnes:, valRotatedOnes:)`) → mlx-swift-lm dispatcher chooses the bias variant when `useBias == true`.

**Closed PRs to NOT reopen.** [mlx#16](https://github.com/ekryski/mlx/pull/16), [mlx-c#8](https://github.com/ekryski/mlx-c/pull/8), [mlx-swift#18](https://github.com/ekryski/mlx-swift/pull/18), [mlx-swift-lm#99](https://github.com/ekryski/mlx-swift-lm/pull/99) tried to fix a different problem (the original two-pass pass2 sinks-fold bug). That's already solved by `turbo_flash_sdpa_v`'s single-pass design. Phase 4 is new kernel work — adding bias-aware reconstruction — not a sinks-fold redo.

**Expected lift.** GPT-OSS-20B decode tok/s should land between today's A and B numbers (currently identical), with the actual win arriving when Phases 1-3 give A path's underlying kernel ≥ B path's performance. Net: Phases 1-3's full lift becomes available to GPT-OSS-20B.

- **Acceptance gate:**
  - Correctness: extend `testTurboQuantCompressedAttentionSinksMatchesReference` to a `_withBias` variant that compares the new kernel output against the Swift-side `bulkDequantRotated + bias + MLXFast SDPA` reference. Tolerance `rtol: 1e-2, atol: 1e-3` (same as existing sinks regression test).
  - End-to-end: GPT-OSS-20B with `--kv turbo4v2` and `TURBO_DEQUANT_SDPA=0` produces coherent Harmony channel preambles at 1k + 8k (today this requires `TURBO_DEQUANT_SDPA=1` or bias-forces-B by default).
  - Performance: GPT-OSS-20B A-path decode tok/s ≥ B-path decode tok/s after Phases 1-3 land. (Before Phases 1-3 it'll match B by definition since the kernel work is on the same SIMD-suboptimal pre-Phase-1-3 baseline.)

**Estimated scope.** ~1 week kernel work + ~3 days cross-repo PR plumbing. Total ~1.5 weeks. Can run in parallel with Phase 1-3 implementation since the kernel files overlap but the changes are additive (new instantiations + new arguments, no template restructure).

## Test plan

### Correctness

For each phase, add a `_phaseN` test variant in `Tests/MLXLMTests/TurboQuantKernelTests.swift` that exercises the new kernel against the scalar reference. The existing `testTurboFlashSDPAvSinksSlidingWindow` and `testTurboFlashSDPAvNoSinksMatchesTurboFlashAttention` already provide the harness — just add a `phase = X` parameter and assert at the tighter tolerance documented per phase.

WikiText-2 PPL gate after each phase: `swift test --filter testWikiText2PPL` on `qwen35-{0.8b, 9b}` and `gemma4-31b` × `--kv turbo4v2` at ctx ∈ {2048, 8192}. Assert relative drift ≤ 1.0% vs the previous phase.

### Performance

Per-phase bench sweep using the same script that produced today's archives:

```sh
./scripts/benchmark.sh \
  --model qwen35-0.8b,qwen35-2b,qwen35-4b,qwen35-9b,gemma4-e2b,gemma4-e4b,nemotron-30b-a3b,gpt-oss-20b,qwen35-35b-a3b,gemma4-26b-a4b,qwen35-27b,qwen36-27b,gemma4-31b \
  --method summarization --quant 4bit \
  --kv turbo4,turbo4v2,turbo8v4 \
  --context 1024,8192
```

156 turbo cells (39 cells × 2 contexts × 2 KV configs ≈ 156). Run once per phase, name the output file `m1-max-64gb-<date>.spec043-phaseN.md`. Compare against today's A-path archive at HEAD; assert no regression on any model > 3% and the per-phase performance gate above.

### Hardware coverage

- M1 Max 64GB (primary dev hardware): full sweep per phase.
- M2 Max 96GB if available — `simdgroup_matrix_*` intrinsics gated on M2+; verify scalar fallback on M1 (Phase 1 uses these intrinsics).
- M3 Max / M4 Max: smoke test on one model (Qwen 9B turbo4v2 8k) per phase, confirm no hardware-family-specific regression.

## Risk / open questions

1. **Threadgroup memory pressure (Phase 1).** Caching the unpacked K block costs `Bk × headDim × 2` bytes of threadgroup memory (FP16). For `Bk=64, headDim=128`: 16 KiB — half the Apple threadgroup budget. For `headDim=256` × `Bk=128`: 64 KiB → over budget. Mitigation: per-`(headDim, blockSize)` static check at dispatch time, fall back to no-cache scalar path when budget exceeds. The Phase 3 tile autotune already picks small `Bk` for `headDim=256`, so the natural intersection is fine.

2. **Phase 2 V-accumulator drift on Gemma 4 31B at 32k ctx.** Spec 042 §7b's note flagged that some long-context regimes need fp32 headroom even on the V accumulator. Run a paranoia check on Gemma 4 31B `--kv turbo4v2` ctx=32k after Phase 2; if PPL drift > 1%, gate the fp16 V accumulator on `headDim < 256` (keep fp32 on the headDim=256 Qwen-27B-class models).

3. **Phase 3 tile-table is hardware-specific.** The starting-point table is M1 Max numbers. M2+ has higher register count and may prefer larger tiles. Use the per-phase sweep to validate; if M2 Max wants different defaults, branch on GPU family at dispatch (similar to `MLX_MAX_OPS_PER_BUFFER`'s per-family logic).

4. **Compounding lifts may not be linear.** Phases 1 + 2 + 3 don't multiply — they each compete for the same cycles. Plan accordingly: target ≥ 35% lift after all three phases, not "20% × 7% × 15% = 1.4x".

5. **Cross-repo PR ordering.** Same chain as prior TurboFlash work — `mlx` kernel changes land first, then `mlx-swift` submodule bump, then `mlx-swift-lm` Swift-side dispatch changes (Phase 3 is the only one with mlx-swift-lm-side code). Each phase is its own PR pair; phases 1 + 2 can ship in one mlx PR if implementation overlaps, with separate acceptance bench-runs.

## Estimated scope

| Phase | Effort | Calendar |
|-------|--------|----------|
| 1 — Bit-unpack reuse | ~1.5 weeks single-engineer | Land first — highest leverage, no Swift-side change |
| 2 — bf16 V accumulator | ~3 days | After Phase 1 lands and bench-validates |
| 3 — headDim tile autotune | ~1 week (sweep-heavy) | After Phase 2 — wants both prior phases as baseline |
| 4 — Bias-aware kernel (GPT-OSS-20B on A path) | ~1.5 weeks | Can run in parallel with Phases 1-3 (kernel changes are additive, no template restructure) |

Total: ~4.5 weeks for the full uplift if Phase 4 is serialised after Phase 3; ~3.5 weeks if Phase 4 lands in parallel. Phase 1 alone should close the gap to dequant-SDPA on most models (mid-class and larger). Phases 2 + 3 mop up the small-model long-context cases. Phase 4 unlocks GPT-OSS-20B (and any future sinks-using bias-correcting models) on the A path.

## Success criteria

After all four phases:

- TurboFlash decode tok/s ≥ 90% of dequant-SDPA decode tok/s on every cell of the 156-cell matrix (today: 60-99% depending on cell).
- Per-model worst-cell delta improves from -56% (Qwen 0.8B turbo8v4 8k) to ≤ -15%.
- Mean turbo* decode-tok/s regression across all 13 models drops from -22% (today) to ≤ -8%.
- GPT-OSS-20B `--kv turbo4v2` runs coherent on the A path with `TURBO_DEQUANT_SDPA=0` (i.e., `useBias: true` no longer forces B). Decode tok/s ≥ B-path baseline.
- WikiText-2 PPL on Qwen 9B turbo4v2 ctx=8192 drifts ≤ 1.5% relative vs HEAD baseline.
- Memory invariant unchanged — same packed K/V storage, no per-step working buffer materialisation.

Once these gates land, the architectural decision in commit `85afa9b` (TurboFlash as the unconditional default) stops costing speed. The B path stays in the codebase as a documented escape hatch but stops being the "fast" option that anyone realistic would reach for.
