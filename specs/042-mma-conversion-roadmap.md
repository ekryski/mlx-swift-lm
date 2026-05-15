# Spec 042 — MMA conversion roadmap

**Branch:** `ek/specs-042-043-kernel-uplift`
**Date:** 2026-05-15
**Status:** Design + per-kernel sketches. Implementation deferred to M2+ session.
**Companion to:** [`042-precision-audit.md`](042-precision-audit.md) (§7 precision work — shipped).

## What this doc is

A concrete per-kernel implementation sketch for spec 042's MMA
(matrix-engine multiply-accumulate) conversion work. Captures:

- Where in each kernel the matmul-shaped inner loop lives
- The exact `simdgroup_matrix_*` tile geometry to use
- How dequant + codebook lookup gets staged into MMA tiles
- The function-constant gate that lets M1 keep scalar / M2+ take MMA

Implementation is deferred because:

1. **M1 limited MMA support.** Per spec 042 §1: "M1 has limited support
   but the kernel template should branch on hardware capability rather
   than always using scalar." MMA code compiles + runs correctly on M1
   (via software emulation) but won't hit the spec's 8× lift target —
   that requires M2+ matrix engine.

2. **Validation needs perf, not just correctness.** Unit tests can
   confirm MMA output matches scalar reference on M1 (since both
   execute the same math just via different intrinsics). The
   acceptance gates (≥ 90% of dequant-SDPA throughput) need M2+ to
   validate.

3. **Restructure cost vs M1-actionable win.** Each kernel MMA conversion
   is days of careful work: tile geometry, dequant staging into half
   tiles, boundary cases for non-multiple dims, threadgroup-memory
   accounting. On M1 this work yields no measurable lift; on M2+ it
   unlocks the spec's primary target. Better to validate on the
   target hardware than to ship blind code.

## Common pattern

Every MMA-converted kernel follows the same scaffolding:

```cpp
// Function constant — default scalar; M2+ dispatcher flips to true.
constant bool tf_use_mma [[function_constant(63)]];

template <int KeyBits, int Dim, int PackedWidth>
[[kernel]] void some_kernel(...) {
    if (tf_use_mma) {
        // ── MMA path ──
        // Tile geometry: 8×8 half MMA tiles, K=8 reduction step
        simdgroup_matrix<half, 8, 8> Q_tile;
        simdgroup_matrix<half, 8, 8> K_tile;
        simdgroup_matrix<float, 8, 8> S_acc(0);

        for (int k_step = 0; k_step < Dim; k_step += 8) {
            // 1) Cooperative dequant K[*, k_step:k_step+8] into K_tile
            //    Pack-extract + codebook lookup + norm multiply per dim
            //    Store into 8x8 tile via simdgroup_store on TG buffer
            // 2) Load Q tile (already dequanted)
            simdgroup_load(Q_tile, q_buf + k_step, Dim);
            simdgroup_load(K_tile, k_tile_tg, 8);
            // 3) MMA accumulate
            simdgroup_multiply_accumulate(S_acc, Q_tile, K_tile);
        }

        // Write S_acc to output
        simdgroup_store(S_acc, scores_out + tile_row * stride, stride);
    } else {
        // ── Scalar path (current implementation) ──
        // ... existing code ...
    }
}
```

The `tf_use_mma` function constant is set by the C++ Primitive at
dispatch time, e.g.:

```cpp
bool use_mma = (gpu_family >= GPUFamily::Apple8); // M2+
metal::MTLFCList func_consts = {
    {&use_mma, MTL::DataType::DataTypeBool, 63},
    // ... existing constants ...
};
```

## Phase 3 — `turbo_score` MMA

**Current shape:** dispatch `(32, num_q, num_k)`, each threadgroup
computes one score cell via `q · k` + `simd_sum`.

**MMA shape:** dispatch `(num_q_tiles, num_k_tiles)`, each threadgroup
computes an 8×8 (or 16×16) score tile.

**Tile geometry:**
- M (queries per tile): 8
- N (keys per tile): 8
- K (dim chunk per MMA step): 8

For Dim=128: 16 MMA steps per (M-tile, N-tile) pair.

**Dequant staging:**

K positions in the tile need packed → dequant per dim. Per K-tile (8
keys × Dim dims):

1. Cooperative load packed K[k_start:k_start+8, :] into TG memory
2. Per (key, dim) thread: bit-extract → `tg_codebook[idx] * norm[key]`
3. Stage 8×8 chunks into a `simdgroup_matrix<half, 8, 8>` via
   `simdgroup_store` to a TG-memory scratch buffer
4. `simdgroup_load` back into the matrix

The codebook hoist + per-simdgroup TG cache from spec 043 Phase 1 +
spec 042 §7b already provides the staging primitives.

**Effort estimate:** ~3 days (1 day dequant staging, 1 day MMA core, 1
day instantiation matrix + cross-shape validation).

**Acceptance gate:**
- Correctness: matches scalar reference at `rtol < 1e-3` for `(bits,
  dim) ∈ {(4, 64), (4, 128), (8, 128), (3, 96)}`
- Perf (M2+ only): ≥ 1.5× scalar `turbo_score` decode tok/s on a
  benchmark hit (rarely-triggered fallback, so absolute numbers
  matter less than the multiplier)

## Phase 3 — `turbo_value` MMA

**Current shape:** dispatch `(32, num_q_heads, num_dim_blocks)`, each
thread computes one output dim via the weighted-sum loop over tokens.

**MMA shape:** the `weights × V` matmul is the inner kernel. With
weights `[nQ, T]` and V `[nKV, T, Dim]`, output is `[nQ, Dim]`.

**Tile geometry:**
- M (queries per tile): 8
- N (output dims per tile): 8
- K (tokens per MMA step): 8

For T=8192: 1024 MMA steps per (M-tile, N-tile) pair. Expensive but
amortises the V dequant cost (V dequanted once per K-step, reused
across 8 queries).

**Dequant staging:**

Per K-step (8 tokens × 8 dims of V):

1. Cooperative load weights[m_tile_start:m_tile_start+8, k_step:k_step+8]
2. Cooperative dequant V[k_step:k_step+8, n_tile_start:n_tile_start+8]
3. MMA accumulate `out_tile += W_tile @ V_tile`

**Effort estimate:** ~3 days. Same complexity as `turbo_score`.

## Phase 2 — `turbo_dequant_rotated`

**Not a matmul kernel** — pure bit-extract + codebook lookup + norm
multiply, written to a strided output buffer. No `Q·K^T` or
`weights·V` inner loop.

**Decision:** skip MMA. The existing fp16 + bf16 instantiations cover
the M1-relevant precision work; MMA conversion has no leverage here.

## Phase 1a — TurboFlash MMA (`turbo_flash.metal` + `turbo_flash_sdpa.h`)

**Current shape:** TurboFlash is a fused score → softmax → V-aggregate
kernel. Per K-block, each simdgroup processes one query position
(or `NR0` queries in the `nr0` variants).

**MMA shape:** restructure to tile geometry along (queries × keys ×
dims) and (queries × dims × tokens) for the two matmul-shaped inner
loops.

This is the **largest restructure** in spec 042. The kernel currently
has:

```cpp
for (k_position in block) {
    // Score = q · k (simd_sum per lane)
    // Online softmax update (m, l, exp_score)
    // V_acc += exp_score * v
}
```

MMA version:

```cpp
simdgroup_matrix<float, 8, 8> S_tile_acc(0);
simdgroup_matrix<float, 8, 8> V_tile_acc(0);

for (k_tile_start in block, step 8) {
    // Stage K tile (8 keys × Dim) via cooperative dequant
    // S_partial[8 queries × 8 keys] += Q_tile @ K_tile^T  (multi-step over Dim/8)
    // Online softmax merge in TG memory (m, l per 8-key block)
    // Stage V tile (8 keys × Dim)
    // V_acc[8 queries × 8 dims] += exp(S - m) @ V_tile  (multi-step over Dim/8)
}
```

**Key challenge:** the online softmax requires cross-tile state (m, l
updates). The MMA reduction across tiles needs to thread the running
max + denominator carefully. Spec 042 §1 calls out this complexity:
"Apple's SDPA tile (Bq=32) assumes K is already FP16. With dequant
in-kernel (codebook lookup + per-token norm multiply), register
pressure goes up. May force tile shrink, which costs some throughput."

**Effort estimate:** ~2 weeks. The fact that Apple's first-party
`scaled_dot_product_attention` uses MMA but does NOT have the dequant
prologue is the load-bearing reason this kernel is hard to
MMA-convert cleanly.

**Acceptance gate (M2+):** ≥ 90% of dequant-SDPA throughput per the
spec.

## Phase 1b — Affine flash kernel (`flash_quantized_sdpa.{h,metal}`)

**Same template as Phase 1a**, applied to the affine equivalent. Only
the dequant prologue differs:
- Affine: `(packed × scale) + bias`
- Turbo: `codebook[idx] × norm`

Per spec 042 §1b: "Once Phase 1a's template lands, Phase 1b is a
same-pattern application — likely shares the same MMA tile scaffolding
via a `dequant` callback / functor template parameter."

**Effort estimate:** ~1 week after Phase 1a (template reuse).

**Acceptance gate:** auto-strategy's `MLX_AFFINE_SDPA` default flips
from `.flash` (dequant-then-MLXFastSDPA) to `.kernel` (fused) on
decode. Closes the 47% prefill / 18% decode regression spec 041 Phase
1.1 carried.

## Phase 4 — `fused_encode_dispatch` profiling

The Walsh-Hadamard butterfly inside `turbo_fused_encode_wht` is
bit-shuffle-heavy:

```cpp
// FWHT stage k butterfly:
// x[i] = x[i] + x[i ^ (1 << k)]  if (i & (1 << k)) == 0
// x[i] = x[i] - x[i ^ (1 << k)]  otherwise
```

This is not matmul-shaped — every butterfly stage is `Dim/2` add-or-
subtract ops on different element pairs. **MMA doesn't apply** — no
matrix multiplication to do.

**Profile-driven decision:** under Instruments / Metal System Trace,
look for:
1. Memory bandwidth: is the encode kernel device-bandwidth-limited or
   compute-limited?
2. Threadgroup memory contention: are the FWHT stages bank-conflicting?
3. Cooperative codebook hoist: would help the boundary-quantize step
   (`for (b in boundaries) compare(x, b)`).

**Conservative call:** skip MMA conversion; consider codebook +
boundaries hoist to TG memory as a §7b-style precision audit
follow-up. Defer until profiling shows ≥ 10% headroom per spec.

## Implementation order recommendation

When a session with M2+ hardware picks this up:

1. **Phase 3** (`turbo_score`, `turbo_value`) — simplest matmul shapes,
   smallest restructure. Validates the MMA staging pattern.

2. **Phase 1a** (TurboFlash) — once Phase 3's tile staging works, apply
   the same pattern with online-softmax additions. Biggest perf payoff
   per spec.

3. **Phase 1b** (Affine flash) — template reuse from 1a. Quick.

4. **Phase 4** (encode profiling) — bench-driven; may yield nothing.

5. **Phase 2** (`turbo_dequant_rotated`) — skipped (not matmul-shaped).

## Why this isn't a regression

The spec 042 §7b precision audit already shipped on this branch:

- `turbo_flash.metal` (4 templates): half codebook + fp16 V accumulator
- `turbo_flash_sdpa.h`: fp16 V accumulator (codebook fp32 — see §7b
  in-file comment)
- `turbo_quant.metal` (turbo_score, turbo_value): half codebook hoist
- `turbo_quant.metal` (turbo_flash_pass2 + _fused_rot): fp16 V
  accumulator
- `turbo_dequant_rotated`: already had bf16 + fp16 output variants

The MMA conversion is additive on top of the precision audit, not a
prerequisite. The audit covers all M1-relevant precision wins; the
MMA work covers M2+ matrix-engine wins. Two distinct deliverables.

## Cross-link

- [`042-precision-audit.md`](042-precision-audit.md) — §7 precision audit shipped
- [`benchmarks/spec-042-043-phase1-4-summary.md`](../benchmarks/spec-042-043-phase1-4-summary.md) — cumulative bench data
- Companion PRs:
  - [ekryski/mlx-c#18](https://github.com/ekryski/mlx-c/pull/18)
  - [ekryski/mlx#33](https://github.com/ekryski/mlx/pull/33)
  - [ekryski/mlx-swift#35](https://github.com/ekryski/mlx-swift/pull/35)
  - [ekryski/mlx-swift-lm#223](https://github.com/ekryski/mlx-swift-lm/pull/223)
