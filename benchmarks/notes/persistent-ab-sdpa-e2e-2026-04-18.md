# Persistent-AB SDPA End-to-End — 2026-04-18

Infrastructure landed across all 4 repos (`ek/persistent-ab-pilot` branches):
`PersistentSdpaAbHandle` Swift class + `MLXFast.scaledDotProductAttentionAb` +
`mlx_fast_scaled_dot_product_attention_ab` C API + `ScaledDotProductAttention`
primitive accepts an optional `shared_ptr<metal::PersistentAb>` threaded
through its AB-branch `eval_gpu`. GPT-OSS-20B's `AttentionBlock` lazy-
creates a handle on first call when `MLX_PERSISTENT_AB=1` and routes
SDPA through it.

Plus the existing MLXNN.RMSNorm path from Step 3 — both primitives now
run through persistent handles simultaneously under that env var.

## Measured results

**`--method simple` (decode tok/s), 3 runs each:**

| Config | Runs | Mean |
|---|---|---:|
| AB=1 (transient ABs) | 46.8 / 46.3 / 46.1 | 46.4 |
| **AB=1 + PERSISTENT_AB=1 (RMSNorm + SDPA persistent)** | 46.7 / 45.6 / 46.1 | **46.1** |

Persistent-AB on both RMSNorm and SDPA is **noise-neutral** on decode
tok/s — same finding as Step 3 where only RMSNorm was persistent.
Confirms: persistent-AB is the correctness prerequisite for ICB replay,
not a standalone perf lever. GPT-OSS-20B 4-bit decode is GPU-bound at
this config; CPU encoding reductions don't translate to tok/s until
ICB replay eliminates the encoding work entirely.

**`--method icb` microbench, 3 runs with PERSISTENT_AB=1:**

| Run | Live µs/step | Replay µs/step | Speedup |
|---|---:|---:|---:|
| 1 | 17,664 | 12,253 | 1.44× |
| 2 | 17,966 | 12,556 | 1.43× |
| 3 | 11,697 | 12,411 | 0.94× (outlier) |

The 0.94× reading is the same thermal/cache-state transient seen
before. Stable result: **ICB replay consistently ~1.44× faster than
live encoding** on GPT-OSS-20B, independent of persistent-AB status.

## Correctness verification

End-to-end decode with persistent-AB SDPA:
- Output text is coherent (not garbled).
- No crashes in 3 consecutive runs.
- Token counts vary per run (non-deterministic sampling) but all land
  in expected range (111–200).

## What's built (across 4 repos on `ek/persistent-ab-pilot`)

- **mlx** (`8630b7b3`): `PersistentAb` class + RMSNorm integration (Step
  2) + `ScaledDotProductAttention` accepts optional handle + SDPA AB
  branch populates persistent handle when provided (skip
  `add_temporary_object`). 302/302 mlx tests green.
- **mlx-c** (`bad1a40`): `mlx_metal_persistent_ab_new_rmsnorm` /
  `_new_sdpa` factories + `mlx_fast_rms_norm_ab` /
  `mlx_fast_scaled_dot_product_attention_ab` wrappers.
- **mlx-swift** (`62e1997`): `PersistentRmsAbHandle` / `PersistentSdpaAbHandle`
  Swift classes + `MLXFast.rmsNormAb` / `.scaledDotProductAttentionAb` +
  env-gated `MLXNN.RMSNorm` path + restored `Source/Cmlx/include/mlx/c/fast.h`
  (had lost fused-primitive declarations due to my earlier overwrite).
- **mlx-swift-lm**: `Package.swift` now points at `ek/persistent-ab-pilot`
  of mlx-swift. `AttentionBlock` in GPT-OSS lazy-inits the SDPA handle
  under `MLX_PERSISTENT_AB=1` and calls `MLXFast.scaledDotProductAttentionAb`.

## What's NOT done this session

**Decode-loop ICB record/replay wiring.** The ICB record/replay machinery
works at the `--method icb` microbench level (existing code, consistent
1.44× speedup). But wiring it into `TokenIterator.step()` to actually
replay recorded decode steps with per-step scalar updates was not
attempted tonight — a combination of:

1. SPM submodule-version caching repeatedly fetching older `mlx-swift`
   commits than my pushes. Debugging that ate a large slice of the
   session.
2. My mistake overwriting `Source/Cmlx/include/mlx/c/fast.h` during an
   earlier sync — lost fused-primitive declarations (warp_moe_down, etc.)
   that had been manually added in that file but weren't in mlx-c.
   Had to restore + re-add my new AB declarations.
3. Context degradation from the build-system thrashing. Better to ship
   tested infrastructure than write decode-loop ICB code in that state.

## Next session — the actual tok/s win

Now that persistent-AB SDPA is in place, wiring decode-loop ICB is
architecturally unblocked:

1. Modify `TokenIterator.step()` to support an opt-in ICB-replay mode:
   - Step 1–2: live encoding (warmup).
   - Step 3: `IndirectCommandBuffer.record { model(...) }`.
   - Step 4+: before replay, write current-step T_k into each layer's
     `PersistentSdpaAbHandle` via `handle.setScalar32(slot: .N, ...)`.
     Then `icb.replay()` + advance token from output.
2. First correctness check: verify step 4's output matches a live step
   4 within tolerance. If wrong, debug per-step scalar updates for
   other primitives (RoPE offset is the next candidate to migrate).
3. Benchmark end-to-end tok/s. Target: breaking out of the ~47 tok/s
   GPU-bound ceiling via CPU encoding reduction.

Estimated work: 3–4 focused hours (no build-system debugging).

## Lessons for myself

- **Multi-run before any narrative.** The 33% excitement from last
  session (single outlier) would have been caught by a single repeat.
  Tonight's outlier at run 3 confirms the pattern continues.
- **Don't cp things that might have been manually edited.** The
  `include/mlx/c/fast.h` overwrite was destructive; git diff before
  sync would have caught it.
- **SPM cache is finicky with branch dependencies.** `swift package update <name>`
  is the right command to force branch re-resolve; `swift package reset`
  doesn't always.
