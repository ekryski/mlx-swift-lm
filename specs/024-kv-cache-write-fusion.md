# 024 — Eliminate per-decode-token KV cache write copies

- **Status:** spec, ready to issue (bounded effort, single primitive change)
- **Branch:** new branch off alpha; depends on the chosen approach (see below)
- **Depends on:** for approach (1), an upstream `mlx` change shipped through `mlx-c` and `mlx-swift`. Approaches (2) and (3) are in-repo only.
- **Supersedes:** investigation on `ek/gemma4-e2b-kv-copy-fusion` (deferred 2026-04-17 — `MLXFast.metalKernel` in-repo fix not viable; root cause confirmed)

## The insight

Every decode token on Gemma4-E2B issues **60 `copy_bfloat16` Metal dispatches** entirely from KV-cache writes — pure overhead, ~6.6% of the 905 total dispatches/token. The arithmetic generalises: every `KVCacheSimple` / `RotatingKVCache` user pays this cost in proportion to the number of non-shared layers that write per step. Gemma4-E2B is a clean diagnostic case (15 non-shared layers × 2 slice-updates × 2 copy kernels = 60), but Qwen 3.5, GPT-OSS, Gemma 4 26B-A4B all eat the same overhead per non-shared write.

The cost has two distinct sources, both inside `SliceUpdate::eval_gpu` ([`mlx/backend/metal/indexing.cpp:725`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/indexing.cpp)):

1. **`vn_copybfloat16bfloat16` (×30 on E2B)** — bulk copy of the **entire cache buffer**. Dispatched because MLX's `is_donatable()` check fails on the cache input, so MLX cannot reuse the cache allocation as the output of the slice-update.
2. **`gg1_copybfloat16bfloat16` (×30 on E2B)** — the actual **strided write** from the transposed `[B, H, T, D]` K/V view (`keys.transposed(0, 2, 1, 3)`) into the contiguous `[B, H, T_max, D]` cache slot. Required because K/V are transposed before write, so the source is strided.

Eliminating either copy class is a measurable win on every decoder; eliminating both is the architectural fix.

## What we already ruled out

The investigation on `ek/gemma4-e2b-kv-copy-fusion` confirmed two dead-ends. Documenting here so we don't re-tread them:

- **Donation-unblock at the Swift layer doesn't work.** Hypothesis was that `lastReturnedKeys`/`lastReturnedValues` views keep an extra ref on the cache's `array_desc_`, defeating `is_donatable()`. Cleared them at the top of `update()` in both `KVCacheSimple` and `RotatingKVCache`. Dispatch count unchanged. Either the Swift subscript-set path or an MLX graph edge holds the extra ref at the moment `is_donatable()` runs. Root-causing requires C++/mlx-c instrumentation.

- **`MLXFast.metalKernel` cannot fuse the write.** `metalKernel` allocates fresh output arrays and cannot write into a pre-allocated cache buffer. Any "fused write" via `metalKernel` would have to output a full cache-sized buffer and copy unchanged parts every step — strictly worse than status quo. The pattern works for `MLXFast.rmsNormRoPE` because that's a pure functional kernel (output is a new array of similar size); KV-cache write is an in-place slot update on a persistent allocation. Different primitive.

## Three viable approaches

Three independent paths get us there. Pick one based on appetite for upstream work versus in-repo scope.

### Approach 1 — MLX-level `SliceUpdate` extension or `MLXFast.writeInto(...)`

Either:

- **Extend `SliceUpdate`** in `mlx/backend/metal/indexing.cpp` to handle a strided source without inserting `copy_gpu_inplace`. This eliminates the `gg1_copy` (strided-source) class. Concurrently, fix donation so `vn_copy` (bulk-copy) elides on the in-place path. ~30 lines in `eval_gpu`, plus a regression test.
- **Or** add a new primitive `MLXFast.writeInto(buffer:slice:source:)` that takes a pre-allocated buffer + slice spec + source, performs the strided write directly, and returns the same buffer (preserves the persistent allocation contract). Wires through `mlx-c`'s `MLX_API` surface and `mlx-swift`'s `MLXFast.swift`.

- **Pros:** clean fix at the source, no Swift-side refactor, every consumer benefits automatically (Gemma 4, Qwen 3.5, GPT-OSS, future models).

- **Cons:** requires shipping bumps across `mlx` + `mlx-c` + `mlx-swift`. Three-repo PR coordination. Upstream review timeline.

- **Approximate win:** eliminates **all 60 copies** on Gemma4-E2B (both classes). Projected decode-tok/s lift: **+5–8% on Gemma4-E2B at 1024 ctx**, scaling proportionally with non-shared layer count on other models.

### Approach 2 — Cache layout refactor `[B, T, H, D]`

Store the cache as `[B, T_max, H, D]` instead of `[B, H, T_max, D]`. The `keys.transposed(0, 2, 1, 3)` step before the write disappears — source is contiguous, slice destination is contiguous, MLX never needs to insert `gg1_copy`.

The transpose moves to the SDPA read path. `MLXFast.scaledDotProductAttention` needs to handle the strided `[B, H, T, D]` view of the `[B, T, H, D]` cache without materialising it; if it doesn't, we reintroduce a copy on the read side. **This is the load-bearing risk** — verify before committing.

- **Pros:** in-repo only, no upstream changes. Clean architectural fix.

- **Cons:** touches `KVCacheSimple` + `RotatingKVCache` + every model's attention call site (every `Models/*.swift` that constructs a cache or reads from one). Wide regression surface. The SDPA-read-side question may force us to pre-compute a transpose anyway, which would partially un-do the gain.

- **Approximate win:** eliminates the `gg1_copy` class (×30 on E2B) cleanly. The `vn_copy` class only goes away if we *also* fix donation independently — a `[B, T, H, D]` cache is still ref-counted from `lastReturnedKeys`/`lastReturnedValues`. So this path on its own ~halves the overhead, not eliminates it.

### Approach 3 — Fuse cache-write + SDPA into one custom kernel

Write a single Metal kernel that takes the freshly-computed `[B, L, H, D]` contiguous K/V plus the existing cache buffer, performs the cache write **and** the attention reduction in one dispatch. Both `vn_copy` and `gg1_copy` go away because there's no separate slice-update — the cache buffer is read and updated in the same kernel that does attention.

- **Pros:** biggest measurable win. Eliminates copies AND saves the SDPA dispatch boundary (one fewer dispatch per layer-token). Composes with attention sinks, sliding window, paged cache.

- **Cons:** biggest project. Real Metal kernel work. Need variants for: standard attention, sliding window, attention sinks, GQA (the four-way matrix that Gemma 4 / Qwen 3.5 / GPT-OSS need). High maintenance burden — new attention features need to be ported into the fused kernel.

- **Approximate win:** eliminates 60 copies + 30 SDPA dispatches per decode token on E2B = ~10% dispatch reduction, projected **+8–12% decode tok/s** on top of approach 1's win. Larger on models with more non-shared layers.

## Recommendation

**Pursue approach 1 first.** Smallest scope, cleanest separation, every consumer benefits automatically. The three-repo coordination is real but tractable — the C++ change is ~30 lines, the C binding is ~10, the Swift wrapper is ~10. Land this as a Phase 1 win, **then** evaluate whether the residual SDPA-dispatch overhead (approach 3 territory) is worth a kernel-fusion project.

Approach 2 is the worst of the three on a cost/risk basis — wide blast radius, partial fix.

## Implementation phases (approach 1)

1. **Phase 1 — Diagnostic baseline against current alpha.** Re-run the dispatch audit on Gemma4-E2B + Qwen 3.5-9B + GPT-OSS-20B at 1024 ctx to confirm the 60-copy pattern still holds and capture per-model baselines. Approach: re-run the bench cell with `MLX_METAL_PROFILE=1` and grep for `copy_bfloat16` in the trace; or land issue #TBD (`--dispatch-audit` flag) first if we want CI-level numbers.

2. **Phase 2 — Upstream MLX patch.** Extend `SliceUpdate::eval_gpu` to (a) elide the bulk `vn_copy` when the donation refcount is unblockable from the C++ side (e.g. the cache buffer's only refs are the slice-update's input + output), and (b) accept a strided source without inserting `copy_gpu_inplace`. PR to `ml-explore/mlx`.

3. **Phase 3 — `mlx-c` binding.** If approach 1 takes the `MLXFast.writeInto(...)` route instead of the in-place fix, expose the new primitive via `mlx_c` headers + impl. Skipped if we go pure `SliceUpdate` extension.

4. **Phase 4 — `mlx-swift` wrapper + version bump.** Bump `Package.swift` mlx-swift dependency. Add Swift wrapper if a new `MLXFast` API was introduced.

5. **Phase 5 — Verify in `mlx-swift-lm`.** Re-run the dispatch audit. Confirm the 60 copies → 0 (or ~halved if only one class was fixed). Run the decode bench across the model zoo. No code changes expected in `Libraries/MLXLLM/Models/*.swift` — the fix is upstream-transparent.

## Expected impact

- **Gemma4-E2B:** decode +5–8% at 1024 ctx (60 copies eliminated, ~6.6% dispatch reduction).
- **Gemma4 26B-A4B (MoE):** smaller relative win — copies are diluted in the larger per-token compute. Still ~3–5%.
- **Qwen 3.5-9B:** depends on layer count; expected +4–6% at 1024 ctx.
- **GPT-OSS-20B:** depends on layer count; expected +3–5% at 1024 ctx.

All numbers projected from the 6.6% dispatch-overhead measurement on E2B; verify on real hardware in Phase 5.

## Risks

1. **Upstream review timeline.** The MLX change is small but it touches a hot path; review may surface concerns we haven't anticipated (numerical equivalence, edge cases around strided destination + strided source).

2. **Donation root cause may be in `mlx-swift`'s subscript-set path, not MLX.** If the extra ref on `array_desc_` traces back to how `MLXArray.subscript[set:]` builds the slice-update graph, the fix lives in `mlx-swift`, not `mlx`. Phase 2 needs a brief instrumentation pass to confirm where the ref leaks.

3. **Strided-source acceptance breaks other `SliceUpdate` callers.** `SliceUpdate` is used beyond KV cache (any `array[slice] = value` with a non-contiguous source goes through it). Need to verify the change doesn't regress anything else in the MLX test suite.

## Files touched

| File | What |
|---|---|
| (`mlx`) `mlx/backend/metal/indexing.cpp` | `SliceUpdate::eval_gpu` — handle strided source without `copy_gpu_inplace`; fix donation. |
| (`mlx`) `mlx/backend/metal/copy.cpp` | If `is_donatable()` logic needs adjustment for the cache use-case. |
| (`mlx-c`) `mlx/c/fast.h` + `.cpp` | Only if `MLXFast.writeInto(...)` route is taken. |
| (`mlx-swift`) `Source/MLX/MLXFast.swift` | Wrapper for new `writeInto(...)` primitive, if added. |
| (`mlx-swift-lm`) `Package.swift` | Version bump on `mlx-swift` dependency. |
| (`mlx-swift-lm`) `benchmarks/notes/2026-MM-DD-spec-024-results.md` (new) | Per-model dispatch + decode-tok/s before/after. |

## Why this is Tier 4

Real win, but per-model rather than universal — speculative-decoding work in Tiers 1–3 unlocks 2–4× speedups on entire model families, while spec 024 is a flat 3–8% per model. Land after Tier 3 stabilises. The `ek/gemma4-e2b-kv-copy-fusion` investigation already preserved the architecture analysis; this spec carries the work forward to the right primitive layer.
