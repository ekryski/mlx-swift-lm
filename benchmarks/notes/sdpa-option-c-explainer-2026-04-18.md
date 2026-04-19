# SDPA Option C — How this works vs. the old way

Written for: future-Eric or a collaborator picking this up cold. Companion to the full plan [sdpa-option-c-plan-2026-04-18.md](sdpa-option-c-plan-2026-04-18.md) and the adoption plan [argument-buffers-adoption-plan-2026-04-17.md](argument-buffers-adoption-plan-2026-04-17.md).

---

## 1. The problem, in one page

Metal Indirect Command Buffers (ICBs) let us record a compute pass once and replay it cheaply for every decode step. On GPT-OSS-20B we already measure **1.45× CPU encoding speedup** from ICB replay alone — but the replay is silently wrong past the recorded T_k. The diagnostic at mlx commit [`6f097aa6`](https://github.com/ekryski/mlx/commit/6f097aa6) caught it:

- Record SDPA with `T_k = 1024`. Command count = 5, segments = 3, arena = 148 bytes.
- Record SDPA with `T_k = 1025`. Command count = 5, **segments = 4**, arena = 148 bytes but **byte-for-byte different**.

Two things went wrong:

1. **Shape scalars baked into the encoder.** When `fast::scaled_dot_product_attention` runs, the CPU writes `T_k`, mask bounds, strides, and the softmax scale via `MTLComputeCommandEncoder::setBytes(…)`. Those bytes spill into the ICB recorder's arena and freeze at `finalize()`. A replay at a different T_k will compute on the recorded T_k, not the current one.

2. **Topology flipped on a threshold.** SDPA has two vector kernels: `sdpa_vector` (single-pass) for small KV, `sdpa_vector_2pass_1 + 2` (two-pass fused) for large KV. The CPU switches at `T_k = 1024` (and again on GQA at 4096). Different kernel → different segment count. Even a perfect "patch the setBytes at replay time" fix can't rescue this, because the *recorded shape of the command graph* is wrong.

Together, these mean: record the ICB at T_k=1024 on step 1 of a decode, replay for steps 2..N at growing T_k, and attention silently ignores every K/V position past 1024. Measured impact (E3 in the adoption plan): for a 1024-ctx, 400-token decode, **~28% of valid K/V positions go un-attended by the final step**.

Not a perf bug. A correctness bug that masquerades as perf. ICB replay is unusable without a fix.

---

## 2. Old way — current SDPA dispatch

### CPU side (today)

Every decode step, at every layer that calls SDPA:

```
ScaledDotProductAttention::eval_gpu(...)                    // scaled_dot_product_attention.cpp:665
├── (possibly) contiguous_copy_gpu for Q/K/V layout fixes
├── branch on q.shape(2) <= 8   // vector / prefill split
│     └── branch on KV-length / GQA / device class
│           ├── sdpa_vector_2pass(...)   // 2 dispatches + barrier
│           │   ├── setComputePipelineState
│           │   ├── setBuffer(Q, 0)  setBuffer(K, 1)  setBuffer(V, 2)  setBuffer(out, 3)
│           │   ├── setBuffer(sums_scratch, 4)  setBuffer(maxs_scratch, 5)
│           │   ├── setBytes(N=T_k, 7)                  ← SHAPE-DEPENDENT
│           │   ├── setBytes(k_head_stride, 8)          ← SHAPE-DEPENDENT
│           │   ├── setBytes(k_seq_stride, 9)           ← SHAPE-DEPENDENT
│           │   ├── setBytes(v_head_stride, 10)         ← SHAPE-DEPENDENT
│           │   ├── setBytes(v_seq_stride, 11)          ← SHAPE-DEPENDENT
│           │   ├── setBytes(scale, 12)
│           │   ├── (has_mask ? setBytes(mask strides 15/16/17))
│           │   ├── (has_sinks ? setBuffer(sinks, 18))
│           │   ├── dispatchThreadgroups(pass1)
│           │   │
│           │   ├── [BARRIER]
│           │   │
│           │   ├── setComputePipelineState(pass2)
│           │   ├── setBuffer(partials, 0) setBuffer(sums, 1) setBuffer(maxs, 2) setBuffer(out, 3)
│           │   ├── setBytes(blocks, 4)
│           │   └── dispatchThreadgroups(pass2)
│           │
│           └── sdpa_vector(...)   // single dispatch
│               (same setBuffer + setBytes pattern, minus partials)
```

Per-step CPU cost (measured, GPT-OSS-20B decode at 1024 ctx): **~17.1 ms/step** across ~1523 dispatches. Attribution (rough):

| Slice | % of per-step CPU encoding |
|---|---:|
| setBytes + setBuffer + setComputePipelineState | **40–55%** |
| Barrier tracking + useResource bookkeeping | **10–15%** |
| Command-buffer commit + ObjC bridge | **5–10%** |
| GPU execution time | **25–35%** |

SDPA-family dispatches (`sdpa_vector_*` + `rope_single_freqs_*`) are only ~4.7% of the step by count but carry disproportionate per-call setBytes weight: ~8 scalar fields per SDPA call × up-to-2 passes.

### Why recording doesn't rescue correctness

The ICB recorder captures whatever bytes the encoder emits. When the CPU calls `setBytes(T_k=1024, …)`, 4 bytes land in the recorder arena. Replay at step 2 with T_k=1025 re-uses those frozen 4 bytes. GPU reads `N=1024`, loops K[0..1023], silently skips K[1024]. No error. Just wrong output.

---

## 3. New way — unified kernel + Argument Buffer

Two changes stacked:

### 3a. One kernel (Option C Phase 1)

`sdpa_unified_vector` is a single kernel that handles the full `(L_q ≤ 8, L_k ∈ [1, 128K])` decode range. It merges the single-pass and 2-pass bodies by taking `blocks` (the K-axis split count) as a **runtime** argument:

```
sdpa_unified_vector<T, D, V>(args...)
├── for (k_block = 0; k_block < blocks; ++k_block) {
│     online_softmax_accumulate(...)   // body of old _2pass_1
│   }
├── threadgroup_barrier
└── merge_partials_and_write(...)       // body of old _2pass_2 but in-kernel
```

`blocks == 1` short-circuits the merge to a direct write — that path is the old single-pass kernel's hot shape.

Why this matters for ICB: **the segment topology no longer flips on T_k**. There's one kernel, one dispatch, one segment boundary pattern, for every T_k in decode.

### 3b. One bind (Option C Phase 2, AB migration)

Instead of N `setBytes` + M `setBuffer` calls, the CPU packs everything into a small shared-storage `MTL::Buffer` — the Argument Buffer:

```
struct SdpaUnifiedArgs {                        // ~128 bytes
  BufferPtrOffset queries, keys, values, out;   // 16B each, addr + offset
  BufferPtrOffset mask, sinks;                  // zero when not present
  int gqa_factor, N, blocks, _pad;
  size_t k_head_stride, k_seq_stride, v_head_stride, v_seq_stride;
  float scale;
  int mask_kv_seq_stride, mask_q_seq_stride, mask_head_stride;
  int num_q_heads;
};
```

The kernel signature becomes one line:

```metal
[[kernel]] void sdpa_unified_vector_ab(
    constant const SdpaUnifiedArgs& args [[buffer(0)]], ...);
```

CPU-side per call:

```cpp
auto ab = std::make_shared<ArgumentBuffer>(d, layout);
ab->set_buffer_ptr(0, q.buffer(), q.offset());
ab->set_buffer_ptr(1, k.buffer(), k.offset());
// ... populate the 18 slots ...
compute_encoder.set_buffer(ab->mtl_buffer(), 0);   // ONE bind
compute_encoder.dispatch_threadgroups(...);
```

The shape scalars live in device memory now. At replay time the GPU reads them fresh — whatever the CPU last wrote into the AB before kicking the replay is what the kernel sees. No stale bytes. No setBytes lanes in the recorder arena at all.

### What the ICB recorder sees, new way

```
ICB arena after recording one SDPA call:
  setComputePipelineState(sdpa_unified_vector_ab)
  setBuffer(ab->mtl_buffer(), 0)         ← stable pointer across replays
  dispatchThreadgroups(grid)              ← grid size depends on D, H, not on T_k at Phase-1-blocks count
```

Three stable items. The arena is the same bytes at T_k=1024 and T_k=1025. The diagnostic that motivated this whole plan now reports "ARENAS IDENTICAL."

---

## 4. How the opt-in layering works

One runtime helper — `ab_enabled()` — drives every AB-migrated primitive's per-call branch. It reads the env composition:

```cpp
bool ab_enabled() {
  const char* ab  = std::getenv("MLX_METAL_AB");
  const char* icb = std::getenv("MLX_METAL_ICB");
  return (ab  && ab[0]  == '1') ||
         (icb && icb[0] == '1');  // ICB implies AB
}
```

Truth table (for users, not internal debug overrides):

| Env | Result |
|---|---|
| (nothing) | **Alpha / legacy.** Byte-identical to today. No new kernels, no AB, no ICB. |
| `MLX_METAL_AB=1` | 9-primitive AB stack active. Unified SDPA used. No ICB. Measurable decode win. |
| `MLX_METAL_ICB=1` | **Implies AB.** Full ICB+AB path. Unified SDPA + AB bind + ICB record/replay. |
| `MLX_METAL_ICB=1 MLX_METAL_AB=0` | Same as above. `AB=0` cannot disable AB when ICB is on — the broken "ICB without AB" config is unreachable by design. |

Internal debug only (not documented for users): `MLX_SDPA_FORCE_LEGACY=1` forces legacy SDPA even when AB is on — used by the Phase 0/1 regression harness for within-binary A/B comparison.

---

## 5. Encoding-time accounting

Measured baseline and projected trajectory (GPT-OSS-20B decode, 1024 ctx, M1 Max 64GB):

| Stage | Per-step CPU encoding | Gate | Notes |
|---|---:|---|---|
| Live dispatch | 17,107 µs | `(none)` | 100% baseline. 1523 dispatches/step. |
| ICB replay, 8 ABs, SDPA stale | 11,831 µs (1.45×) | `MLX_METAL_ICB=1` before this branch | **Correctness broken past recorded T_k.** |
| + SDPA unified + AB (this branch) | ~9.0–10.5k µs (est.) | `MLX_METAL_ICB=1` on `ek/sdpa-option-c` | Correctness unlocked; ~1.7–1.9×. |
| + weight heap + decode-loop ICB wiring | ~3.5–5.5k µs (est.) | follow-on | 3–5× ceiling per adoption-plan attribution. |
| Decode tok/s target | ≥85 tok/s | | vs 64 baseline, +33% |

SDPA is ~4.7% of per-step dispatches by count — its direct encoding contribution is small. The value in **this branch** is the correctness unlock; the bigger remaining wins come from wiring the ICB into the decode loop on the mlx-swift-lm side and landing the weight heap (see §6).

---

## 6. What this branch does NOT do

- **Prefill / full-attention path** (`q.shape(2) > 8`). Uses different kernels (`steel_attention_*` / `steel_attention_nax_*`). Not touched. ICB value is low there — one dispatch per prompt amortized over hundreds of decode steps.
- **SDPA VJP** (`ScaledDotProductAttentionVJP::eval_gpu`). Training codepath. Untouched.
- **Legacy kernel retirement.** `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2` stay shipped as-is. Alpha branch of mlx-swift-lm depends on them. Option C Phase 3 (retirement) is deferred indefinitely.
- **Per-layer ICB wiring in mlx-swift-lm's decode loop.** The recorder + replay primitives exist (`IndirectCommandRecorder`, `tag_binding`, `replay_with_overrides`), but no model's decode loop uses them yet. Plan-doc target: GPT-OSS-20B first.
- **Weight heap.** `MTL::Heap` + `useHeap:` to collapse per-segment `useResource` iteration. Additive infra work, not gating correctness.
- **Swift-side AB content-update path.** The AB pointer is stable; per-step `abCurrentStep.setScalar32(slot: .TK, …)` needs a Swift API surface if one isn't there.
- **Multi-stream AB ownership.** Each stream should write into its own AB. Test coverage needed.

All six are tracked in the plan file's "After this branch" section and will be picked up in follow-on branches.

---

## 7. Quick reference

| Thing | Where |
|---|---|
| Legacy vector SDPA kernel | [`mlx/backend/metal/kernels/sdpa_vector.h`](../../../mlx/mlx/backend/metal/kernels/sdpa_vector.h) |
| Legacy CPU dispatch | [`mlx/backend/metal/scaled_dot_product_attention.cpp:665`](../../../mlx/mlx/backend/metal/scaled_dot_product_attention.cpp#L665) |
| Shape-sensitivity diagnostic | mlx [`6f097aa6`](https://github.com/ekryski/mlx/commit/6f097aa6) (in `tests/icb_real_primitive_tests.cpp:208`) |
| ArgumentBuffer class | [`mlx/backend/metal/argument_buffer.h`](../../../mlx/mlx/backend/metal/argument_buffer.h) |
| AB pilot (RMSNorm pattern) | [`mlx/backend/metal/normalization.cpp:20-147`](../../../mlx/mlx/backend/metal/normalization.cpp#L20) |
| Phase 0 regression harness | `mlx/tests/sdpa_regression_tests.cpp` on `ek/sdpa-option-c` |
| Phase 0 baseline artifact | [`benchmarks/notes/sdpa-option-c-baseline-2026-04-18.md`](sdpa-option-c-baseline-2026-04-18.md) |
| Full plan | [`benchmarks/notes/sdpa-option-c-plan-2026-04-18.md`](sdpa-option-c-plan-2026-04-18.md) |
| Adoption-plan context | [`benchmarks/notes/argument-buffers-adoption-plan-2026-04-17.md`](argument-buffers-adoption-plan-2026-04-17.md) |
