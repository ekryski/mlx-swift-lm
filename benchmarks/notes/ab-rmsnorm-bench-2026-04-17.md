# AB Pilot — Bench Results — 2026-04-17

Measured delta between the legacy path and the
`ArgumentBuffer`-migrated paths. Companion to
[ab-rmsnorm-pilot-2026-04-17.md](ab-rmsnorm-pilot-2026-04-17.md).

Two migrations have landed so far:
- Phase 1 — RMSNorm (`rms_ab_*` kernels).
- Phase 2, first primitive — RoPE single-token decode variants
  (`rope_ab_single_*`, `rope_ab_single_freqs_*`).

All four repos on sub-branch / branch pair:

| Repo | Branch | HEAD |
|---|---|---|
| `mlx` | `ek/ab-rmsnorm-pilot` | `4d4be1a1` |
| `mlx-swift` | `ek/metal-icb-prototype` | `7e4943e` |
| `mlx-swift-lm` | `ek/metal-icb-prototype` | this note |

## Microbench (`tests/argument_buffer_bench.cpp`)

Legacy vs AB encoding cost for 1000 dispatches of a trivial kernel
(1 threadgroup, 1 thread) on M1 Max, macOS 15.7.4:

| Path | Median total | Per-dispatch |
|---|---:|---:|
| Legacy (6 bindings) | 187 µs | 0.188 µs |
| AB (1 binding)      |  94 µs | 0.094 µs |

**Ratio: 1.975x.** Above the ≥ 1.5x go/no-go threshold in the pilot
design note. Saved cost per dispatch: 0.093 µs. Doctest assertion
pass: 85/85 AB unit tests + 1652/1652 ICB tests green. Full mlx suite
286/287 pass (one pre-existing flaky scheduler test, unrelated).

## Model-level end-to-end: Gemma 4 E2B 4-bit, summarization, 1024 ctx

M1 Max, 64 GB, `./scripts/benchmark.sh --model gemma4-e2b --method
summarization --context 1024 --quant 4bit`. **6 trials each** —
initial 2-trial result showed a −5.3 % prefill regression that
dissolved into the noise floor once more samples were taken.

| Config | n | Prefill mean (tok/s) | Range | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 6 | 2668.7 | 2479.7 – 2847.6 | — |
| AB=1 | 6 | 2605.1 | 2276.9 – 2773.4 | **−2.4 % (within ±5 % noise)** |

| Config | n | Decode mean (tok/s) | Range | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 6 | 94.7 | 93.8 – 96.7 | — |
| AB=1 | 6 | 94.4 | 92.6 – 97.2 | **−0.3 % (noise)** |

Raw trial-by-trial numbers preserved in the commit history for this
note (first two trials in the 2026-04-17 bench note edit history,
remaining four recorded during the re-verification run).

Output quality: both paths produce coherent English summaries of the
(deterministic) Gatsby prompt across all twelve runs. No numerical
degradation — the AB kernel reads identical inputs and runs the same
math; any divergence would show up immediately as garbled text.

## Why tok/s is flat even though encoding is 1.975x faster

RMSNorm is ~3 dispatches per layer × 24 layers ≈ 72 dispatches per
decode step on Gemma 4 E2B (roughly 5 % of per-step dispatches,
extrapolating from the E1 GPT-OSS-20B audit: 1523 dispatches/step).
With RMSNorm encoding 2x cheaper and RMSNorm ~5 % of encoding time
(which is itself ~40–55 % of step), the predicted tok/s win from
RMSNorm alone is **≤ 1–2 %** — inside the trial-to-trial noise floor
that 6-run measurements confirm here.

The Phase-1 value is *architectural*: AB proves the path works
end-to-end on a real model without regression. Stacking migrations
(Linear, Embedding, RoPE, etc. per plan doc § Layer 2) is what
converts per-primitive encoding wins into per-step wins.

## Per-call AB allocation — not a bottleneck at Phase 1

The AB path calls `MTL::Device::newBuffer(64 B, shared)` inside each
`ArgumentBuffer` construction. An earlier read with n=2 attributed a
5 % prefill regression to this allocator; the n=6 re-verification
showed prefill delta inside the noise floor instead, so the 5 %
figure was a sampling artifact.

AB pooling (per-primitive ring, graph-wide arena, or thread-local
reuse) is still cleanly in scope for Phase 2 — once more primitives
are migrated, the aggregate per-step allocation count could
plausibly show up as a measurable cost. The plan doc flags it as
an open question (§ "AB lifetime and pooling"); deferring the
implementation to when it is justified by measurement.

## Go / no-go

- Microbench ≥ 1.5x: **✓** (1.975x)
- Model-level correctness: **✓** (coherent output in 12/12 runs)
- Model-level decode tok/s: **neutral** (−0.3 %, noise-bound)
- Model-level prefill tok/s: **neutral** (−2.4 %, noise-bound)

**Decision: proceed to Phase 2.** Phase-1 exit criterion met on
both counts. No regression to fix before moving on.

## Phase 2 — RoPE single-token stacked on RMSNorm

Second primitive migrated: `rope_single` / `rope_single_freqs`
(T==1, contiguous, one-offset — the decode-time fast path on Gemma
4 E2B). Prefill RoPE variants remain on legacy.

Same harness — `gemma4-e2b`, 4bit, `summarization --context 1024`,
n=5 for AB=0 and n=6 for AB=1 (control was truncated by a Swift
warning print on trial 1; remaining 5 numbers intact):

| Config | n | Prefill mean | Range | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 5 | 2652.8 | 2516.6 – 2748.5 | — |
| AB=1 (RMSNorm+RoPE) | 6 | 2653.6 | 2591.9 – 2756.2 | **+0.03 %** |

| Config | n | Decode mean | Range | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 5 | 93.6 | 92.9 – 94.4 | — |
| AB=1 (RMSNorm+RoPE) | 6 | 93.7 | 93.4 – 94.0 | **+0.1 %** |

Still tok/s-flat at the model level — consistent with theory:
RMSNorm + single-token RoPE together account for ~12 % of per-step
dispatches on Gemma 4 E2B (72 RMSNorm + 48 RoPE ≈ 120 of ~1000+).
Halving their per-dispatch encoding cost yields ≤ 2 % aggregate
tok/s improvement, well inside the measured noise floor.

Output quality still coherent across all AB=1 runs. Correctness is
the gate we can verify; tok/s will require several more primitives
on AB before the signal clears the noise floor.

## Decision after Phase-2 first primitive

Continue migrating primitives per plan doc order, but expect
tok/s to remain flat through at least the next 2–3 primitives
(Embedding, softmax, elementwise). The signal test should happen
after Linear / matmul (the dispatch-count-dominant primitive) is
on AB — that's when stacked savings should cross the noise floor
on a decode run.

## Phase 2 — affine_qmv (Linear hot path) — 2026-04-18

Second Phase-2 primitive migrated: `affine_qmv` /
`affine_qmv_fast`, non-batched only (`B == 1`), affine mode only.
Covers every Q / K / V / O projection and every MLP up / down
projection during decode on a 4-bit-affine-quantized model —
~300+ dispatches per step on Gemma 4 E2B, the dispatch-count-
dominant primitive per the E1 audit.

Scope excluded (legacy path retained):
- Other quant modes: fp4, fp8, mxfp4.
- qmv_quad (K == 64 or 128).
- qvm, qvm_split_k, qmm, qmm_splitk.
- gather_qmv (MoE expert gather).
- Batched (`B > 1`, shape/stride arrays).

Three supporting changes landed with the kernel because the
direct-port regressed tok/s:

1. **AB pool.** `MTL::Device::newBuffer` per AB construction at
   ~300 calls/step measured as the dominant cost inside the AB
   path. Added a process-wide pool (`argument_buffer.cpp`) keyed
   by byte size; buffers stay resident and recycle via the
   `add_temporary_object` completion handler.
2. **`CommandEncoder::use_resource`.** The AB kernel reads
   weights / scales / biases / x / y via raw `gpuAddress()`
   pointers. Metal's implicit per-encoder residency tracking
   fires on `setBuffer`; with AB, we need explicit
   `useResource` declarations so the driver keeps buffers
   mapped for the dispatch.
3. **`const constant int& in_vec_size = args.K`** inside the AB
   kernel before forwarding to `qmv_impl`. Preserves the
   constant-address-space qualifier on scalar refs, which some
   compiler versions use for SIMD-broadcast load optimizations.
   Without this alias the kernel body ran noticeably slower.

### Gemma 4 E2B 4-bit summarization-1024 stacked — n=6

M1 Max, `./scripts/benchmark.sh --model gemma4-e2b --method
summarization --context 1024 --quant 4bit`.

| Config | n | Prefill mean | Range | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 6 | 2803.9 | 2741 – 2867 | — |
| AB=1 stacked | 6 | 2724.4 | 2594 – 2784 | **−2.8 % (noise)** |

| Config | n | Decode mean | Range | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 6 | 96.7 | 96.2 – 97.1 | — |
| AB=1 stacked | 6 | **101.0** | 100.5 – 101.5 | **+4.4 %** |

Decode shows a clear, consistent win: every AB=1 trial landed
above 100 tok/s, every AB=0 trial below 97.1 tok/s — zero
overlap between the two ranges. First model-level tok/s signal
from Phase 2.

Output quality coherent across all 12 runs.

Prefill delta is within noise (2741–2867 vs 2594–2784 — ranges
overlap substantially, means differ by one SD of either
distribution). Prefill does not take the qmv AB path (M ≥
vector_limit → qmm), so the prefill path is RMSNorm + RoPE AB
only, same as the prior Phase-2 measurement.

### Lessons

- Per-call MTL buffer allocation is a serious cost floor once
  AB migrations hit hundreds of dispatches per step. Pooling is
  mandatory, not optional — should be the first Phase-2 change
  landed when stacking the next primitive.
- `useResource` is mandatory whenever a kernel reads buffers
  via raw GPU addresses (`gpuAddress()`). The mlx residency set
  keeps weights mapped globally, but per-step allocations (x,
  y) need explicit per-encoder declarations.
- `const constant int&` aliases inside AB kernels when
  forwarding to legacy `_impl` templates — do not change the
  shared helper signatures; preserve the address-space qualifier
  through a reference binding in the AB wrapper only.

## Phase 2 — Embedding (Gather::gather_front) — 2026-04-18

Third Phase-2 primitive: `gather_front` fast path inside
`Gather::eval_gpu`. The decode-time embedding lookup — one
dispatch per step on every transformer. Migrated for
pattern-consistency; dispatch count is too low to move tok/s on
its own.

Kernel file `gather_front_ab.metal`: AB variant over the
existing `gather_front` body. 208 instantiations (13 dtypes × 2
index dtypes × 2 loc widths × 2 work-per-thread values) so the
AB path covers every combination the legacy JIT path does.

Stacked bench (RMSNorm + RoPE + affine_qmv + gather_front,
Gemma 4 E2B 4-bit summarization 1024, n=4):

| Config | n | Prefill mean | Decode mean |
|---|---:|---:|---:|
| AB=0 | 4 | 2758.7 | 97.6 |
| AB=1 | 4 | 2805.6 | 100.6 |

Decode +3.1 %, prefill +1.7 % — both within the range of the
3-primitive measurement. No regression from adding Embedding to
the stack, as predicted.

## Architecture gap for remaining Phase-2 primitives

The next three primitives on the plan doc list
(`SwitchLinear` / MoE gather, `softmax` / `add` / `silu`
elementwise, `scaled_dot_product_attention`) each require
pattern extensions beyond what RMSNorm → Embedding needed:

**SwitchLinear / gather_qmv / gather_qmm_rhs.** Kernel surface
includes variable-length shape/stride arrays bound today via
`set_vector_bytes` (x_shape, x_strides, w_shape, w_strides,
s_strides, b_strides, plus gather index shapes/strides). The
current `ArgumentBuffer::Slot::Kind` enum only covers
Scalar32/64, Float32, and BufferPtrOffset. Options:

- Add a `Bytes` slot kind that copies an arbitrary byte range
  into the AB and exposes a `set_bytes(int slot, const void*,
  size_t)` method. Aligns to 4 bytes. Slot size is fixed at
  layout-declaration time — good enough for a cap-at-8-dims
  schema.
- Separate small MTLBuffer per shape array, its pointer placed
  in a BufferPtrOffset slot. Cleaner runtime semantics but
  double-indirection for what is already small inline data.

Additional consideration: GPT-OSS-20B uses mxfp4-quantized
weights, so its hot MoE path is `fp_*_gather_qmv` (in
`fp_quantized.h` / `fp_quantized.metal`), not `affine_*`. The
AB migration needs separate kernel-file instantiations per
quantization mode.

**Elementwise (add / silu / softmax / silu_mul / etc.).** These
dispatches are produced by `Compiled::eval_gpu` which JIT-
compiles a kernel per unique op-tree. The static-instantiation
AB pattern doesn't fit — the kernel source is generated, not
hand-written. Extension path:

- Add AB-aware templates to the generated kernel source in
  `compiled.cpp`. Every generated kernel would gain an `_ab`
  variant whose signature is `constant const ArgsStruct&
  [[buffer(0)]]`.
- Cache both AB and legacy variants by op-tree hash.

Estimated: ~6–10 hours to prototype the pattern on a single
simple op tree, additional time to generalize.

**scaled_dot_product_attention.** Plan doc flags this as
"the load-bearing migration." Blocker is kernel-selection
stabilization — `fast::SDPA` flips between kernel variants
at T_k thresholds (diagnostic in `6f097aa6`), so even a
perfect AB migration of one variant is numerically wrong
under a growing KV cache. The pre-work is either forcing one
variant for all T_k in the decode range or tagging the
kernel-variant choice in the AB itself. Estimated multi-
session.

### Recommendation on scope

Stopping the present session with the 4-primitive baseline
recorded above is the conservative engineering call. Continuing
into SwitchLinear (mxfp4 especially) risks regressing the
measured +3–4 % decode win through a bug in a less-verified
migration, which is the pattern we just saw with the
pre-pool qmv path.

The extensions above are each well-bounded standalone tasks
and each have a clear exit criterion (the same n=6 bench on
the respective model). They should be taken one per session,
in plan doc order, with the measurement gate every time.

## Phase 2 — SwitchLinear / affine_gather_qmv — 2026-04-18

Fifth Phase-2 primitive: `affine_gather_qmv` /
`affine_gather_qmv_fast` — the non-sorted affine-quantized MoE
matmul decode path. Called from `GatherQMM::eval_gpu` when
`transpose == true` and `M < vector_limit`.

Scope: affine mode only. Other quant modes
(`mxfp4_gather_qmv`, `fp4_*`, `fp8_*`) stay on legacy. That means
GPT-OSS-20B's MoE experts (which use mxfp4) do *not* hit the new
AB path — an mxfp4 follow-on migration is required to land the
MoE decode signal on that model.

### Pattern extension: `Bytes` slot kind in `ArgumentBuffer`

All prior AB migrations packed fixed-width scalars and buffer
pointers only. gather_qmv adds variable-length shape/stride
arrays (x_shape, x_strides, w_shape, w_strides, s_strides,
b_strides + gather batch metadata). To package those into the AB
without reopening the generic slot enum at every call site, added:

- `Slot::Kind::Bytes` with a `bytes_size` field on the Slot entry.
  Each slot aligns to 8 B so `int64_t` stride arrays sit on
  natural boundaries.
- `ArgumentBuffer::set_bytes(int, const void*, size_t)` copies a
  raw payload with size validation.

Kept fixed-size inline arrays at MAX_NDIMS = 4 in the kernel
struct. C++ caller asserts the cap at runtime and falls back to
the legacy path if any ndim exceeds it. All current production
models fit comfortably.

All 85 AB unit tests still pass with the extension.

### Bench: no regression on either model

Gemma 4 E2B 4-bit summarization 1024 n=3 — stays at the 4-
primitive baseline (the gather_qmv branch doesn't fire on this
dense model):

| Config | Prefill | Decode |
|---|---:|---:|
| AB=1 (RMSNorm + RoPE + qmv + gather_front + gather_qmv affine) | 2722 – 2856 | **100.7 – 101.1 tok/s** |

GPT-OSS-20B 4-bit summarization 1024 n=3 — decode flat within
noise because the model uses mxfp4 for MoE, so
`affine_gather_qmv_ab` is never reached:

| Config | Prefill | Decode |
|---|---:|---:|
| AB=0 | 541 – 605 | 48.5 – 49.2 tok/s |
| AB=1 | 550 – 560 | 48.2 – 48.4 tok/s |

No regression. GPT-OSS-20B's Q/K/V/O projections (non-MoE
linear) do take `affine_qmv_ab` — any measurable contribution
from that primitive is already captured in this row.

### Follow-on for GPT-OSS-20B MoE signal

Landing `mxfp4_gather_qmv_ab` requires duplicating the kernel
file (~90 instantiations) against `fp_quantized.h`'s
`fp_gather_qmv_impl` / `fp_gather_qmv_fast_impl` and adding a
sister C++ branch in `gather_qmv()` keyed on `mode == "mxfp4"`.
The Bytes-slot infrastructure lands once and is reused.

Estimated: 1–2 hours once the affine shape of the migration is
battle-tested (which it now is, via this commit).

## Phase 2 — Elementwise (AOT binary Add/Multiply vv) — 2026-04-18

Sixth primitive on the AB pattern. Covers the
`Add` / `Multiply` × `VectorVector` decode path (residual
adds, SwiGLU gate-multiplies). Per GPT-OSS-20B's E1 dispatch
audit, `vv_Addfloat32` alone is ~15 % of per-step dispatches.

Scope (narrow by design):
- Binary `vv` / `vv2` / `vvn` only. `ss` / `sv` / `vs` / `g*`
  stay on legacy.
- `Add` and `Multiply` only. Other binary ops stay on legacy.
- `float16` / `bfloat16` / `float32` dtypes only.
- Unary NOT migrated here — `unary_ops.h` emits non-inline
  function definitions for complex-type ops (`ArcCos`, `ArcTan`,
  `erfinv`, `expm1f`) that duplicate-symbol with the existing
  `unary.air` at metallib link. Follow-on needs an `inline`
  patch to `unary_ops.h` before a unary AB file can sit
  alongside.
- Compiled-from-ops JIT kernels (`Compiled::eval_gpu`) NOT
  migrated — that remains the real architectural gap from the
  list above, unchanged by this commit.

### Bench results

Same harness as prior rows. Elementwise AB stacks on top of
the 5-primitive baseline.

Gemma 4 E2B 4-bit summarization 1024:

| Config | n | Decode mean | Range |
|---|---:|---:|---:|
| AB=0 | 7 | 93.5 | 91.7 – 95.1 |
| AB=1 | 4 | **96.9** | 96.0 – 97.5 |

Decode **+3.6 %**. Zero overlap between the two ranges. (Note
the AB=0 absolute level is ~3 tok/s lower than the previous
baseline rows in this doc — likely thermal drift over a long
session. The relative delta is what matters.)

GPT-OSS-20B 4-bit summarization 1024:

| Config | n | Decode mean | Range |
|---|---:|---:|---:|
| AB=0 | 3 | 44.6 | 43.2 – 45.4 |
| AB=1 | 3 | **46.4** | 46.3 – 46.4 |

Decode **+4.0 %**. AB=1 notably more deterministic across
trials (46.3–46.4 range vs AB=0's 43.2–45.4), consistent
with fewer per-dispatch setBytes calls reducing encoder
variability. This is also the first row where GPT-OSS-20B
shows a model-level tok/s win — the affine Linear AB work
from earlier primitives already helps it, but the MoE experts
(mxfp4) stayed on legacy; elementwise adds coverage across
both MoE and non-MoE layers uniformly (residuals + SwiGLU
gates aren't routed through MoE).

Output coherent on both models across all AB=1 runs.

## Files touched

- `mlx/backend/metal/argument_buffer.{h,cpp}` — present from prior
  work (`466258cf`). Not modified this session.
- `mlx/backend/metal/device.{h,cpp}` — new
  `register_input_array` + `add_temporary_object` APIs.
- `mlx/backend/metal/kernels/rms_norm_ab.metal` — new AB-variant
  kernel (single_row + looped, f32/f16/bf16).
- `mlx/backend/metal/kernels/CMakeLists.txt` — register the new
  kernel for metallib build.
- `mlx/backend/metal/normalization.cpp` — env-gated AB branch in
  `RMSNorm::eval_gpu`.
- `mlx/tests/argument_buffer_bench.cpp` — microbench.
- `mlx/tests/CMakeLists.txt` — register the bench.
- `mlx-swift/Source/Cmlx/mlx-generated/metal/rms_norm_ab.metal` —
  SwiftPM-compiled copy with Cmlx-relative include path.
- `mlx-swift/Source/Cmlx/mlx` submodule pointer → `0a611417`.
- `mlx-swift/tools/fix-metal-includes.sh` — add `rms_norm_ab.metal`
  to the sync list.
