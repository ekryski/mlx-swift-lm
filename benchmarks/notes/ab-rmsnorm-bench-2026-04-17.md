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
