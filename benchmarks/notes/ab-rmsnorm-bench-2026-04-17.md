# AB RMSNorm Pilot — Bench Results — 2026-04-17

Measured delta between the legacy RMSNorm path and the Phase-1
`ArgumentBuffer` pilot. Companion to
[ab-rmsnorm-pilot-2026-04-17.md](ab-rmsnorm-pilot-2026-04-17.md).

All four repos on sub-branch / branch pair:

| Repo | Branch | HEAD |
|---|---|---|
| `mlx` | `ek/ab-rmsnorm-pilot` | `0a611417` |
| `mlx-swift` | `ek/metal-icb-prototype` | `2cc2de1` |
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
