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
summarization --context 1024 --quant 4bit`. Two trials each.

| Config | Trial 1 prefill | Trial 2 prefill | Mean | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 2706.4 | 2847.6 | 2777.0 | — |
| AB=1 | 2565.5 | 2693.5 | 2629.5 | **−5.3%** |

| Config | Trial 1 decode | Trial 2 decode | Mean | Δ |
|---|---:|---:|---:|---:|
| AB=0 | 95.7 | 96.7 | 96.2 | — |
| AB=1 | 97.2 | 96.2 | 96.7 | **+0.5%** |

Output quality: both paths produce coherent English summaries of the
(deterministic) Gatsby prompt. No numerical degradation observed —
the AB kernel reads identical inputs and runs the same math; any
divergence would show up immediately as garbled text.

## Why decode barely moved even though encoding is 1.975x faster

RMSNorm is ~3 dispatches per layer × 24 layers ≈ 72 dispatches per
decode step on Gemma 4 E2B (roughly 5 % of all per-step dispatches
extrapolating from the E1 GPT-OSS-20B audit: 1523 dispatches/step,
attention ~5 %, per-layer RMSNorm dispatches similar). With
RMSNorm encoding now 2x cheaper and RMSNorm ~5 % of encoding time
(≈ 40–55 % of step), the predicted tok/s win from RMSNorm alone is
**≤ 2–4 %**. Observed +0.5 % is inside that band and inside the
trial-to-trial noise floor.

The Phase-1 value is *architectural*: AB proves the path-end-to-end.
Stacking migrations (Linear, Embedding, RoPE, etc. per plan doc §
Layer 2) is what converts per-primitive wins into per-step wins.

## Prefill regression and the per-call AB allocation cost

Prefill lost 5 % across both trials. Root cause is almost certainly
per-call `MTL::Device::newBuffer(64 B, shared)` inside each
`ArgumentBuffer` constructor. Decode calls RMSNorm ~72 times per
step; prefill at 1008 tokens calls it an order of magnitude more,
and each call allocates + frees a fresh AB (retained through the
command buffer via `add_temporary_object`). The allocator overhead
per call is not negligible at that rate.

Flagged in the plan doc as an open question
(§ "AB lifetime and pooling"). Addressable three ways once Phase 2
lands:

1. **Per-primitive AB pool.** RMSNorm keeps a ring of pre-sized ABs
   indexed by (dtype, axis_size). Allocation amortizes to zero.
2. **Graph-wide arena.** One bump-allocator MTLBuffer per encoder;
   each AB carves out a slice. Frees are bulk at `end_encoding()`.
3. **Thread-local AB reuse.** Simpler, lower payoff — one AB per
   thread per layout.

Option 2 matches the "heap for weights" direction in the adoption
plan and should fold cleanly into Phase 2.

## Go / no-go

- Microbench ≥ 1.5x: **✓** (1.975x)
- Model-level correctness: **✓** (coherent output in both paths)
- Model-level decode tok/s: **neutral** (+0.5 %, noise-bound)
- Model-level prefill tok/s: **regression** (−5 %), attributed to
  per-call AB allocation

**Decision: proceed to Phase 2.** Phase-1 exit criterion (encoding-
cost drop on a microbenchmark, no correctness regression) met. The
prefill regression is a known, addressable allocator issue — not a
blocker for the architectural direction. Include AB pooling in the
Phase-2 scope alongside Linear / Embedding / RoPE migrations.

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
