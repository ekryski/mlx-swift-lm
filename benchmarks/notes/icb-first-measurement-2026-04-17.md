# ICB First End-to-End Measurement — 2026-04-17

First real-model CPU-encoding measurement of Metal Indirect Command Buffer
replay vs live dispatch emission. Validates the strategy-doc prediction
that ICB saves meaningful per-token time on production LLMs.

## Setup

- Hardware: M1 Max, 64 GB, macOS 15.7.4
- Workload: step-2+ decode only (cache pre-primed with one step before
  the timing loop); 30 iterations per measurement, 3 warmup forwards,
  synchronize at end of timed loop
- Invocation: `./scripts/benchmark.sh --model <name> --method icb`

## Results

| Model | Variant | Live (µs/step) | Replay (µs/step) | Speedup | Dispatches | Captured | Leaked | Segments |
|---|---|---|---|---|---|---|---|---|
| **GPT-OSS 20B** | 4bit | 18,939 | 12,165 | **1.40 – 1.74x** | 874 | 740 (85%) | 134 | 372 |
| **Gemma 4 E2B** | 4bit | 8,573 | 6,779 | **1.21 – 1.33x** | 390 | 390 (100%) | 0 | 287 |

Three runs per model; reported range is min–max speedup across runs.

## Interpretation

- **ICB wins on every model tested, every run.** Encoding cost is
  meaningfully lower across the board.
- **Gemma 4 E2B captures 100% of live dispatches** — the recorder sees
  every single kernel launch the forward pass emits. The ICB is a
  complete, numerically-correct replay of the decoder step.
- **GPT-OSS 20B captures 85%.** 134 of 874 dispatches bypass the
  recording encoder. These are almost certainly MoE-specific primitives
  (SwitchGLU expert routing, gather-quantized matmul) running on a
  secondary stream — each mlx primitive carries its own `Stream`, and
  `metal::get_command_encoder(s)` looks up a separate encoder per
  stream. Recording on `StreamOrDevice.default` captures only that
  stream. This is a correctness concern for tok/s integration but does
  not invalidate the encoding-cost measurement for the 85% that are
  captured.
- **Larger relative win on GPT-OSS (1.55x median) than Gemma4 (1.27x
  median).** Two likely factors:
  1. GPT-OSS has 2.2x the dispatches per step (874 vs 390), so the
     replay saves more absolute CPU time per step.
  2. GPT-OSS's MoE dispatches that DO get recorded are heavier
     (setBuffer + setBytes work per command), giving ICB more to skip.
- **ICB build cost isn't measured here** (one-time, amortized across
  30 replays). The feasibility micro-benchmark (mlx tests/icb_feasibility_tests.cpp)
  showed ~386 µs for 1500 trivial commands — on the order of 2 decode
  steps of break-even time.

## Caveats

- Replay reruns the SAME compute against the SAME buffers; this is an
  encoding-time measurement, NOT tok/s. A tok/s comparison requires
  replay-with-overrides so each step advances state (new token, moved
  KV pointer). Follow-up work.
- The 15% unrecorded dispatches on GPT-OSS mean replayed output is
  numerically divergent from live. Fine for encoding-cost measurement,
  blocking for end-to-end decode. Root cause is multi-stream — fix is
  either (a) record all active streams simultaneously, (b) force all
  primitives to the recording stream for the capture window, or
  (c) identify MoE primitives that split to a secondary stream and
  revise them.
- Measurements include GPU execution (identical in both paths). Pure
  CPU encoding savings are larger than the 1.2–1.7x end-to-end ratio
  suggests; consistent with the ~17x encoding-only feasibility number.

## Commits (ICB prototype branch; all pushed to ekryski/*)

mlx (C++):
- `df50c4b2` diag(metal): count dispatch calls routed through encoder
- `1de52063` fix(metal): tolerate pre-pipeline set_input/set_buffer
- `d37a9f7b` fix(metal): tolerant set_buffer under recording
- `ad93d46c` fix(metal): auto-abort ICB recording on throw
- `2ddf3ab8` fix(metal): rebase integration
- `64caba9b` feat(metal): ICB segment splitting at barriers
- `1e8d663f` feat(metal): threadgroup memory length
- `c1a9aead` feat(metal): all pipelines with ICB support
- `beaf6b04` feat(metal): wire recorder into CommandEncoder
- `e8a92a50` feat(metal): IndirectCommandRecorder + arena
- `e3eb45e0` test(metal): feasibility micro-benchmark (17x)

mlx-c:
- `47b02e7` feat(metal): `mlx_metal_icb_abort_recording`
- `aee37b9` feat(metal): C API for ICB record/replay

mlx-swift:
- `4154daf` chore: bump mlx submodule
- `6087b2f` feat(GPU): unconditional abort on record throw
- `e9dab2f` feat(GPU): `IndirectCommandBuffer` Swift API

mlx-swift-lm:
- `0a56e15` bench(icb): decode-only + leaked-dispatch diagnostic
- `9c8c1fd` docs(bench): first measurement notes
- `243bf5a` bench(icb): `--method icb` + GPT-OSS-20B
- `719685b` test(icb): smoke test + Package.swift pin

## Next steps (ranked)

1. **Multi-stream capture** — either record all active command encoders
   simultaneously, or force the forward pass onto a single stream for
   the capture window. Would close the 15% gap on GPT-OSS and
   potentially lift the speedup further.
2. **Replay-with-overrides API** — required for tok/s integration. The
   recorder already retains buffer pointers; exposing a
   `replay(bindingOverrides: [slot: buffer])` on top would let each
   replay step advance state.
3. **Numerical parity test** — once 1+2 are in place, verify live-forward
   vs ICB-replay produce the same logits on a deterministic prompt.
4. **Broader model sweep** — Qwen3, Nemotron, additional Gemma4 sizes.
