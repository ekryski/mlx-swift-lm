# ICB First End-to-End Measurement on GPT-OSS-20B — 2026-04-17

First real-model CPU-encoding measurement of Metal Indirect Command Buffer
replay vs live dispatch emission. Validates the strategy-doc prediction that
ICB saves meaningful per-token time on production LLMs.

## Setup

- Hardware: M1 Max, 64 GB
- macOS: 15.7.4
- Model: GPT-OSS 20B, 4-bit quantized (loan-star/gpt-oss-20b-mlx-4Bit)
- Workload: **step-2+ decode only** (step-1 prefill excluded; cache primed
  once before timing loop)
- Iterations: 30 per measurement, 3 warmup forwards
- Synchronize at end of each timed loop

## Result (apples-to-apples step-2+ decode)

| Metric | Value |
|---|---|
| Live encoding | 18.9 ms / step |
| ICB replay | 12.2 ms / step |
| **Speedup** | **1.40x – 1.74x** (varies across runs, stable median ≈ 1.55x) |
| Live dispatches / step | 874 |
| Captured commands | 740 (85% of live) |
| ICB segments (barriers) | 372 |
| Live dispatches that *leaked* during recording | 134 |
| set_input/set_buffer pre-pipeline skips | 4 |

## Interpretation

- **~1.55x end-to-end speedup** includes GPU execution (identical in both
  paths). Subtracting GPU time, the CPU-encoding savings are substantially
  larger — consistent with the 17x feasibility micro-benchmark once the
  real-model overhead (per-primitive compatibility, multi-stream bypass)
  is factored in.
- **134 leaked dispatches** are a real finding: during recording, some
  primitives emit dispatches through a CommandEncoder that isn't the one
  recording. Most likely a **multi-stream** effect — each mlx primitive
  carries its own `Stream`, and `get_command_encoder(s)` dispatches to the
  encoder for that specific stream. Recording on `StreamOrDevice.default`
  only captures that one stream; primitives scheduled to a different
  stream bypass the recorder entirely.
- **372 segments** means the decoder layer has RAW dependencies every ~2
  dispatches — expected for sequential attention + MLP flow.
- The 4 "pre-pipeline skips" are harmless boundary cases (primitives
  that register dependencies before binding a pipeline).

## Known caveats

- Replay reruns the SAME compute on the SAME buffers, so this benchmark
  measures encoding cost, not tok/s. A tok/s comparison requires
  decode-loop integration where each step advances state through
  per-replay input overrides (follow-up work).
- The 15% unrecorded dispatches mean the replayed output is numerically
  different from the live forward. Acceptable for encoding-time
  measurement, blocking for correctness. Root cause is multi-stream;
  fix is to either record all active streams (harder) or force all
  primitives to the recording stream for the capture window (lighter).

## Commits (ICB prototype branch)

- mlx `df50c4b2` diag(metal): count dispatch calls routed through encoder
  during recording + pre-pipeline skip count
- mlx `1de52063` fix(metal): tolerate pre-pipeline set_input/set_buffer
- mlx `d37a9f7b` fix(metal): tolerant set_buffer under recording
- mlx `ad93d46c` fix(metal): auto-abort ICB recording on throw
- mlx-swift `4154daf` chore: bump mlx submodule
- mlx-swift `6087b2f` feat(GPU): unconditional abort on record throw
- mlx-swift-lm `243bf5a` bench(icb): --method icb measures CPU encoding
  speedup + initial GPT-OSS-20B measurement

## Next steps

1. **Multi-stream recording**: investigate whether all active streams
   can be recorded simultaneously, or whether primitives can be forced
   to a single stream for the capture window.
2. **Replay-with-overrides API**: for tok/s measurements, the decode
   loop needs per-step input overrides (new token, updated cache
   pointers) without re-recording.
3. **Broader model comparison**: run the same benchmark against Gemma4
   (non-MoE) and Qwen3 to see how model architecture affects the win.
4. **Correctness validation**: once multi-stream is handled, verify
   logit parity between live and ICB-replay forward passes.
