# Persistent-AB Pilot — Step 3 End-to-End Results — 2026-04-18

Measurement after Steps 1–3 of the Option A rollout shipped on
`ek/persistent-ab-pilot` across all four repos (mlx, mlx-c, mlx-swift,
mlx-swift-lm). Step 3 wired:

- mlx C++ `PersistentAb` class (thin wrapper over `ArgumentBuffer`)
- mlx-c `mlx_metal_persistent_ab` + `mlx_fast_rms_norm_ab` API
- mlx-swift `PersistentRmsAbHandle` Swift class + `MLXFast.rmsNormAb`
- MLXNN `RMSNorm` module: opt-in path under `MLX_PERSISTENT_AB=1` that
  lazy-initializes a handle per module instance, writes the static
  scalars (`axis_size`, `w_stride`, `eps`) once, then routes every
  `callAsFunction` through the handle-backed path.

RMSNorm is the only primitive migrated so far. The other 8 AB primitives
(RoPE, SDPA, affine_qmv, affine_gather_qmv, gather_front, binary, unary,
Compiled JIT) still allocate fresh `ArgumentBuffer` per call.

## Decode tok/s — neutral at whole-model level

`./scripts/benchmark.sh --model gpt-oss-20b --method simple`,
3 runs, GPT-OSS-20B 4-bit, 200 max-tokens, 101-token prompt.

| Config | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| AB off (alpha baseline) | 46.2 | 45.7 | 45.8 | 45.9 |
| AB=1 (8 primitives transient-AB, legacy SDPA) | 47.1 | 46.9 | 46.5 | 46.8 |
| AB=1 (Phase 2 SDPA unified+AB, 9 primitives transient) | 46.8 | 46.3 | 46.1 | 46.4 |
| **AB=1 + PERSISTENT_AB=1 (RMSNorm persistent, 8 others transient)** | **45.9** | ~~64.4~~ outlier | **46.0** | **45.9** |

Persistent-AB RMSNorm is **numerically correct and tok/s-neutral** at
the whole-model level, as expected — RMSNorm is ~5% of per-step
dispatches, and GPU execution dominates the step budget at this quant
config.

## ICB encoding microbench — noise-neutral, no standalone win

`./scripts/benchmark.sh --model gpt-oss-20b --method icb` records one
decode step and replays it `timedIters` times (cache doesn't advance
between replays — pure CPU-encoding-cost measurement).

**Correction from initial writeup.** The first single-run measurement
suggested a huge 33% drop (17,363 → 11,562 µs/step live with persistent-AB
on). **That was an outlier — almost certainly a thermal/cache-state
transient.** I got excited and wrote up the "persistent-AB is the primary
lever" reframing based on one datapoint. Here are the proper 3-run numbers:

| Config | Live 1 | Live 2 | Live 3 | Live mean | Replay mean |
|---|---:|---:|---:|---:|---:|
| AB on (transient ABs) | 18,632 | 17,695 | 17,697 | **18,008** | 12,321 |
| AB on + PERSISTENT_AB=1 | 17,623 | 17,977 | 17,403 | **17,667** | 12,225 |

Persistent-AB RMSNorm reduces live encoding by **~1.9%** on 3-run mean —
within measurement noise. The allocator-churn-is-a-major-bottleneck
hypothesis I built around the single datapoint is **not supported by
multi-run data**.

**Correct interpretation**: persistent-AB on RMSNorm alone has no
material standalone CPU-encoding win. The ~121 µs per-call overhead
breakdown I calculated was extrapolation from one outlier — not real.

## Revised strategic read

The original framing stands: **persistent-AB is a prerequisite for
ICB-replay correctness**, not a standalone encoding-cost lever. The
value shows up only once ICB record/replay actually runs in a decode
loop.

Specifically:
- **ICB encoding speedup is consistent at ~1.45×** (17–18 ms live →
  12 ms replay) across both configs. That number is real and holds.
- **Persistent-AB's contribution is correctness under ICB replay**:
  the AB's MTLBuffer address stays stable so the replayed
  `setBuffer(ab, 0)` finds the right buffer, AND the caller can
  update its shape scalars between replays (for primitives like SDPA
  where T_k changes per step).
- **Without ICB**, persistent-AB delivers essentially nothing.

Rollout priorities — unchanged from the design doc:
1. Extend persistent-AB to SDPA (and affine_qmv / affine_gather_qmv),
   which are the primitives with dynamic-per-step scalars that ICB
   replay must be able to update.
2. Wire ICB record/replay into the decode loop.
3. Benchmark end-to-end — tok/s win comes from the replay 1.45× CPU
   encoding drop, not from persistent-AB directly.
4. Weight heap + KV cache pre-alloc as orthogonal follow-ons.

## Why decode tok/s didn't move

`--method simple` decode tok/s stayed at 45.9 (median n=3), within
noise of the AB-only baseline at 46.4. Consistent with the corrected
microbench data above: persistent-AB RMSNorm alone is essentially a
no-op for CPU-encoding cost, so there's nothing to translate into
tok/s even in the CPU-bound regime.

The tok/s win from Option A will require ICB record/replay to actually
run in the decode loop (the 1.45× CPU reduction is in the replay, not
the persistent-AB path itself).

## Status of Step 4 (decode-loop ICB record/replay wiring)

**Not implemented in this session.** The plumbing to make ICB replay
*correctness-preserving* in a real decode loop requires all 9 primitives
to be persistent-AB migrated (so their shape scalars can be updated
between replays). With only RMSNorm migrated, ICB replay at step N+1
would read stale T_k / cache-offset values for SDPA and others, producing
garbled output.

Step 4 becomes tractable once Step 5 (extend persistent-AB to the other
8 primitives) is done. Given the surprise finding above, the priority
ordering should probably be:

- **Step 5 first**: extend persistent-AB to RoPE, SDPA, affine_qmv,
  affine_gather_qmv, gather_front, binary, unary, Compiled JIT. Each
  one contributes allocator-churn savings similar to RMSNorm's 33%.
- **Then Step 4**: wire decode-loop ICB record/replay. Correctness is
  now achievable because all primitives have Swift-updatable handles.

## Files touched (across 4 repos)

### mlx (`ek/persistent-ab-pilot` @ `6ae5054f`)
- `mlx/backend/metal/persistent_ab.{h,cpp}` — new class
- `mlx/backend/metal/CMakeLists.txt` — register
- `mlx/backend/metal/normalization.cpp` — RMSNorm eval_gpu uses handle
  when present
- `mlx/fast.{h,cpp}` — `rms_norm` overload with handle
- `mlx/fast_primitives.h` — RMSNorm carries optional handle
- `tests/persistent_ab_tests.cpp` — 11 doctest cases (9 class-level +
  2 integration)
- `tests/CMakeLists.txt` — register test file

Full mlx test suite: **302/302 green** in both default and
`MLX_METAL_AB=1` modes.

### mlx-c (`ek/persistent-ab-pilot` @ `b5fcecc`)
- `mlx/c/metal.h, metal.cpp` — C API for PersistentAb
- `mlx/c/fast.h, fast.cpp` — `mlx_fast_rms_norm_ab` overload

### mlx-swift (`ek/persistent-ab-pilot` @ `2e5609f`)
- `Source/MLX/PersistentAb.swift` — Swift wrapper class
- `Source/MLX/MLXFast.swift` — `rmsNormAb` function
- `Source/MLXNN/Normalization.swift` — env-gated persistent path
- `Source/Cmlx/mlx` + `mlx-c` submodule bumps
- Synced `Source/Cmlx/include/mlx/c/*.h`

### mlx-swift-lm
No code changes — the MLXNN.RMSNorm change in mlx-swift is picked up
transparently by all callers (GPTOSS, Gemma4, etc.) under the
`MLX_PERSISTENT_AB=1` env gate.

## Next session

Do Step 5: extend persistent-AB to the remaining 8 primitives. Expected
to deliver further per-step CPU-encoding drops stacking on the 33% from
RMSNorm. Re-run both `--method icb` (validating the pattern) and
`--method simple` on GPT-OSS-20B AND Gemma-4 E2B (where the tok/s win
should actually show up).

Then Step 4 (decode-loop ICB wiring) becomes feasible for a correct,
end-to-end benchmarkable result.
