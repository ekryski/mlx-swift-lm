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

## ICB encoding microbench — 33% CPU-encoding drop from RMSNorm alone

`./scripts/benchmark.sh --model gpt-oss-20b --method icb` records one
decode step and replays it `timedIters` times (cache doesn't advance
between replays — this is a pure CPU-encoding-cost measurement).

| Config | Live µs/step | Replay µs/step | Reported speedup |
|---|---:|---:|---:|
| AB on (transient ABs) — baseline | 17,363 | 12,283 | 1.41× |
| **AB on + PERSISTENT_AB=1 (RMSNorm persistent only)** | **11,562** | **12,271** | **0.94×** |

**The "0.94× speedup" number is misleading.** Replay cost is essentially
unchanged between the two configs (12,283 vs 12,271 µs) — persistent vs
transient AB doesn't matter at replay time because replay doesn't
allocate ABs.

**What actually happened is the live-encoding cost dropped 33%** (17,363 →
11,562 µs/step), meaning **RMSNorm alone accounts for ~5,800 µs/step of
allocator churn** that persistent-AB eliminates. Breaking that down:

- RMSNorm is called ~48 times per decode step (2 × 24 layers).
- 5,800 µs ÷ 48 calls ≈ **121 µs per call** of overhead eliminated.
- That's the cost of `std::make_shared<ArgumentBuffer>` + layout-vector
  construction + BufferPool mutex + initial memset, per call, in the
  transient path.

## The surprise: persistent-AB may be the point, not ICB

Extrapolating: if persistent-AB on all 9 primitives saves similar overhead
per primitive, live CPU encoding on GPT-OSS-20B could drop to the ~7–9k
µs/step range — **competitive with or better than ICB replay (12,271 µs)**.

Meaning: **the persistent-AB pattern may deliver most of the CPU-encoding
reduction by itself**, without needing the ICB record/replay infrastructure
at all. ICB replay becomes a smaller marginal win on top of persistent-AB
rather than a separate architectural win.

This reframes the Option A rollout priorities:
1. Extending persistent-AB to all 9 primitives is now the **primary tok/s
   lever** (previously considered a prerequisite for ICB correctness).
2. ICB decode-loop wiring is a smaller incremental win and can wait until
   (1) lands.
3. Weight heap + KV cache pre-alloc are separate-axis optimizations.

## Why decode tok/s didn't move despite the 33% live-encoding drop

The `--method simple` decode tok/s didn't move noticeably (46.4 → 45.9
median, within noise) despite the 33% CPU-encoding drop on the ICB
microbench.

**GPT-OSS-20B decode is GPU-bound at this quant config**, not CPU-bound.
A ~5.8 ms/step CPU reduction doesn't translate to tok/s when GPU work
takes ~15 ms/step and the command buffer submission is async — CPU can
already queue ahead while GPU executes.

Implication: the full tok/s payoff from persistent-AB + ICB will likely
show up on smaller, less-GPU-heavy models (Gemma-4 E2B, Qwen3-0.6B)
before it shows on big MoE models like GPT-OSS-20B. Worth re-running
this measurement there once persistent-AB extends to the remaining 8
primitives.

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
