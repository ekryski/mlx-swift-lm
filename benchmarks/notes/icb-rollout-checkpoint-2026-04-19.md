# ICB Rollout Checkpoint — 2026-04-19

Continuation of `persistent-ab-sdpa-e2e-2026-04-18.md` and the
`persistent-ab-pilot-design-2026-04-18.md` plan. This session completed
the AB-and-tagging foundation across all 9 hot-path primitives and
landed KV pinning, but stopped short of decode-loop ICB orchestration.

## What landed this session

### mlx (`ek/persistent-ab-pilot`)
- **`b71d1909`** — RoPE accepts optional `PersistentAb` handle
  (Class 2; mirrors RMSNorm + SDPA pattern). Both layouts (6-slot
  base path, 7-slot freqs path).
- **`2210f201`** — `tag_ab_binding(MTL::Buffer*)` helper on
  `CommandEncoder` + auto-counter reset on `begin_icb_recording`.
  Wired into all 7 Class-1 primitive AB branches: binary, unary,
  indexing (gather_front), quantized (affine_qmv +
  affine_gather_qmv), compiled (JIT). Also wired into the transient
  AB fallback paths of RMSNorm / SDPA / RoPE for caller flexibility.
- mlx test suite: **304/304 green** under `MLX_METAL_AB=1`.

### mlx-c (`ek/persistent-ab-pilot`)
- **`ffa07da`** — C API for RoPE `PersistentAb`:
  `mlx_metal_persistent_ab_new_rope` (6 slots),
  `mlx_metal_persistent_ab_new_rope_freqs` (7 slots),
  `mlx_metal_persistent_ab_set_scalar64`, `mlx_fast_rope_ab`.

### mlx-swift (`ek/persistent-ab-pilot`)
- **`5185b22`** — `PersistentRopeAbHandle` + `PersistentRopeFreqsAbHandle`
  Swift classes, two `MLXFast.ropeAb()` overloads. Vendored
  `Source/Cmlx/include/mlx/c/{fast,metal}.h` updated with the new
  decls. Submodule bumps to mlx `b71d1909` + mlx-c `ffa07da`.

### mlx-swift-lm (`ek/persistent-ab-pilot`)
- **`e3869b6`** — Opt-in KV cache pre-allocation:
  - `KVCacheSimple.preallocateMaxSize: Int?`
  - `RotatingKVCache.preallocateFull: Bool`

  Both default off. When enabled, the first `update()` allocates
  the full ceiling so the K/V buffer gpuAddresses are stable for
  the cache lifetime.

## Architecture decision: Class 1 vs Class 2

Per the design doc, the 9 AB-migrated primitives split into two
classes by per-step scalar dynamism:

- **Class 2 (per-step scalars dynamic)**: SDPA (T_k), RoPE (offset).
  Use `PersistentAb` handles owned at the layer level; mlx C++
  rewrites contents per call; Swift writes only the static
  scalars at handle construction.
- **Class 1 (no per-step scalar changes)**: binary, unary, gather,
  affine_qmv, affine_gather_qmv, Compiled JIT, plus the
  transient-AB fallbacks of Class-2 primitives. Use the existing
  `IndirectCommandRecorder::tag_binding` + `replay_with_overrides`
  mechanism. Each transient AB is auto-tagged with a sequential
  ID at record time; the decode-loop orchestrator overrides each
  tag with the next step's AB MTLBuffer at replay.

Both halves are now in place at the **mlx C++ level**. The Swift-
side decode-loop orchestration that consumes them is not yet
written — see "Remaining work" below.

## Remaining work

### 1. Decode-loop ICB orchestration (the critical missing piece)

The infrastructure to **collect** Class-1 AB overrides at replay time
isn't implemented yet. The fundamental constraint:

- During the recorded forward pass, mlx C++ allocates transient
  ABs and tags each with a sequential ID (1, 2, 3, ...).
- At replay time, the K-th tag's MTLBuffer must be the K-th
  current-step AB. But ABs are constructed inside `eval_gpu` calls
  during a normal forward pass, and a normal forward pass also
  dispatches the kernels — defeating the ICB.

**Three viable architectures** (none implemented):

#### Option (a) — Build-only encoder mode
Add a `CommandEncoder::build_only_mode_` flag. When set, primitives:
- Allocate output arrays + transient ABs (same code path).
- Call `tag_ab_binding` on each AB so the recorder can pair it
  with the recorded ordinal.
- Skip `set_buffer` + skip `dispatch_*`.

The Swift orchestrator runs `model(...)` in build-only mode to
collect the new ABs, then calls `replay_with_overrides`.

**Cost**: medium-large. Need to thread the flag through every
primitive's `eval_gpu` and confirm allocations don't depend on
inputs being computed. Probably 1-2 days.

#### Option (b) — Pinned-activation pool
Pre-allocate every layer activation at fixed addresses upfront.
The recorded ICB's pointer references stay valid because
addresses don't change. No overrides needed.

**Cost**: large. Requires inferring activation shapes per layer
and adding pinning hooks. The design doc notes this as the
"pinned activations" stretch goal (largest refactor, lowest
marginal gain over options a/c).

#### Option (c) — Stable AB pool ordering
Reuse the same AB MTLBuffer for the K-th AB across steps via a
deterministic pool. Recorded ICB's pointer K matches step-N+1's
pointer K because the pool returns them in the same order.

**Cost**: medium. Requires AB pool refactor with strict ordering
guarantees, or a parallel pool keyed by recording-time ordinal.

**Recommendation**: Option (a). Cleanest contract, smallest blast
radius, fits the existing `tag_binding` / `replay_with_overrides`
contract perfectly.

### 2. End-to-end correctness gate

Once orchestration lands, verify byte-equivalence:
- Steps 1–2 live (warmup).
- Step 3 record + dispatch.
- Step 4+ replay-with-overrides.
- Token sequence matches the live-only baseline bit-for-bit.

### 3. tok/s benchmark

GPT-OSS-20B 4-bit decode. Live baseline ≈ 47 tok/s. Per the
design doc, expected ceiling with full Class-1+2 pipeline:
~150–170 tok/s (CPU-encoding-bound limit lifted; GPU ≈4 ms +
CPU ≈1 ms per step).

## Why this stopped short

`tag_ab_binding` is a one-liner per primitive — straightforward to
land. The override-collection mechanism is genuinely a multi-day
architectural change (option a, b, or c above). I don't have
the runway to ship it correctly in one session, and the wrong
architecture here would be expensive to undo.

The infrastructure pieces in place are reusable for whichever
orchestration option is chosen — they form the "tagging" half
of the tag/override contract.

## Code state at session end

All 4 repo branches (`ek/persistent-ab-pilot`) pushed to GitHub.
`mlx-swift-lm` builds clean against the new mlx-swift tip.
mlx test suite green. No regression risk to existing functionality
— all new behavior is behind opt-in flags
(`MLX_METAL_AB=1`, `MLX_PERSISTENT_AB=1`, `KVCacheSimple.preallocateMaxSize`,
`RotatingKVCache.preallocateFull`).
