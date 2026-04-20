# ICB decode-loop architecture — 2026-04-20

Architectural reference for the `MLX_ICB_DECODE_LOOP=1` code path.
Walks the data + control flow across all four repos
(mlx-swift-lm → mlx-swift → mlx-c → mlx), what gets cached at
record time vs. mutated per step, and where the Metal API boundary
sits.

Written after a session where we confirmed the decode-loop is a
net throughput loss on GPT-OSS 20B (see
[icb-ab-bench-2026-04-19.md](icb-ab-bench-2026-04-19.md) for
numbers) and wanted to make the cost model explicit before
deciding whether to spend more on segment coarsening or retire the
path.

## 1. The three phases

```
  STEP 0..2                    STEP 3                       STEP 4..N
  ─────────                    ──────                       ─────────
  LIVE WARMUP                  RECORD + INITIAL REPLAY      REPLAY-ONLY

   model(input)          ┌──  pin.record {                  recordedIcb
        │                │                                     .replay(
        ▼                │      IndirectCommandBuffer.          overrides: …)
   sampler               │        recordWithBindings { …  }        │
        │                │                                          ▼
   previous = next       │      // single live eval of            sampler
                         │      //   the recorded graph             │
                         │      capturedIcb.replay()           previous = next
                         │
                         │      sampler
                         │      previous = next
                         └──  }
```

* **Steps 0..2** — plain live forward. Primes the KV cache (so
  the cache.offset when we record is the real decode position
  after prefill + 3 tokens). No recording, no replay. Identical
  to the non-ICB path.

* **Step 3** — one scoped block does three things:

  1. `pin.record` starts a `PinSession` so every `set_data`
     allocation made inside the scope lands in a stable-address
     pool (so downstream replays that refer to those buffers by
     pointer see the same address).
  2. `IndirectCommandBuffer.recordWithBindings` flips the
     `CommandEncoder` into "recording" mode. `eval_gpu` calls no
     longer dispatch — they accumulate into the
     `IndirectCommandRecorder`, which owns one or more
     `MTLIndirectCommandBuffer` *segments*.
  3. After the record scope closes, `capturedIcb.replay()`
     actually executes the forward pass once to produce step-3's
     logits. Sampler runs. Next token is stashed into `previous`.

* **Step 4+** — `recordedIcb.replay(overrides: …)` replays the
  pre-built ICB with the per-step buffer substitutions applied
  (new input-token buffer pointer inside the gather AB, new
  KV-slice-update offsets, updated SDPA `N`, updated RoPE
  offset). No graph walk, no primitive `eval_gpu`, no kernel
  selection.

## 2. What gets captured vs. what mutates

```
   ┌───────────────────── CAPTURED ONCE at step 3 ──────────────────────┐
   │                                                                    │
   │  MTLIndirectCommandBuffer segments[~650]                           │
   │    ├─ each segment owns ~2.3 commands (compute pipelines + slot    │
   │    │   bindings pre-set at record time)                            │
   │    └─ segment split on every RAW dependency                        │
   │                                                                    │
   │  PinSession.slots[~1631]                                           │
   │    └─ every record-time set_data allocation, pinned at stable      │
   │        address so bindings remain valid                            │
   │                                                                    │
   │  record-time AB contents (inside PersistentAb MTLBuffers)          │
   │    ├─ 24 SDPA handles (Q/K/V/out ptrs, strides, scale, gqa, N)     │
   │    ├─ 48 RoPE-freqs handles (in/out/offset/freqs/scale/stride)     │
   │    └─ 3 gather-front handles (src/indices/out/stride/size)         │
   │                                                                    │
   │  tag_locations map                                                 │
   │    └─ name_id → [(segment_idx, command_idx, slot, offset)]         │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘

   ┌──────────────────── MUTATED per replay step ───────────────────────┐
   │                                                                    │
   │  overrides dict passed to replay_with_overrides                    │
   │    ├─ icbInputBinding → previous.tokens buffer (via the            │
   │    │   PersistentGatherFrontAbHandle indices-ptr slot, not         │
   │    │   command-slot rebind — gather packs indices inside its AB)   │
   │    ├─ icbKStartBinding(layer: i) → startArr(writeIdx)              │
   │    ├─ icbVStartBinding(layer: i) → startArr(writeIdx)              │
   │    └─ … deduped to 2 unique startArrs (sliding + full)             │
   │                                                                    │
   │  PersistentSdpaAbHandle.updateNPerLayer(perLayerTk)                │
   │    └─ writes the 24 handles' `N` slot in place (CPU memcpy into    │
   │        each AB's shared-storage buffer)                            │
   │                                                                    │
   │  PersistentRopeFreqsAbHandle.setOffsetOnAll(ropeOffsetArr)         │
   │    └─ rewrites slot 2 (offset buffer ptr) on every live handle     │
   │                                                                    │
   │  PersistentGatherFrontAbHandle.setIndicesPtrOnAll(gatherInput)     │
   │    └─ rewrites slot 1 (indices buffer ptr) on every live handle    │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

## 3. Swift → C++ call chain

```
  TokenIterator.icbStep  (Libraries/MLXLMCommon/Evaluate.swift)
        │
        ├── build overrides + perLayerTk from cache[].icbStepState(…)
        ├── PersistentSdpaAbHandle.updateNPerLayer(perLayerTk)
        ├── PersistentRopeFreqsAbHandle.setOffsetOnAll(…)
        ├── PersistentGatherFrontAbHandle.setIndicesPtrOnAll(…)
        │
        ▼
  IndirectCommandBuffer.replay(overrides:)  (mlx-swift)
        │
        └── mlx_metal_icb_replay_with_overrides(…)
                  │
                  ▼
        CommandEncoder::replay_icb_with_overrides  (mlx, device.cpp)
              ├── input-side memoryBarrier if override buf is a prev output
              └── recorder.replay_with_overrides(enc, overrides)
                        │
                        ▼
              IndirectCommandRecorder::replay_with_overrides  (icb.cpp)
                    │
                    ├── for each (name_id, new_buf, offset) in overrides:
                    │     for each loc in tags_[name_id]:
                    │         seg.icb->indirectComputeCommand(loc.cmd)
                    │            ->setKernelBuffer(new_buf, offset, slot)
                    │         override_extra_buffers.insert(new_buf)
                    │
                    ├── single useResource sweep:
                    │     for buf in all_resources_: enc->useResource(buf)
                    │     for buf in override_extra_buffers: enc->useResource(buf)
                    │
                    └── for si in 0..segments_.size():
                          if si > 0: enc->memoryBarrier(BarrierScopeBuffers)
                          enc->executeCommandsInBuffer(seg.icb, range)
```

## 4. The cost model (per replay step)

Measured numbers from GPT-OSS 20B, ctx = 101 (simple method), M1
Max, with `MLX_ICB_REPLAY_TIMING=1` capturing CPU-side split
timing at the `recordedIcb.replay()` boundary:

```
   ICB decode-loop path (per replay step)
   ──────────────────────────────────────
   replay_submit  (CPU time in recordedIcb.replay)   ≈ 1.3–1.7 ms
   sync_wait      (Stream.synchronize after replay)  ≈ 19–21 ms
   total                                             ≈ 21–22 ms

   Plain AB + PersistentAB path (per decode step)
   ──────────────────────────────────────────────
   forward        (model(input) — graph build +
                   primitive eval_gpu + Metal submits) ≈ 2.3–3.2 ms
   sampler_sync   (convertToToken)                     ≈ 5 µs
   total                                               ≈ 2.6–3.2 ms
```

**This is the real cost model.** The ICB path's 21 ms/step is
not a slow replay — it's a CPU that's *waiting* 19–21 ms/step for
the GPU to complete before the next step can start. The plain
path's 2.6 ms/step is the CPU _not waiting_: Metal's lazy eval
and command-buffer queuing let the CPU keep submitting steps
while the GPU crunches prior ones in parallel. The actual GPU
work per token is roughly the same in both paths (~20 ms on the
GPU for 1 500 kernels); the difference is just whether the CPU
is pipeline-overlapped with the GPU or serialized behind it.

### Why the ICB path has to synchronize

Removing the `Stream.synchronize` after `recordedIcb.replay`
produces garbage output — specifically the "WeWeWe…" loop where
every decode step reuses the record-step's input token. Root
cause: the persistent AB architecture has an implicit
one-replay-in-flight constraint.

```
  Step N          Step N+1
  ──────          ────────
  GPU still                  CPU calls setIndicesPtrOnAll
  reading AB's   ──race──→   on the SAME AB handle — over-
  slot 1 (input              writes slot 1's bytes while GPU
  token ptr)                 is still reading them
                             ↓
                             Replay reads junk pointer or
                             mid-write state → garbage
```

Every persistent AB (SDPA `N`, RoPE offset, gather-front indices
ptr) is backed by a **single** shared-storage `MTLBuffer`. Each
step's orchestrator mutates that buffer's bytes from CPU before
issuing the next replay. If the previous replay's GPU work hasn't
completed, the mutation clobbers values still being consumed.

The plain AB + PersistentAB path avoids this because **every
step allocates fresh transient ABs** — the "persistent" handles
on that path only exist for RoPE/SDPA reads-from-stable-
buffers, and the heavy per-step state (K/V write positions, T_k)
lives in per-call allocations that can't race.

### What would remove the sync

Multi-way rotation of each persistent AB:

```
  ab_handle                 ab_handle[0]  ab_handle[1]  ab_handle[2]
    │                         │              │              │
    ▼                         ▼              ▼              ▼
  one AB            ──→    rotated among N pre-allocated ABs
  (CPU must sync             (step N uses AB[N % 3], CPU can
   before reusing)            prep AB[N+1 % 3] while GPU reads
                              AB[N % 3])
```

Requires:
* Allocating N copies of each persistent AB at record time.
* Retargeting the recorded `set_buffer(ab_mtl, 0)` slot per step
  via `tag_ab_binding` overrides — same plumbing as K/V start
  overrides, wired to `PersistentSdpaAbHandle`/`Rope`/`Gather`
  registries.
* Tracking which rotation index is safe to mutate (N-deep pipeline
  depth, so after N steps the Nth-oldest AB is guaranteed
  GPU-free).

Not a small change. Feasibility open — would need to verify that
per-step tag_ab_binding overrides actually compose with the
existing K/V overrides on the same replay.

## 5. Why 650 segments

The recorder splits into a new segment on every RAW dependency:

```
  CommandEncoder::maybeInsertBarrier()              (device.cpp:518)
     │
     ├── live mode:     get_command_encoder()->memoryBarrier(…)
     └── record mode:   active_recorder_->split_segment()

  Each primitive's eval_gpu calls set_input_array / set_output_array,
  which populate prev_outputs_ / next_outputs_ via:

     needs_barrier_ |= prev_outputs_.contains(input_buffer)

  maybeInsertBarrier is called before every dispatch. If
  needs_barrier_ is set, either a live memoryBarrier fires or the
  recorder splits into a new segment. In both cases the same
  barrier count is generated.
```

So: **the live path and the ICB path emit the same number of
memory barriers**. What differs is the command-level dispatch
cost (direct `dispatchThreadgroups` × 1 500 vs. indirect
`executeCommandsInBuffer` × 650).

Metal only exposes `IndirectCommandTypeConcurrentDispatch` for
compute (no serial variant). Commands within one ICB segment run
concurrently, so segmenting + per-segment barriers is the only
correct way to represent RAW dependencies across the ICB
boundary.

## 6. The correctness mechanisms — what each fix solved

This is the part that's easy to forget — quite a few session-2
fixes land as invariants the architecture depends on.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Per-step mutation            What breaks if it stops working    │
  ├─────────────────────────────────────────────────────────────────┤
  │  input-token ptr (gather AB)  every replay runs with the record- │
  │                               step's token → "WeWeWe" loop      │
  │                                                                  │
  │  K/V start offsets per layer  writes clobber wrong cache slot    │
  │                               or go out of bounds on sliding KV  │
  │                                                                  │
  │  SDPA N per layer             SDPA reads wrong number of K       │
  │                               entries → wrong attention scores   │
  │                                                                  │
  │  RoPE offset                  all replay steps rotate at the     │
  │                               record-step's position → wrong     │
  │                               positional encoding                │
  └─────────────────────────────────────────────────────────────────┘
```

And three invariants that the record step has to establish:

```
  Invariant                                          Enforced by
  ───────────                                        ───────────
  K/V cache buffer addresses stay stable across      RotatingKVCache
  record → replay                                    preallocateFull
                                                     (icbDynamicOffset
                                                     path)

  Record-step primitive outputs stay at the same     PinSession: every
  MTLBuffer address across replays                   set_data lands in
                                                     a pinned pool

  Gather, RoPE, SDPA per-step scalars (T_k, RoPE     PersistentAb
  offset, input indices ptr) live in a buffer        handles +
  whose address is stable across replays             registries on the
                                                     Swift side
```

Without any one of these, the ICB's recorded pointers dangle on
replay. All three landed incrementally during the session-1 +
session-2 sweeps.

## 7. Files to know

```
  mlx/backend/metal/icb.h                        Segment, Command,
                                                 TagLocation structs
  mlx/backend/metal/icb.cpp                      record / finalize /
                                                 replay / replay_with_
                                                 overrides
  mlx/backend/metal/device.cpp                   CommandEncoder
                                                 begin/end_icb_
                                                 recording, steer,
                                                 maybeInsertBarrier
  mlx/backend/metal/persistent_ab.{h,cpp}        PersistentAb
                                                 (caller-owned AB)
  mlx-c/mlx/c/metal.{h,cpp}                      C API for all the
                                                 above + persistent-
                                                 AB factories
  Source/MLX/IndirectCommandBuffer.swift         Swift wrappers
                                                 + recordWithBindings
  Source/MLX/PersistentAb.swift                  Handle classes +
                                                 registries
  Source/MLX/PinSession.swift                    PinSession wrapper

  Libraries/MLXLMCommon/Evaluate.swift           TokenIterator.icbStep
                                                 (orchestrator)
  Libraries/MLXLMCommon/KVCache.swift            icbStepState hooks
  Libraries/MLXLLM/Models/GPTOSS.swift           AttentionBlock
                                                 uses RoPE + SDPA
                                                 persistent handles
```
