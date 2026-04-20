# Persistent-AB refactor — unblock decode-loop pipeline overlap

**Date:** 2026-04-20
**Status:** proposal, not implemented
**Prerequisite reading:** [icb-decode-loop-architecture.md](icb-decode-loop-architecture.md) §4 (cost model) and the "Root cause of the decode-loop gap" block in [icb-ab-bench-2026-04-19.md](icb-ab-bench-2026-04-19.md)

## The problem in one sentence

`Stream.synchronize` after every ICB replay adds ~20 ms/step of
pure CPU wait that pins decode-loop throughput to serial-GPU
speed, while plain AB + PersistentAB pipelines CPU+GPU and hits
~47 tok/s on the same kernels.

## Why the sync can't just be deleted

The three persistent-AB handles — `PersistentSdpaAbHandle`,
`PersistentRopeFreqsAbHandle`, `PersistentGatherFrontAbHandle` —
each back onto a **single** shared-storage `MTLBuffer` whose
contents are mutated from CPU every decode step:

```
  CPU step N:   setIndicesPtrOnAll(tokenN_buf)
                ↓ memcpy into gather AB slot 1
                (overwrites the pointer GPU is still reading
                 from step N-1's replay)
  CPU step N:   updateNPerLayer(tkN)
                ↓ memcpy into 24 × SDPA AB slot 12
                (same race on each layer's N value)
  CPU step N:   setOffsetOnAll(ropeOffsetN)
                ↓ memcpy into 48 × RoPE AB slot 2
                (same race on each layer's offset ptr)
```

Drop the sync and the CPU writes overtake the GPU reads →
"WeWeWe…" loop (the per-step token gets clobbered mid-gather).

## Recommended fix — option 1: override-bound fresh buffers

For every per-step mutable value, migrate from "scalar/ptr
baked inside the AB, memcpy'd each step" → **"BufferPtr slot
pointing at a fresh `MLXArray` each step, bound via the existing
override mechanism."**

The K/V `startArr` path already works this way and is proven
race-free — each step allocates a fresh `MLXArray([writeIdx])`,
registers it through the `overrides: [BindingName: MLXArray]`
dict, and `replay_with_overrides` calls `setKernelBuffer` on the
recorded ICB command + `useResource` on the encoder. Metal's
command-buffer completion handler holds the buffer alive until
the GPU is done with it. The orchestrator doesn't need to sync.

Net changes per handle:

| Handle | Today | Target |
|---|---|---|
| `PersistentSdpaAbHandle` | `N` is `Scalar32` slot 12 inside the AB. Orchestrator memcpys per step. | Slot 12 becomes `BufferPtr`. Orchestrator allocates `MLXArray([currentTk])` each step, binds via overrides. |
| `PersistentRopeFreqsAbHandle` | `offset` is `BufferPtr` slot 2. Orchestrator rewrites the ptr via `setBufferPtr` memcpy each step. | Same BufferPtr slot, but the new buffer is routed through `overrides: [icbRopeOffsetBinding(layer:)]` so `override_extra_buffers → useResource` keeps it alive. |
| `PersistentGatherFrontAbHandle` | `indices` is `BufferPtr` slot 1. Same setBufferPtr pattern. | Same migration — overrides-bound instead of setBufferPtr. |

The three AB MTLBuffers themselves can keep their stable
addresses — they're immutable after record time under this
scheme, so they need no rotation and no lifetime management
beyond the existing handle retention.

## What changes in each layer

### mlx C++ (`backend/metal/scaled_dot_product_attention.cpp`)

SDPA AB slot layout changes from 18 slots to 18 slots of slightly
different kinds:

```
  old layout (Scalar32 N)             new layout (BufferPtr N)
  ───────────────────────             ────────────────────────
  slot 10  Float32  scale             slot 10  Float32       scale
  slot 11  Scalar32 gqa_factor        slot 11  Scalar32      gqa_factor
  slot 12  Scalar32 N         ─────→  slot 12  BufferPtrOffset n_ptr
  slot 13  Scalar32 blocks            slot 13  Scalar32      blocks
```

Kernel-side: update `SdpaUnifiedArgs` struct in
`kernels/sdpa_unified.h` and the kernel to dereference `*n_ptr`
instead of reading the inline scalar. One extra memory load per
thread — negligible on Apple Silicon's memory bandwidth.

### mlx-c (`mlx/c/metal.cpp`)

`mlx_metal_persistent_ab_new_sdpa` layout declaration picks up
the same slot-kind swap. Function signature unchanged.

Also add one-shot `mlx_metal_icb_register_persistent_ab_slot_binding`
that lets the caller pre-declare "slot K of persistent AB `ab` is
a per-step BufferPtr that the orchestrator will override via
name_id `X`." This is the bridge between the record-time
`tag_ab_binding` registration and the replay-time `override`
lookup — conceptually the same as K/V starts but scoped to a
specific AB slot.

### mlx-swift (`Source/MLX/PersistentAb.swift`)

`PersistentSdpaAbHandle`:

```swift
// remove:
public static func updateNPerLayer(_ ns: [UInt32])

// add:
public static func registerNBinding(_ bindings: [BindingName])
    // one-shot: tells the orchestrator which N-buffer binding
    // name maps to this handle's slot 12
```

Same shape for `PersistentRopeFreqsAbHandle.offset` and
`PersistentGatherFrontAbHandle.indices` — rename the
setter-side APIs so they register a binding name instead of
mutating contents.

### mlx-swift-lm (`Libraries/MLXLMCommon/Evaluate.swift`)

`icbStep` replay branch becomes (diff, not full file):

```swift
-        // 24 mutations on shared-storage memory — blocks on sync
-        PersistentSdpaAbHandle.updateNPerLayer(perLayerTk)
+        // per-layer T_k via fresh 4-byte buffers, bound via overrides
+        for (i, tk) in perLayerTk.enumerated() {
+            let n = MLXArray([Int32(tk)])
+            eval(n)
+            overrides[Self.icbSdpaNBinding(layer: i)] = n
+        }

-        let ropeOffsetArr = MLXArray([ropePosition])
-        eval(ropeOffsetArr)
-        PersistentRopeAbHandle.setOffsetOnAll(ropeOffsetArr)
-        PersistentRopeFreqsAbHandle.setOffsetOnAll(ropeOffsetArr)
+        let ropeOffsetArr = MLXArray([ropePosition])
+        eval(ropeOffsetArr)
+        overrides[Self.icbRopeOffsetBinding] = ropeOffsetArr
+        // (if per-layer offsets ever differ, one binding per layer)

-        let gatherInput = previous[text: .newAxis].tokens
-        eval(gatherInput)
-        PersistentGatherFrontAbHandle.setIndicesPtrOnAll(gatherInput)
+        let gatherInput = previous[text: .newAxis].tokens
+        eval(gatherInput)
+        overrides[Self.icbGatherIndicesBinding] = gatherInput

         recordedIcb!.replay(overrides: overrides)
-        Stream.defaultStream(.gpu).synchronize()
+        // sync removed — all per-step state is now override-bound,
+        // so Metal's completion handler owns the retention and the
+        // next step's CPU work overlaps with this step's GPU work
```

Plus one-shot registration at record time (inside the
`recordWithBindings` scope) so the ICB's slot 12 bindings are
tagged with `icbSdpaNBinding(layer:)` etc. That's where
`set_buffer_ptr(12, …)` inside each AB becomes a tagged
binding instead of a hard-coded value.

## Expected outcome

Once the sync is gone and per-step work overlaps with GPU:

```
   Plain AB + PersistentAB today:    ~47 tok/s  (21 ms GPU, 2.6 ms CPU, overlapped)
   ICB decode-loop today:             ~42 tok/s  (21 ms GPU, 1.4 ms CPU, serialized)
   ICB decode-loop after refactor:   ~48–50 tok/s (same GPU, same CPU, overlapped)
```

Back-of-envelope: replay_submit is already 1.4 ms (less than
plain's 2.6 ms forward), so if the overlap works the ICB path
matches plain on CPU side and the GPU is the ceiling either way.
The win from ICB here isn't faster — it's that **1.4 ms/step of
CPU is available for other work** (logits post-processing, batch
coordination, speculative decoding) compared to 2.6 ms in the
plain path. That's the only real motivation left to keep the
infrastructure on this workload.

## Risk + scope

| Risk | Mitigation |
|---|---|
| Refactoring a kernel struct (`SdpaUnifiedArgs` slot 12 kind change) breaks the AB packing | AOT-regenerate kernel binary; covered by existing ICB correctness tests which already diff AB layouts |
| Fresh `MLXArray([n])` per step per layer = 24 extra allocs/step on GPT-OSS | Negligible — already allocate 2 startArrays/step via the same pattern; alloc pool reuses 4-byte int32 buffers |
| Override path's `setKernelBuffer` only hits recorded command slots; an AB-internal slot isn't a command slot | This is the real technical crux — see next section |

## The technical crux

The override mechanism today rewrites `icmd->setKernelBuffer(…,
slot)` on indirect compute commands. It doesn't know how to reach
*inside* an AB and mutate slot 12 of an AB's packed layout.

Two ways around this:

1. **Lift the N / offset / indices values out of the AB entirely**
   and bind them as additional kernel arguments at slots 1, 2, 3
   (the AB moves to slot 0, the three per-step values get their
   own command-slot bindings). Kernels take two extra bufferptr
   args; `SdpaUnifiedArgs` shrinks to its static members.
   Orchestrator overrides land on regular command slots — no new
   plumbing needed.

2. **Add a "mutate AB slot via command-side fresh buffer" override
   primitive** that walks the recorded commands' AB bindings,
   finds the AB, and rewrites a specific slot's BufferPtr. More
   plumbing, keeps the AB layout flat.

Option 1 is cleaner and reuses existing infrastructure. It's the
recommended path.

## Suggested sequence for the next session

1. Prototype on SDPA `N` alone — smallest change, biggest impact
   (24 per-layer N writes per step is the largest share of the
   three handles). Verify numerically with greedy parity vs the
   plain path. Measure tok/s.
2. If (1) lands clean: migrate RoPE `offset` and gather `indices`
   the same way. Drop the sync.
3. Benchmark against session-3 numbers; update
   [icb-ab-bench-2026-04-19.md](icb-ab-bench-2026-04-19.md). If
   throughput matches or beats plain AB+PersistentAB the
   decode-loop becomes a viable alternative.
4. If throughput still doesn't beat plain, at least we've
   recovered the CPU headroom (1.4 ms vs 2.6 ms per step), which
   is the only remaining motivation.

## Open questions

* Is there any other shared-storage state mutated per step that
  we haven't listed? The Swift-side inventory in this doc covers
  the registry-based handles. Primitive-side we should audit
  `RMSNorm::eval_gpu` for any set_scalar calls on a persistent
  handle's buffer — those would race the same way but aren't
  visible from the Swift layer.
* The post-fix throughput estimate assumes GPU work is the same.
  Under perfect CPU/GPU overlap, any new CPU cost in the per-step
  overrides (building 24 `MLXArray([tk])` values vs one memcpy)
  could bottleneck us earlier than the GPU ceiling. Benchmark
  needed.
* If we migrate away from persistent-AB shared-storage mutation,
  does the "Persistent AB" abstraction even earn its name any
  more? It becomes "caller-owned stable-address AB" which could
  be simplified further — possibly a follow-up after (1).
