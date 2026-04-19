# Persistent-AB Pilot — Design Doc — 2026-04-18

Branch: `ek/persistent-ab-pilot` (cut from `ek/metal-icb-prototype` on mlx / mlx-swift / mlx-swift-lm).

## Why this doc exists

I've been thrashing on the design in-flight. Getting it on paper first because the implementation is ~4 days across 4 repos and the wrong choice at the foundation will cascade. Review this before I commit to code.

## Foundational context — what the terms mean

Written out because the rest of the doc assumes these and I'd rather state them than have them implicit.

### A "decode step"

One iteration of the generation loop — producing exactly one output token. At GPT-OSS-20B's measured 47 tok/s on M1 Max, each step is ~21 ms. Inside one step, the model does a full forward pass: 24 transformer layers, each running RMSNorm → SDPA → RMSNorm → MLP with residuals. That's ~1500 GPU dispatches per step total (E1 dispatch-audit number from the adoption plan).

### Buffers vs scalars — what the CPU passes to a kernel

When the CPU launches a Metal compute kernel, it tells the kernel two kinds of things:

- **Buffers** — pointers to device memory. These hold tensor data: model weights, layer activations, KV cache entries, the output slot. On Apple Silicon, "device memory" is the same RAM the CPU sees (unified memory), so these are not copies — they're shared addresses.
- **Scalars** — small inline constants the kernel needs in order to *interpret* the buffers. Things like `axis_size` (how wide is this tensor), `scale` (softmax scale factor), `k_seq_stride` (how many bytes apart are adjacent K-sequence entries), or `T_k` (how many K entries are valid right now).

A kernel signature in Metal looks like:
```metal
[[kernel]] void rms_norm(
    const device float* x [[buffer(0)]],      // buffer
    const device float* w [[buffer(1)]],      // buffer
    device float* out [[buffer(2)]],          // buffer
    const constant uint& axis_size [[buffer(3)]],  // scalar
    const constant float& eps [[buffer(4)]],       // scalar
    ...);
```

On the CPU side, binding looks like:
```cpp
encoder.set_buffer(x_buf, 0);          // binds buffer at slot 0
encoder.set_buffer(w_buf, 1);          // slot 1
encoder.set_buffer(out_buf, 2);        // slot 2
encoder.set_bytes(&axis_size, 4, 3);   // inlines 4 bytes at slot 3
encoder.set_bytes(&eps, 4, 4);         // slot 4
encoder.dispatch_threadgroups(...);
```

The "AB migration" work is: instead of calling `set_buffer` and `set_bytes` six separate times per kernel, pack all six values into one tiny struct in device memory and bind that struct once. That's the AB.

### What `T_k` is specifically

`T_k` = the number of valid K (key) entries in the KV cache at the moment a given SDPA call fires.

In a decode run with a 101-token prompt:
- Step 1 (first generated token): the prompt is already in the cache; `T_k = 101`. SDPA's inner loop iterates K[0..100].
- Step 2: last step's generated token was appended; `T_k = 102`.
- Step N: `T_k = 100 + N`.

The SDPA kernel has a loop `for (i = 0; i < N; ...)` where `N` is `T_k`. If the kernel has T_k=101 baked into its encoded state but actually wants to attend to 102 positions, it reads only the first 101 — silently producing wrong output. This is the exact class of bug the adoption plan measured at 28% K/V drop after 400 decode steps.

Other SDPA scalars: `gqa_factor`, `k_head_stride`, `k_seq_stride`, `v_head_stride`, `v_seq_stride`, `scale`, mask-shape strides, `num_q_heads`. Most are constant across a run. **T_k is the main one that grows per step.** K/V strides can also shift if the cache buffer reallocates during growth.

### Where weights live

At model load, each weight tensor (`down_proj.weight`, `attention.q_proj.weight`, etc.) becomes an `MLXArray` backed by its own individually-allocated `MTL::Buffer`. These are shared-storage (unified memory), populated by mmap'ing / reading from the model file.

They are **not currently packed into a single `MTL::Heap`.** Each weight has its own independent GPU address. The address is stable for the lifetime of the run — weights don't move after load. This is important for the persistent-AB design because it means we can write a weight's `gpuAddress` into an AB once at handle init, and never touch it again.

A `WeightHeap` — one big `MTL::Heap` containing all weights, bound per-encoder with a single `useHeap:` — was proposed in the adoption plan but hasn't been built. It would replace the current per-segment `useResource:` iteration with a single call. Not blocking for this pilot; orthogonal optimization.

### Where the KV cache lives

Per-layer `KVCache` instance (36 for GPT-OSS-20B, one per transformer block). Two implementations:

- **`KVCacheSimple`** (default) — starts small, grows by appending along axis 2. When the current allocation fills, reallocates a bigger buffer and copies. *Address can change on growth.*
- **`RotatingKVCache`** — pre-allocates to a max length. Writes rotate in-place (sliding window). *Address is pinned from the start.*

`cache.offset` (a scalar) tracks how many tokens have been written. After step N, `cache.offset = 100 + N`. This value IS `T_k` for SDPA on that layer.

For ICB replay to work, **the cache's K and V buffer addresses need to be stable across the replay range.** Options:
- Use `RotatingKVCache` and size it to a ceiling (simplest).
- Fix `KVCacheSimple` to pre-allocate upfront when a max length is known (cache-passthrough work noted in my user memory).
- Re-record the ICB when the cache grows (defeats replay's purpose).

Pilot targets the first option — pin the cache to a known max size upfront.

### CPU → GPU reference passing, today

Every kernel call from the CPU side passes addresses to the GPU one of two ways:

- **Legacy path**: `encoder.set_buffer(x.mtl_buffer, slot_1)` — Metal encoder bakes the buffer's gpuAddress into the command stream at the slot.
- **AB path** (Phase 2): `ab.set_buffer_ptr(slot_0, x.mtl_buffer, x.offset)` — writes x's gpuAddress + offset into the AB's shared-storage memory at a known byte position, then `encoder.set_buffer(ab.mtl_buffer, 0)` binds the AB itself.

Either way, the ultimate thing the GPU reads is a pointer. The CPU's job is to put that pointer where the kernel expects it. The difference is whether it goes through Metal's encoder API (one call per argument) or through a tiny packed struct in unified memory (one call total).

### Is this "taking maximal advantage of unified memory"?

Mostly yes:
- Shared-storage buffers, zero CPU↔GPU copies ✓
- GPU reads weights / activations / KV cache at DRAM speed (~400 GB/s) ✓
- AB migration replaced many small `set_bytes` calls with one bind + a shared-storage struct ✓

Gaps:
- Weights aren't heap-packed (minor; optimization pending as `WeightHeap`).
- `KVCacheSimple` can reallocate during growth (addresses not guaranteed stable). Rotating cache pins them; pilot will prefer it.
- **The bottleneck we're fighting isn't memory read speed.** It's the CPU-side encoder cost: ~1500 dispatches per step × ~5–10 µs per dispatch for `setComputePipelineState` + `setBuffer` + `setBytes` + `dispatchThreadgroups`. ICB replay skips that encoder work — recorded dispatches replay with near-zero CPU cost.

The design concern in the rest of this doc is: once the CPU stops re-emitting dispatches (ICB replay), *someone* still has to make sure the scalars and buffer pointers the kernels read are the current step's values. That "someone" has to be Swift writing into the AB between replays, or the ICB replay mechanism has to know which binds to update.

## Problem statement

Phase 2 unified SDPA + AB-migrated the 9 hot-path primitives. SDPA's setBytes arena is provably empty (diagnostic `6f097aa6` gate passes). But ICB replay in a real decode loop is still architecturally broken:

- **Current AB lifecycle is per-call.** Each `rms_norm()` / `sdpa()` / etc. call allocates a fresh `ArgumentBuffer` via the pool, populates its slots with THIS call's buffer addresses + scalars, binds at slot 0, dispatches, and releases the AB back to the pool after the command buffer completes.
- **ICB replay captures the bind but not the contents.** When the ICB recorder records `setBuffer(ab_step2, 0)`, it retains the MTLBuffer pointer (lifetime OK) but the AB's contents reflect step 2's state. A replay at step 3+ dispatches the recorded command. The kernel reads step 2's buffer addresses + scalars from the AB — even though step 3's actual data lives elsewhere.
- **Net effect**: same class of correctness bug Option C was supposed to close, relocated from `setBytes` (encoder-internal) to AB contents (device-memory).

## The core design question

Where does AB state live across decode steps?

| Approach | Who owns the AB | Who updates contents | Cost |
|---|---|---|---|
| **A — Caller-owned handle** | mlx-swift-lm model layer (per call site) | Swift writes directly to AB buffer pre-step | Moderate: typed Swift API per primitive |
| B — Recorder-managed ABs | ICB recorder tracks AB binds as taggable | Swift supplies fresh AB each step via override | High: extends tag_binding to raw buffers; still per-step alloc |
| C — Thread-local ABs keyed by call site | mlx C++ side cache | Primitive reuses same AB; contents update per call normally | Moderate: call-site identity is tricky in mlx |
| D — Fresh per call + override on replay | per-call (current) | Swift does fresh forward each step, passes ABs via replay | High: defeats the point of replay (still does CPU encoding work) |

I'm choosing **Option A**: caller-owned handle. Rationale:

1. **Closest to Apple's AB-with-ICB guidance.** An AB is a regular device buffer whose contents the CPU updates between dispatches. Owning it at the caller level matches how Apple describes the pattern.
2. **Makes the persistence explicit at the right layer.** The model layer (Attention module, MLP module) is where the lifetime should live — per-layer, per-primitive-call-site. The decode loop creates a handle, reuses it every step.
3. **No new recorder infrastructure.** tag_binding/replay_with_overrides stay as they are; we don't tag AB binds. The AB buffer pointer is stable; ICB records it and replays it. Contents update happens outside the ICB replay call, from Swift, directly into shared-storage memory.
4. **Scales primitive-by-primitive.** Each primitive gets its own typed handle. Incremental adoption — we start with RMSNorm, extend to the other 8 if the pattern pans out.

## Scope — pilot vs full

### Pilot: RMSNorm only

Why RMSNorm:
- Smallest AB surface (3 buffer pointers + 1 float + 2 scalar32 = 6 slots).
- Called 2× per layer × 24 layers = 48 times per decode step in GPT-OSS-20B.
- Simplest semantics — axis_size and w_stride are STATIC across decode (hidden dim doesn't change). Only the buffer pointers need per-step updating.
- Regression sweep doesn't exist for RMSNorm specifically, but existing mlx tests cover correctness.

**Pilot exit criteria:**
1. mlx: RMSNorm can accept an externally-owned `ArgumentBuffer` and reuse it across dispatches with updated contents.
2. mlx-c: C API for creating + updating + destroying a persistent AB handle for RMSNorm.
3. mlx-swift: Swift wrapper `PersistentRmsAbHandle` with typed slot-update methods.
4. mlx-swift-lm: one transformer layer's RMSNorm wrapper uses the handle. Decode still produces byte-identical outputs to the current Phase 2 code (correctness gate).
5. Benchmark: decode tok/s with persistent-AB RMSNorm ≥ current Phase 2 decode tok/s (not a regression; may be neutral if the RMSNorm AB allocation savings are small).

### Full Option A: remaining 8 primitives

After pilot validates the pattern, extend to:
- RoPE
- SDPA (unified) — big one; T_k scalar updates per step
- affine_qmv (Linear)
- affine_gather_qmv (SwitchLinear / MoE gate)
- gather_front (Embedding)
- binary Add/Multiply (residuals + MoE combine)
- unary (SiLU/Exp/Abs/Sigmoid)
- Compiled JIT fused-op AB

Plus the ICB record/replay wiring around the decode-step boundary in mlx-swift-lm.

**Full Option A exit criteria:**
1. ICB recorded at step 2. Subsequent steps 3..N replay the ICB + update all persistent ABs in-place for current step state.
2. Generated tokens byte-identical to current Phase 2 decode path (correctness gate).
3. GPT-OSS-20B decode tok/s ≥ 55 tok/s (the alpha-mean of 47.9 × 1.15 as a target; further depends on how much of the 1.40× encoding microbench translates).
4. Gemma 4 E2B decode tok/s ≥ prior baseline.

## The layering — concrete surfaces

### Layer 0 — mlx C++

New: `mlx/backend/metal/persistent_ab.h` (exported) — a thin wrapper class over `ArgumentBuffer` with an explicit "this is externally owned" lifecycle marker (no add_temporary_object on dispatch).

```cpp
namespace mlx::core::metal {

class MLX_API PersistentAb {
 public:
  // Construct with a layout. The underlying MTL::Buffer is pooled-
  // backed (same as transient ArgumentBuffers) but the lifetime is
  // tied to `this` — released to pool on destruction, not after
  // the next command buffer.
  PersistentAb(Device& d, std::vector<ArgumentBuffer::Slot> layout);

  // Typed setters, same as ArgumentBuffer. Expose read access to
  // the slot layout for callers that need to validate offsets.
  void set_scalar32(int slot, uint32_t value);
  void set_scalar64(int slot, uint64_t value);
  void set_float32(int slot, float value);
  void set_buffer_ptr(int slot, const MTL::Buffer* buf, int64_t offset);

  // Raw access for integration with the ICB recorder and compute
  // encoder.
  MTL::Buffer* mtl_buffer() const;
  const std::vector<ArgumentBuffer::Slot>& layout() const;

 private:
  ArgumentBuffer ab_;
};

} // namespace mlx::core::metal
```

New: `fast::rms_norm` accepts an optional `PersistentAb*` via a new overload:

```cpp
// Existing signature — unchanged
array rms_norm(const array& x, const array& w, float eps, StreamOrDevice s = {});

// NEW overload — persistent AB. Caller owns.
array rms_norm(
    const array& x,
    const array& w,
    float eps,
    std::shared_ptr<metal::PersistentAb> ab_handle,  // NEW
    StreamOrDevice s = {});
```

The overload constructs a `RMSNorm` primitive with the handle baked in. `RMSNorm::eval_gpu` reads `ab_handle_`; if non-null, populates its slots + binds it instead of allocating a fresh one. No `add_temporary_object` call — the handle's lifetime is external.

Regression test: construct an `PersistentAb` with the right layout, call `rms_norm` twice with different inputs passing the same handle both times, verify both outputs match the reference path.

### Layer 1 — mlx-c

Expose the handle via a C-type + create/destroy/update APIs:

```c
typedef struct mlx_metal_persistent_ab_ mlx_metal_persistent_ab;

mlx_metal_persistent_ab mlx_metal_persistent_ab_new_rms(
    mlx_stream stream /* for device lookup */);
int mlx_metal_persistent_ab_free(mlx_metal_persistent_ab ab);
int mlx_metal_persistent_ab_set_buffer_ptr(
    mlx_metal_persistent_ab ab, int slot,
    mlx_array buffer_backing_array, int64_t offset);
int mlx_metal_persistent_ab_set_float32(
    mlx_metal_persistent_ab ab, int slot, float value);
int mlx_metal_persistent_ab_set_scalar32(
    mlx_metal_persistent_ab ab, int slot, uint32_t value);
```

Plus a variant of `mlx_fast_rms_norm` that accepts an `mlx_metal_persistent_ab`.

### Layer 2 — mlx-swift

Swift wrapper with slot enums:

```swift
public final class PersistentRmsAbHandle {
    // Slot layout (fixed by RMSNorm):
    //   0: BufferPtrOffset  x
    //   1: BufferPtrOffset  w
    //   2: BufferPtrOffset  out
    //   3: Float32          eps
    //   4: Scalar32         axis_size
    //   5: Scalar32         w_stride
    public enum Slot: Int {
        case x = 0, w = 1, out = 2, eps = 3, axisSize = 4, wStride = 5
    }

    init(stream: Stream)
    deinit

    public func setBufferPtr(slot: Slot, buffer: MLXArray, offset: Int = 0)
    public func setFloat32(slot: Slot, value: Float)
    public func setScalar32(slot: Slot, value: UInt32)
}

// Usage site in a fast wrapper
public func rmsNorm(
    _ x: MLXArray, _ w: MLXArray, eps: Float,
    ab handle: PersistentRmsAbHandle? = nil,  // NEW
    stream: StreamOrDevice = .default
) -> MLXArray
```

### Layer 3 — mlx-swift-lm

Update `RMSNorm` module (or whatever carries the fast::rms_norm calls) to own a `PersistentRmsAbHandle` lazy-initialized on first call:

```swift
private var _abHandle: PersistentRmsAbHandle?

public func callAsFunction(_ x: MLXArray) -> MLXArray {
    if _abHandle == nil {
        _abHandle = PersistentRmsAbHandle(stream: x.stream)
        _abHandle!.setScalar32(slot: .axisSize, value: UInt32(axisSize))
        _abHandle!.setScalar32(slot: .wStride, value: UInt32(wStride))
        _abHandle!.setFloat32(slot: .eps, value: eps)
    }
    return rmsNorm(x, weight, eps: eps, ab: _abHandle)
}
```

The handle persists across forward passes. Only x/out buffer pointers get refreshed per call (inside the mlx C++ eval_gpu, since they depend on the current inputs/outputs).

Wait — who updates x/out per call? If the Swift layer owns the handle, Swift has to write x's gpu address before each call. But Swift also has to call rmsNorm(), which triggers eval_gpu that also writes x's gpu address. Conflict or duplication.

**Resolution:** the mlx C++ `eval_gpu` always writes the buffer pointers (they're inputs to the primitive, the primitive is the authority on them). Swift only writes the scalars that are constant-across-steps (eps, axis_size, w_stride). Those are effectively one-time writes at handle construction.

Revised usage:

```swift
if _abHandle == nil {
    _abHandle = PersistentRmsAbHandle(stream: x.stream)
    // One-time constant writes — Swift never touches these again
    _abHandle!.setScalar32(slot: .axisSize, value: UInt32(axisSize))
    _abHandle!.setScalar32(slot: .wStride, value: UInt32(wStride))
    _abHandle!.setFloat32(slot: .eps, value: eps)
}
// mlx C++ side writes buffer ptrs for x/w/out per call
return rmsNorm(x, weight, eps: eps, ab: _abHandle)
```

Under ICB recording this still doesn't quite work — the recorded setBuffer(ab, 0) captures step 2's buffer ptrs because eval_gpu wrote them. Replay at step 3 would use step 2's ptrs.

**The real mechanism under ICB replay needs to be:** tag the x/w/out MLXArray inputs of `rmsNorm()` as overridable bindings. Then `replay(overrides:)` rebinds the buffer. But the AB still has step 2's pointers baked in — the `replay_with_overrides` doesn't reach inside the AB bytes.

So we need one of:
- Tag AB slot-0 bind as overridable with a custom-binding type "AB"; override at replay with a fresh AB that has step-3's ptrs
- Push the buffer-pointer writes into a GPU-visible place the kernel can re-read per dispatch

Option 1 is cleanest. Swift builds a fresh AB per step (cheap — pooled), passes it via replay overrides. The AB's constant scalars (eps, etc.) are copied from the template on construction.

At that point we don't really need a "persistent" AB at all — we just need the AB bind to be a tagged binding. The AB lifetime is per-step (allocated + released via pool).

Hmm. Let me revise.

## Revised design (v2) — tagged-AB-bind approach

Actually the correct observation: **persistent AB isn't what we need; what we need is overridable AB binds during ICB replay.**

The buffer pointers change per step because inputs/outputs change per step. The scalars change per step too (T_k for SDPA, not for RMSNorm). So the AB CONTENTS are fundamentally per-step. No amount of "persistence" fixes that for SDPA.

The real architecture is:
1. Each decode step's forward produces a fresh AB per primitive call (current Phase 2 behavior).
2. During ICB recording, when the compute encoder sees `setBuffer(ab_mtl, 0)`, the recorder tags this bind with a name derived from the primitive + call-site (implicitly or explicitly).
3. Swift provides per-step overrides: `replay(ab_overrides: [ab_name: current_step_ab_mtl])`.

This is actually Option B from the earlier enumeration, reconsidered.

**Required changes:**
- mlx C++ CommandEncoder: when binding an AB (slot 0 with a small shared-storage buffer matching a known pattern), automatically tag it. Or — expose a tag_ab_binding API the primitive calls.
- Recorder: store these tags with a unique name (sequential counter is fine).
- Swift: `replay(abOverrides: [Name: MTLBuffer])` overrides AB binds at replay.

Under this design, each primitive call still allocates a fresh AB (cheap pool hit). Its contents have THIS call's state. The bind goes through the recorder with a tag. At replay, Swift provides the current-step AB buffer.

Cost per step: 
- Allocate + populate N fresh ABs (cheap — pool + small memcpy)  
- Replay with those ABs as overrides

The CPU encoding savings come from NOT re-emitting all the setComputePipelineState + dispatchThreadgroups + setBuffer calls — just the AB contents-update + the replay trigger.

## Which design is right — the actual call

The v2 "tagged AB bind" is architecturally simpler AND more general:
- No per-primitive Swift handle classes
- No separate persistent-AB C++ class
- Just a tagging mechanism in the recorder + an override API

Downside: Swift still has to do the "run the forward pass to generate fresh ABs each step" work. Which... wait, that's a lot of work. It means running the full model graph in mlx-swift-lm each step, which does all the op dispatching at the Swift layer.

Unless: the framework exposes a "just build the ABs without dispatching" path. Which is essentially the same as having persistent ABs with Swift-side content update.

---

**OK I think I've been missing the real tension.** Let me name it:

- The CPU encoding cost we want to eliminate is **emitting dispatches + setBuffer calls** from mlx → Metal.
- ICB replay eliminates THAT cost — but only if the replay is a near-pure "GPU kick" with no Swift-side graph eval.
- For the replay to produce correct output, the AB contents must reflect THIS step's state.
- For the AB contents to be correct, SOMEONE has to write them each step.
- If Swift writes them, Swift needs to know each AB's exact layout + identity. Per-primitive handles.
- If mlx C++ writes them (via re-running the op graph), we're back to doing the work ICB was supposed to skip.

So **there's no way around per-primitive Swift handles** if we want the full ICB win.

(There's an alternative: write a fat single-buffer for ALL primitives, keyed by dispatch order. Swift updates slot ranges by dispatch index. But that just hides the per-primitive schema behind an ordering convention — equivalent complexity.)

## Final design (v3) — committing to v1 with clarity

Back to v1. Per-primitive handles, Swift owns them, Swift writes contents per step.

For RMSNorm, the handle contents break down as:
- Static (write once at handle init): `eps`, `axis_size`, `w_stride`
- Per-step (write per decode step from Swift): `x.addr/offset`, `out.addr/offset`
- Shouldn't change in decode: `w.addr/offset` (weights are fixed; write once)

For SDPA, everything is per-step: Q/K/V/out pointers + T_k + mask strides + scale (sometimes).

So Swift's per-step work:
- For each layer, for each AB-using primitive: update buffer pointers (`x.addr`, `out.addr`, etc.) + shape scalars that vary.
- The shape scalars that vary per step are relatively few (mainly T_k for SDPA).
- The buffer pointer updates are the bulk of the work.

Swift knows which MLXArray is "x" for layer N's RMSNorm — it's a local variable in the forward pass. So Swift can update `rmsHandle.setBufferPtr(.x, x)` naturally.

BUT: if the Swift forward pass is still running and computing x, we haven't actually skipped the forward work. The Swift forward pass dispatches ops that run on GPU...

**Unless**: Swift's forward pass runs in "ICB-replay mode" where ops are no-ops. Just traces structure, updates AB handles, triggers replay at the end.

That mode needs to exist. It's essentially what mlx-swift's `IndirectCommandBuffer.replay(overrides:)` does — replay without re-running the graph.

But the override dict is the ENTIRE mechanism. If Swift's forward pass is NOT running ops (just building overrides), the overrides dict can only contain MLXArray bindings (inputs/outputs tagged during recording). That's what we already have.

Which means: **the current `replay_with_overrides` API already handles input/output buffer overrides**. The gap is for AB-packed contents — the scalars and pointers that live INSIDE the AB bytes.

If AB contains only shape scalars (no buffer pointers), the buffer pointers go back to regular setBuffer binds and are overridable via the existing API. And the shape scalars in the AB are static-ish (can be written once at handle init).

For RMSNorm where all scalars are static, this works cleanly.
For SDPA where T_k changes every step, this doesn't — the T_k would be baked in the AB.

So for SDPA specifically, we need the Swift-side AB content-update API to rewrite T_k per step.

---

## Truly final design (v4) — lifecycle by primitive class

Different primitives, different answers:

### Class 1 — Static-scalar primitives (RMSNorm, maybe RoPE)
Shape scalars are constant across decode (axis_size, w_stride, theta). Buffer pointers change per step.

**Design**: AB contains only the static scalars + pointers. Buffer pointers are ALSO in the AB (for one-bind efficiency). The AB is effectively recomputed per step (pool-backed, cheap) and tagged as an overridable binding at record time. Swift builds fresh ABs per step, passes via replay(ab_overrides:).

This is v2. Cost: Swift must know how to build the AB per step — implies Swift exposure to layout.

### Class 2 — Dynamic-scalar primitives (SDPA)
Shape scalars change per step (T_k, cache offset). Buffer pointers also change.

**Design**: Persistent AB owned by Swift (or at Swift's layer). Swift writes T_k and pointers per step directly into the AB's shared-storage MTLBuffer. The ICB records the bind once; replay uses the stable AB with updated contents.

This is v1. Cost: typed Swift API per primitive.

### Common infrastructure
Both classes need: Swift-accessible AB with typed slot writes. So build that first.

Then:
- Class 1 primitives use the common infra but let Swift rebuild + override per step.
- Class 2 primitives use the common infra with persistent handles.

## Concrete pilot plan — revised

### Step 1 — common infra (smallest committable unit)
In mlx C++: add `PersistentAb` class (wraps `ArgumentBuffer`, no auto-release). Export via mlx-c + mlx-swift.

Test: Swift can create a PersistentAb, write + read its contents, destroy it. No primitive integration yet.

### Step 2 — RMSNorm integration
Wire `fast::rms_norm` to accept optional `PersistentAb*`. When provided, use it + have mlx C++ write buffer pointers per call.

Test: C++ side test that reuses AB across two calls with different x.

### Step 3 — Swift RMSNorm wrapper
Swift `PersistentRmsAbHandle` class. One transformer layer's RMSNorm uses it.

Test: end-to-end decode produces byte-identical output to Phase 2 path. Benchmark: decode tok/s doesn't regress.

### Step 4 — ICB record/replay around RMSNorm
Record a decode step. Replay N times with persistent handles. Verify correctness + measure.

This is the first real signal on whether the pattern delivers tok/s.

### Step 5 — Extend to remaining primitives
One at a time. Most are Class 1 (RoPE, binary, unary, gather_front, compiled JIT). SDPA, affine_qmv, affine_gather_qmv are Class 2 (dynamic scalars).

### Total effort
- Step 1: 0.5 day
- Step 2: 0.5 day
- Step 3: 1 day
- Step 4: 1 day
- Step 5: 2-3 days
- Benchmarks + regressions: 1 day
- **Total: ~6-7 days**

## Questions to resolve before starting

1. **Naming**: `PersistentAb` vs `ManagedArgumentBuffer` vs `PersistentArgumentBuffer`? I'll default to `PersistentArgumentBuffer` for clarity.
2. **MLX-c handle types**: one generic `mlx_metal_persistent_ab` or one per primitive (`mlx_metal_rms_ab_handle` etc.)? Generic is simpler; per-primitive is more type-safe.
3. **Swift handle types**: Similarly — one generic + enum slots, or typed per primitive? Typed per primitive trades code size for safety.
4. **What happens if the handle's layout doesn't match the primitive's expected layout?** Runtime error or compile-time? I'd vote runtime check with clear message.
5. **Tests on both the C++ and Swift side** or only Swift? C++ tests are faster to write and diagnose.

## Recommendation

Ship Step 1 + Step 2 + basic test this week. Decide on Step 3 onwards based on feedback.

If the foundation feels right, continue. If not, reshape before Swift layer is committed.

Committing this doc to `ek/persistent-ab-pilot` on mlx-swift-lm so it travels with the work.
