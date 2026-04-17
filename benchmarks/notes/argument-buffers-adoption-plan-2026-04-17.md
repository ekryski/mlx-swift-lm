# Argument Buffers + ICB Decode Integration — Adoption Plan — 2026-04-17

Design spec for the next phase of Metal CPU-encoding optimization in
mlx. Grounded in the shape-sensitivity diagnostic captured earlier
today ([`6f097aa6`](https://github.com/ekryski/mlx/commit/6f097aa6) in
mlx) which established that ICB replay alone cannot carry a growing
KV cache without a deeper architectural change.

**Prior context:**

- Strategy doc: [cpu-encoding-optimization-strategies-2026-04-17.md](cpu-encoding-optimization-strategies-2026-04-17.md)
- ICB prototype plan: [cpu-encoding-optimization-plan-04-17-2026.md](cpu-encoding-optimization-plan-04-17-2026.md)
- First encoding measurement: [icb-first-measurement-2026-04-17.md](icb-first-measurement-2026-04-17.md)
- Tok/s projection + baselines: [icb-tok-per-sec-projection-2026-04-17.md](icb-tok-per-sec-projection-2026-04-17.md)
- Working branch: `ek/metal-icb-prototype` on all four repos

---

## Why argument buffers — the data that forced the move

The ICB prototype established, across all four repos, that Metal
Indirect Command Buffers can be recorded once and replayed cheaply
(1.45x–1.74x encoding speedup measured on live models). The remaining
gap to real tok/s improvement is **shape-sensitivity of the recorded
setBytes payloads**.

### Diagnostic data (`6f097aa6`, tests/icb_real_primitive_tests.cpp:192)

`fast::scaled_dot_product_attention` recorded twice with identical Q
shape but different K/V sequence length:

| T_k | Commands | Segments | bytes_used | Arena contents |
|---:|:---:|:---:|:---:|:---|
| 1024 | 5 | **3** | 148 | baseline |
| 1025 | 5 | **4** | 148 | **differs byte-by-byte** |

Two independent problems:

1. **setBytes payload is T_k-dependent.** Shape scalars (batch, seq
   len, stride, valid mask bound) are written via
   `MTL::ComputeCommandEncoder::setBytes` during record, spill into
   the recorder's shared arena, and are frozen when `finalize()` runs.
   A replay at a different T_k would compute on the recorded T_k, not
   the current one.

2. **Segment topology is T_k-dependent.** SDPA selects a different
   kernel variant (or emits different intermediate allocations — the
   specific cause is out of scope here) at some threshold between
   T_k=1024 and T_k=1025, producing a different number of barrier
   boundaries. The ICB's segment count is baked at `finalize()`. Even
   a perfect "override all setBytes values" patch would not rescue
   this path.

### Why argument buffers fix it

The Metal programming guide positions Argument Buffers (AB) as the
designed-for-ICB way to pass dynamic kernel state:

> *Argument Buffers are almost always used alongside ICBs to manage
> the resources (textures, buffers) that those commands need. [...]
> If you plan to encode ICBs or write into Argument Buffers directly
> from a shader, your device must support Tier 2 Argument Buffers.*

Three properties of ABs map directly onto the problems above:

1. **Scalars live in device memory, not encoder-captured bytes.** The
   AB is a regular `MTL::Buffer`. Its contents at dispatch time are
   whatever the caller last wrote — the kernel reads T_k and mask
   bounds from the AB at GPU execution, not from the encoder's
   baked-in bytes.

2. **One AB-bind replaces many setBuffer/setBytes.** A decode-layer
   dispatch today issues ~5–30 individual binding calls. With AB,
   that collapses to one AB-pointer bind per dispatch.

3. **Tier-2 ABs can be written from shaders.** Per-step updates
   (T_k, cache offset, mask) can be computed in a tiny GPU dispatch
   rather than driven from the CPU, which keeps the update cost off
   the critical CPU encoding path.

The Metal guidance also co-recommends `MTL::Heap` + `useHeap:` for
read-only resources (model weights). Today, the per-encoder
`useResource:` bookkeeping iterates every bound buffer per segment;
replacing that with a single `useHeap` on the weight heap both
reduces the replay cost and simplifies ICB residency tracking.

With ABs + heaps in place, the ICB's recorded payload is reduced to:
pipeline state (stable), AB pointer (stable), heap residency (stable),
dispatch threadgroup count (stable for decode — T_q=1). The entire
shape-sensitivity class of problems disappears.

---

## Baseline to beat

Measured on M1 Max, 64 GB, macOS 15.7.4, `ek/metal-icb-prototype`,
1024-ctx summarization. Rows live in
[m1-max-64gb-2026-04-17.md](../m1-max-64gb-2026-04-17.md).

| Model | KV | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Gemma 4 E2B | none | 2888.2 | **101.8** |
| GPT-OSS 20B | none | 580.8 | **64.0** |

Per-step encoding cost (ICB microbenchmark, step-2+ decode only):

| Model | Live (µs/step) | ICB replay (µs/step) | Current ICB speedup |
|---|---:|---:|---:|
| Gemma 4 E2B | 8,573 | 6,779 | 1.27x |
| GPT-OSS 20B | 17,107 | 11,831 | 1.45x |

Attribution of the ~17 ms live step on GPT-OSS 20B (rough, from
dispatch audit + pipeline labels):

- setBytes + setBuffer + setComputePipelineState: **~40–55%**
- Barrier tracking + useResource overhead: **~10–15%**
- MTL command-buffer commit + ObjC bridge: **~5–10%**
- GPU execution time: **~25–35%**

The ICB-alone win (1.45x) captures most of the pipeline-state +
dispatch-threadgroup overhead but leaves the setBuffer / setBytes
portion largely untouched. AB targets that portion directly. Rough
theoretical ceiling on encoding-cost reduction (not tok/s — GPU
execution is still there): **3–5x** vs live.

---

## Architecture

### Layer 0: `ArgumentBuffer` — mlx C++

New file pair `mlx/backend/metal/argument_buffer.{h,cpp}`.

```cpp
namespace mlx::core::metal {

// A pre-sized `MTL::Buffer` holding a packed layout of kernel
// arguments. The layout is declared at construction; slots are
// typed and positional. The buffer lives in shared storage so the
// CPU side can write updates between replays without a GPU-side
// copy.
class ArgumentBuffer {
 public:
  struct Slot {
    enum class Kind { Scalar32, Scalar64, Float32, BufferPtr, BufferPtrOffset };
    Kind kind;
    size_t byte_offset;  // into the argument buffer contents
  };

  ArgumentBuffer(Device& d, std::vector<Slot> layout);

  // CPU-side setters. Aligned writes into the shared-storage buffer.
  void set_scalar32(int slot, uint32_t value);
  void set_scalar64(int slot, uint64_t value);
  void set_float32(int slot, float value);
  void set_buffer_ptr(int slot, const MTL::Buffer* buf, int64_t offset);

  MTL::Buffer* mtl_buffer() const { return buffer_.get(); }
  size_t size_bytes() const { return size_; }

 private:
  std::vector<Slot> layout_;
  NS::SharedPtr<MTL::Buffer> buffer_;
  size_t size_;
};

} // namespace mlx::core::metal
```

Residency: ABs are shared-storage `MTL::Buffer`s, so `useResource` is
sufficient. Alternatively, pool ABs for a graph into a single `MTLHeap`
and call `useHeap` once — deferred to a later phase.

### Layer 1: Weight heap — mlx C++

Pack a model's immutable weights into a single `MTL::Heap` at load
time. Each weight `mlx::core::array` retains a heap-backed `MTL::Buffer`
instead of an independent allocation.

```cpp
namespace mlx::core::metal {

class WeightHeap {
 public:
  // Size known up-front from total weight bytes + alignment padding.
  WeightHeap(Device& d, size_t bytes);

  // Allocate a sub-range for one weight tensor. Returns a buffer view
  // (MTL::Buffer*) into the heap. The heap retains ownership.
  MTL::Buffer* allocate(size_t bytes, size_t alignment);

  MTL::Heap* mtl_heap() const { return heap_.get(); }

 private:
  NS::SharedPtr<MTL::Heap> heap_;
  // ...
};

} // namespace mlx::core::metal
```

`CommandEncoder::replay_icb` / `replay_icb_with_overrides` grows a
new entry point:

```cpp
void CommandEncoder::declare_heap_residency(MTL::Heap* heap);
```

Called once per encoder-lifetime (or once per replay loop), translated
to `useHeap:` on the live encoder. When combined with heap-backed
weights, the per-segment `useResource` iteration collapses from dozens
of weight pointers to zero (the heap covers them all).

### Layer 2: per-primitive eval_gpu migration

Each primitive's `eval_gpu` is rewritten in three steps:

1. Allocate (or reuse) an `ArgumentBuffer` sized to the primitive's
   kernel argument layout.
2. Populate the AB from the current call's inputs/outputs.
3. Emit a single `set_buffer(ab.mtl_buffer(), 0)` and
   `dispatch_threadgroups(...)` — no more per-field setBytes, no more
   per-resource setBuffer.

Order of migration (by decode-path impact — first pass keeps
attention-free for corner cases, adds SDPA last):

| # | Primitive | Rationale |
|---|---|---|
| 1 | `RMSNorm` | Smallest surface; good pilot. Touches every layer. |
| 2 | `Linear` / matmul | Largest setBuffer count per call. Q/K/V/O proj + MLP up/down. |
| 3 | `Embedding` | Called once per step; simple layout. |
| 4 | `RoPE` | Shape scalars are the entire migration target. |
| 5 | `SwitchLinear` / MoE expert gather | MoE-specific but high payoff on GPT-OSS. |
| 6 | `softmax`, `add`, `silu`, elementwise | Cleanup pass; tiny per-call. |
| 7 | **`scaled_dot_product_attention`** | **The load-bearing migration.** T_k → AB scalar; unify kernel selection so segment topology is stable. |

### Layer 3: ICB integration

Once enough primitives use ABs, the ICB recorder's setBytes arena
becomes nearly empty (AB pointer bindings + threadgroup memory slots
only). The recorder's complexity drops correspondingly.

The decode-loop integration pattern now works cleanly:

```swift
// Record once at step 1:
let icb = try IndirectCommandBuffer.recordWithBindings { tagger in
    tagger.tag(hidden,          as: "input")
    tagger.tag(abCurrentStep,   as: "step_args")
    hidden = layer(hidden, mask: mask, cache: cache)
    tagger.tag(hidden,          as: "output")
}

// Per subsequent step:
abCurrentStep.setScalar32(slot: .TK, value: cache.offset + 1)
abCurrentStep.setBufferPtr(slot: .cacheKeys, buffer: cache.keysMTL, offset: 0)
// ... etc
icb.replay(overrides: [
    "input": nextHidden,
    "output": outputSlot,
])
```

Note: `step_args` (the AB) does *not* need to be in the overrides dict
because its buffer pointer is stable — its *contents* changed, which
the GPU reads at replay time.

---

## Phased rollout

### Phase 1 — AB + heap infra, one pilot primitive (~1 week)

- `ArgumentBuffer` class in mlx C++ with unit tests.
- `WeightHeap` class in mlx C++ with unit tests.
- `CommandEncoder::declare_heap_residency` API.
- Migrate `RMSNorm::eval_gpu` to AB. Single-kernel change; easy to
  roll back.
- Benchmark delta: isolated RMSNorm microbenchmark before/after,
  confirm ~3–4x encoding-cost drop for that primitive.
- Regression test: full existing mlx test suite still green.

**Exit criterion:** RMSNorm-only migration shows the predicted
encoding drop on a microbenchmark, no regressions elsewhere.

### Phase 2 — decode-path migration (~1 week)

Migrate primitives 2–6 from the table above. Run full model
benchmarks (`--method simple --context 1024` on Gemma4-E2B and
GPT-OSS-20B) after each primitive to catch regressions early.

**Exit criterion:** decode-path primitives (excluding SDPA) all on
AB. Per-step encoding cost on GPT-OSS-20B drops from 17,107 µs to
sub-9,000 µs (conservative: 2x improvement from AB-migrated primitives
alone, before any ICB integration).

### Phase 3 — SDPA migration + kernel-selection stabilization (~1–1.5 weeks)

- Audit `fast::scaled_dot_product_attention` variants. Identify the
  T_k threshold that flips segment topology (from the diagnostic
  test: between 1024 and 1025 on tiny shapes; the actual threshold
  at production shapes may differ).
- Decide: force one kernel variant for all T_k in the decode range,
  or track kernel-variant as a per-decode-step choice.
- Migrate SDPA's argument payload to AB.
- Verify: shape-sensitivity diagnostic (already committed) reports
  "ARENAS IDENTICAL" across T_k=1024 and T_k=1025 after migration.

**Exit criterion:** SDPA recording is T_k-independent in bytes_arena
contents AND segment count.

### Phase 4 — ICB decode integration (~3–4 days)

With AB + heap in place and SDPA stable, the per-layer ICB
integration the original plan doc sketched becomes straightforward.
Wire it on GPT-OSS-20B first (clean layer loop), measure tok/s.

**Exit criterion:** GPT-OSS-20B decode tok/s ≥ 85 (baseline 64.0
→ +33%, conservative vs the 1.76x theoretical ceiling). Output
token-identical to live path on a deterministic prompt.

### Phase 5 — Rollout (as follow-up)

- Gemma 4 E2B integration (messier layer loop, but infra is now
  proven).
- Other models (Qwen3.5 hybrids may still be out of scope due to
  dispatch variability).

---

## Open questions to resolve during Phase 1

1. **AB storage mode.** Shared (CPU updates land in GPU-visible
   memory immediately) vs. private + staged writes. Likely shared
   is fine for the sizes involved (<1 KB per AB).

2. **AB lifetime and pooling.** Per-call allocation vs. primitive-
   owned vs. graph-wide arena. A per-graph arena matching the AB
   layout would amortize allocation but needs lifetime tracking.

3. **Heap sizing.** Preallocate exact bytes from weight inventory at
   load time, or grow-on-demand. Preallocate is simpler; mlx already
   computes the weight byte count during load.

4. **Kernel rewrites.** Do we require every migrated kernel to
   accept its arguments via an AB struct, or do we keep the old
   setBytes path as a fallback for non-migrated callers? Fallback
   path doubles compile time but eases rollback — probably worth it.

5. **Multi-stream interaction.** ICB multi-stream capture (shipped
   today, commit `aeeba108`) steers secondary-stream dispatches to
   the recording encoder. AB/heap integration must preserve that
   property — each stream's primitives should write into their own
   AB, not share one. Needs explicit test coverage.

---

## Fallback options

If Argument Buffers prove intractable (timeline overrun, unforeseen
Metal API limitation, kernel rewrites more invasive than scoped), the
shape-sensitivity finding forces a choice among the following
narrower alternatives. They were enumerated in the session that
produced this plan and are preserved here so the decision tree is
self-contained.

### D1 — Skip attention in the ICB

Record + replay only the shape-*independent* dispatches (RMSNorm,
MLP / MoE, projections, residual adds). Leave the SDPA call live.

- Correctness: replay always produces the right answer because
  nothing recorded is shape-dependent.
- Coverage: ~75–80% of the 874 dispatches per step on GPT-OSS-20B
  (attention is ~6–8 dispatches per layer × 24 layers ≈ 144–192
  dispatches skipped).
- Expected uplift: ~60–70% of the full ICB win, i.e. encoding-cost
  drop ~1.25–1.30x per step vs live (rough).
- Scope: mlx-swift-lm only. One function (`callAsFunction`) per
  model, bracketing the attention call with live pass-through.
  ~4–6h implementation.
- Downside: ships a partial solution. Doesn't reduce the per-
  dispatch setBuffer/setBytes overhead at all on the replayed
  portion — only saves pipeline-state setup.

### D2 — Rounded fixed K/V shape + dynamic mask

Pre-allocate `cache.keys` / `cache.values` at `maxKVSize`. SDPA
always sees `T_k = maxKVSize`. Correctness maintained via a mask
MLXArray that marks invalid positions.

- Correctness: depends on whether setBytes encodes just T_k or also
  a mask bound separately. Mask bound would still vary per step →
  setBytes still T_k-dependent.
- Scope: requires the mask to be `tag`ged for override, and the
  cache layout changed. Touches both mlx-swift-lm (cache class) and
  potentially mlx (SDPA kernel path if it optimizes out
  zero-masked regions).
- Doesn't handle the segment-topology difference unless the chosen
  `maxKVSize` is past every internal SDPA threshold — fragile.
- Estimated 1.5–2 weeks, uncertain payoff.

### D3 — Bucketed ICBs

Record one ICB per distinct T_k value encountered during
generation. At each decode step, select and replay the ICB matching
the current T_k.

- Correctness: each bucket is recorded at the T_k it replays at.
  Exact.
- Cost: O(max_gen_tokens × layers × ICB_bytes) memory. At 4K
  decode: 4K × 24 × ~200 bytes ≈ 19 MB — negligible. Warmup
  cost: first decode pass is entirely live, building 24 ICBs per
  layer. For long prompts the warmup itself is a tok/s hit.
- Novelty: unusual pattern, no prior art in mlx or similar projects
  that we know of. Higher implementation risk.
- Estimated 1–2 weeks to prototype + measure.

### D4 — Attention-kernel rewrite only

Narrower than full AB adoption: rewrite just
`scaled_dot_product_attention` so T_k and mask bounds come from a
buffer instead of setBytes, and unify its kernel-selection logic so
segment topology is T_k-stable.

- Scope: touches one primitive (high-leverage, critical path) and
  the associated Metal kernels.
- Would unblock full-ICB decode integration without touching every
  other primitive.
- Estimated 2–3 weeks including perf validation and parity
  testing against the existing kernel.
- Downside vs. full AB adoption: doesn't address the
  setBuffer/setBytes encoding overhead on the *other* 20+
  primitives per layer, which is where most of the CPU encoding
  time lives. Ships the correctness fix but not the performance
  ceiling.

---

## Relationship to already-shipped ICB work

Nothing in this plan supersedes what already landed on
`ek/metal-icb-prototype`. Specifically:

- `IndirectCommandRecorder` + `tag_binding` + `replay_with_overrides`
  (mlx / mlx-c / mlx-swift) — directly reused. AB shifts *what* gets
  tagged, not the tagging mechanism.
- Multi-stream capture (mlx `aeeba108`) — required for any AB
  integration that touches MoE. Shipped.
- `--method icb` benchmark (mlx-swift-lm) — continues to measure
  encoding-cost replay in isolation; useful regression tool
  throughout the migration.
- Shape-sensitivity diagnostic (mlx `6f097aa6`) — will be the
  acceptance gate for Phase 3.

The ICB tests in `icb_recorder_tests.cpp` continue to exercise the
core recorder mechanics. New tests will cover the AB + heap paths.

---

## Starting state (as of this plan)

- Multi-stream capture landed, 100% dispatch coverage on GPT-OSS-20B
  (874/874 captured, 0 leaked).
- Encoding-time measurements stable and reproducible via
  `./scripts/benchmark.sh --model <m> --method icb`.
- Baseline tok/s + KV regression findings documented in the
  projection doc and `m1-max-64gb-2026-04-17.md`.
- All four repos on `ek/metal-icb-prototype`, pushed to `ekryski/*`.

A fresh session starts Phase 1 with no rederivation needed.
