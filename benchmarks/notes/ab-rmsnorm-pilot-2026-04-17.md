# AB RMSNorm Pilot — Design Note — 2026-04-17

Phase 1 of the argument-buffers adoption plan
([argument-buffers-adoption-plan-2026-04-17.md](argument-buffers-adoption-plan-2026-04-17.md))
calls for one pilot primitive migration on top of the
`ArgumentBuffer` infrastructure. RMSNorm is the chosen pilot.

This note captures exactly what needs to change, grounded in the
current mlx code so a fresh session can execute without
re-deriving.

## Infrastructure already landed

- `mlx::core::metal::ArgumentBuffer` class with typed packed slots
  ([argument_buffer.h](../../../mlx/mlx/backend/metal/argument_buffer.h)
  / [.cpp](../../../mlx/mlx/backend/metal/argument_buffer.cpp), mlx
  commit `466258cf`).
- 9 unit tests at
  `tests/argument_buffer_tests.cpp` (73 assertions, all pass).
- `BufferPtrOffset` slot layout on GPU is `{ uint64_t addr; uint64_t offset; }`,
  16 bytes — designed to be read as a raw address + cast to `device
  T*` in the kernel.

## What RMSNorm's `eval_gpu` does today

[mlx/backend/metal/normalization.cpp:17-93](../../../mlx/mlx/backend/metal/normalization.cpp#L17)

One dispatch with **6 individual binding calls**:

```cpp
compute_encoder.set_compute_pipeline_state(kernel);
compute_encoder.set_input_array(x, 0);        // setBuffer(buf, off, 0)
compute_encoder.set_input_array(w, 1);        // setBuffer(buf, off, 1)
compute_encoder.set_output_array(out, 2);     // setBuffer(buf, off, 2)
compute_encoder.set_bytes(eps_, 3);           // setBytes(&eps_, 4, 3)
compute_encoder.set_bytes(axis_size, 4);      // setBytes(&axis_size, 4, 4)
compute_encoder.set_bytes(w_stride, 5);       // setBytes(&w_stride, 4, 5)
compute_encoder.dispatch_threads(grid, group);
```

Each `set_input_array` / `set_bytes` is one ObjC message. Under an
ICB recorder, each also mirrors into the recorder's `Command`
struct + arena. That per-dispatch overhead is the thing AB removes.

## Metal kernel — current signature

[mlx/backend/metal/kernels/rms_norm.metal:12-80](../../../mlx/mlx/backend/metal/kernels/rms_norm.metal#L12)

```metal
template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_single_row(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{ ... }
```

The dtype-instantiated names today: `rmsfloat32`, `rmsfloat16`,
`rmsbfloat16`, and their `rms_looped*` counterparts.

## Target kernel — AB signature

New kernel file `rms_norm_ab.metal` with a single buffer(0) argument
struct matching the C++ `ArgumentBuffer` byte layout:

```metal
// Layout MUST match ArgumentBuffer slots declared by the C++ caller.
// Each BufferPtrOffset slot is 16 bytes: { uint64_t addr; uint64_t offset }.
struct BufferPtrOffset {
  uint64_t addr;
  uint64_t offset;
};

struct RmsArgs {
  BufferPtrOffset x;      // bytes 0..15
  BufferPtrOffset w;      // bytes 16..31
  BufferPtrOffset out;    // bytes 32..47
  float eps;              // bytes 48..51
  uint  axis_size;        // bytes 52..55
  uint  w_stride;         // bytes 56..59
  uint  _pad;             // bytes 60..63  (round the struct to 16-byte alignment)
};

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_ab_single_row(
    constant const RmsArgs& args [[buffer(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  const device T* x =
      reinterpret_cast<const device T*>(args.x.addr + args.x.offset);
  const device T* w =
      reinterpret_cast<const device T*>(args.w.addr + args.w.offset);
  device T* out =
      reinterpret_cast<device T*>(args.out.addr + args.out.offset);

  // ... body identical to rms_single_row using x, w, out, args.eps,
  //     args.axis_size, args.w_stride.
}
```

Instantiations (match existing kernel naming):
- `rms_ab_single_row_float32`, `_float16`, `_bfloat16`
- `rms_ab_looped_float32`, `_float16`, `_bfloat16`

## Eval_gpu migration — gated by env var

Add an env-var gate matching the `MLX_METAL_ICB` precedent. Enable
only when the caller is using ICB capture; otherwise keep the live
path unchanged so non-ICB decode pays zero cost from AB bookkeeping:

```cpp
static bool rms_ab_enabled_() {
  static const bool v = []() {
    const char* e = std::getenv("MLX_METAL_AB");
    return e != nullptr && e[0] == '1';
  }();
  return v;
}

void RMSNorm::eval_gpu(...) {
  // ... existing input/output setup unchanged ...

  if (!rms_ab_enabled_()) {
    // Legacy path (today's code) — unchanged.
    /* set_compute_pipeline_state, 6 individual bindings, dispatch */
    return;
  }

  // AB path.
  using Slot = ArgumentBuffer::Slot;
  ArgumentBuffer ab(d, {
    {Slot::Kind::BufferPtrOffset, 0, "x"},
    {Slot::Kind::BufferPtrOffset, 0, "w"},
    {Slot::Kind::BufferPtrOffset, 0, "out"},
    {Slot::Kind::Float32,         0, "eps"},
    {Slot::Kind::Scalar32,        0, "axis_size"},
    {Slot::Kind::Scalar32,        0, "w_stride"},
  });
  ab.set_buffer_ptr(0, x.buffer().ptr_as<MTL::Buffer>(),   x.offset());
  ab.set_buffer_ptr(1, w.buffer().ptr_as<MTL::Buffer>(),   w.offset());
  ab.set_buffer_ptr(2, out.buffer().ptr_as<MTL::Buffer>(), out.offset());
  ab.set_float32(3, eps_);
  ab.set_scalar32(4, axis_size);
  ab.set_scalar32(5, w_stride);

  auto kernel = d.get_kernel("rms_ab_single_row_" + type_to_name(out));
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_buffer(ab.mtl_buffer(), 0);
  compute_encoder.dispatch_threads(grid, group);
  compute_encoder.add_temporary(/* something that keeps ab alive */);
}
```

**Gotcha:** the AB object must outlive the dispatch. Current mlx has
`add_temporary(array)` for array lifetimes but not for raw helper
objects. Options:
1. Wrap the AB in an `mlx::core::array` — overkill.
2. Add `CommandEncoder::add_temporary_object(std::shared_ptr<void>)`
   — new API, minimal.
3. Keep a per-encoder vector of retained ABs; flushed on
   `end_encoding()` — cleanest if we're going to do this across
   many primitives.

Option 3 is the right long-term design. Add it as a small
CommandEncoder change in the same PR as the RMSNorm migration.

## Microbenchmark

Add `tests/argument_buffer_bench.cpp` modelled on
`tests/icb_feasibility_tests.cpp`:

- Iteration count: 1000 dispatches of a tiny rms kernel.
- Variant A: legacy path (6 bindings per dispatch).
- Variant B: AB path (1 binding per dispatch, shared AB).
- Report: µs per dispatch for each, ratio.

Expectation from the adoption plan: **2–3× encoding-cost drop for
this specific primitive**. If measured, RMSNorm migration is
validated; if not, we've learned the savings are smaller than
forecast and can recalibrate before doing the other primitives.

## Integration path

1. Land the new kernel + eval_gpu branch + microbench on
   `ek/metal-icb-prototype` (mlx).
2. Run the full mlx test suite to verify RMSNorm correctness on
   both the legacy and AB paths (use `MLX_METAL_AB=1` for the
   latter).
3. Run the microbench, record the ratio in
   `benchmarks/notes/ab-rmsnorm-bench-YYYY-MM-DD.md`.
4. If the ratio is ≥ 1.5x, proceed to Phase 2 (migrate Linear /
   Embedding / RoPE / SwitchLinear / softmax / elementwise). If
   not, write up the gap — the per-primitive encoding breakdown may
   differ from the macro-model the plan doc used.

## Related files at the time of this note

- Plan: `benchmarks/notes/argument-buffers-adoption-plan-2026-04-17.md`
- AB class: `mlx/mlx/backend/metal/argument_buffer.{h,cpp}`
- AB tests: `mlx/tests/argument_buffer_tests.cpp`
- RMSNorm eval_gpu (to modify): `mlx/mlx/backend/metal/normalization.cpp`
- RMSNorm kernel (to duplicate as `*_ab`):
  `mlx/mlx/backend/metal/kernels/rms_norm.metal`
- Test infra for micro-benching: `mlx/tests/icb_feasibility_tests.cpp`
