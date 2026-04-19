# SDPA Option C — Single-Kernel Rewrite — Plan — 2026-04-18

Unblocks AB migration of `fast::scaled_dot_product_attention`.
Companion to [`argument-buffers-adoption-plan-2026-04-17.md`](argument-buffers-adoption-plan-2026-04-17.md)
§ Phase 3 and the diagnostic at
[mlx `6f097aa6`](https://github.com/ekryski/mlx/commit/6f097aa6).

## Why Option C

Plan-doc options for unblocking SDPA's AB migration:

| Option | Approach | Cost |
|---|---|---|
| A | Force one kernel variant for entire decode T_k range | Small. Perf hit at corner cases; variant-selection logic to audit. |
| B | Variant-aware AB dispatch | Medium. CPU picks per-step; partially defeats AB's single-bind value. |
| **C** | **Rewrite SDPA into a single generic kernel** | **Large. But removes the root cause instead of papering over it.** |

Option C was chosen 2026-04-18 with a test-coverage mandate. The
rewrite removes kernel-selection flips so AB migration becomes
straightforward and ICB replay under a growing KV cache becomes
numerically correct for the first time (per the E3 measurement in
the adoption plan: stale-T_k replay drops ~28 % of valid K/V
positions by the end of a 400-token decode with one recorded
SDPA variant).

## Audit: current SDPA dispatch tree

`ScaledDotProductAttention::eval_gpu`
([scaled_dot_product_attention.cpp:665](../../../mlx/mlx/backend/metal/scaled_dot_product_attention.cpp#L665))
has four paths keyed by query length and device class:

```
q.shape(2) <= 8                              // "vector" / decode path
├── k.shape(2) >= 1024 && arch∈{d,s}         // A19/M5 large-KV, or
│   OR (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)  // GQA long-KV
│   → sdpa_vector_2pass                      // 2-kernel fused (pass1 + pass2)
│     kernels: sdpa_vector_2pass_1_<T>_<D>_<V>
│              sdpa_vector_2pass_2_<T>_<V>
│     `blocks` ∈ {64,128,256,512,1024} keyed on (N, n_simds, device)
│
└── → sdpa_vector                            // single-kernel vector SDPA
      kernel:  sdpa_vector_<T>_<D>_<V>
      D ∈ {64, 96, 128, 256, 512}; V = D in all current instantiations
      function_constants (20..25): has_mask, query_transposed,
                                    do_causal, bool_mask, float_mask, has_sinks

q.shape(2) > 8                               // "full" / prefill path
├── NAX device                               // M5 Neural-Accel matmul
│   → sdpa_full_self_attention_nax
│     kernel: steel_attention_nax_<T>_<bq>_<bk>_<bd>_<wm>_<wn>_mask<T>
│
└── → sdpa_full_self_attention_metal         // default full-attention
      kernel: steel_attention_<T>_<bq>_<bk>_<bd>_<wm>_<wn>_mask<T>
      function_constants (200,201,300..302): align_Q, align_K, has_mask,
                                              do_causal, has_sinks
```

Dispatch-topology transitions observed in the E2 diagnostic:

| Transition | Effect on encoded commands |
|---|---|
| `sdpa_vector` → `sdpa_vector_2pass` at N=1024 | commands 5 → 5, **segments 3 → 4**. New barrier. |
| Within `sdpa_vector_2pass`: `blocks` 64 → 128 at N=1024 (s-arch) | Same commands. Grid Y dim flips. |
| Within `sdpa_vector_2pass`: `blocks` flip again at N=8192, 32768, 65536 | Same commands. Grid Y dim flips. |
| `sdpa_vector` internal at T_k=64 → 96 | 2 → 3 segments. Threshold-based reduction split. |

Plus the byte-for-byte setBytes payload differs at every distinct
N, so within a single topology bucket the ICB replay still reads
stale mask bounds / scales / strides without an AB.

## Scope decision

**This plan covers the vector / decode path only:**

- `q.shape(2) <= 8` branch (decode inference).
- Merge `sdpa_vector` + `sdpa_vector_2pass` into a **single unified
  vector kernel**.
- Full-attention / prefill (`q.shape(2) > 8`) stays on the
  existing multi-variant path. Prefill is one dispatch per prompt,
  so ICB value is low — unifying it isn't on the critical path.

Rationale:
- Decode is where ICB + AB pays back. Prefill already pays the
  per-prompt encoding cost once and amortizes across hundreds of
  decode steps.
- Constrains the rewrite surface to ~800 lines of kernel + 400
  lines of CPU-side dispatch instead of the full SDPA surface.
- Delivers all of the correctness-under-growing-KV-cache benefit
  the plan doc targets.

Out of scope here (follow-on sessions):
- Full-attention kernel unification.
- NAX-specific full-attention path.
- SDPA VJP (`ScaledDotProductAttentionVJP::eval_gpu`) — training
  codepath, untouched.

## Phase breakdown

### Phase 0 — Test harness (mandatory gate before any kernel changes)

Build an end-to-end regression test that compares the unified
kernel's output against the current multi-variant path across the
full decode parameter space. **No kernel edits land until this
passes on the legacy code paths.**

Location: `mlx/tests/sdpa_regression_tests.cpp` (new file).

Coverage sweep (outer product — ~400–600 combinations total):

| Axis | Values |
|---|---|
| batch B | 1, 2, 4 |
| q heads H_q | 1, 4, 8, 32 |
| k heads H_k | 1, 4, 8 (cover GQA factor ∈ {1, 2, 4, 8}) |
| query len L_q | 1, 4, 8 (all vector-path cases) |
| kv len L_k | 1, 8, 32, 63, 64, 96, 127, 128, 256, 768, **1023, 1024, 1025**, 2048, 4096, **4095, 4097**, 8192, 32768 |
| head dim D | 64, 96, 128, 256 |
| causal | {true, false} |
| mask type | {none, bool, float} |
| sinks | {none, present} |
| dtype | {float16, bfloat16, float32} |

Bolded L_k values cross known topology thresholds per E2. Each
case computes:

```
y_ref = current_sdpa(Q, K, V, ...)   // multi-variant path
y_new = unified_sdpa(Q, K, V, ...)   // Phase-1 unified kernel
assert allclose(y_ref, y_new, rtol=1e-3, atol=1e-3)  // float16/bf16
assert allclose(y_ref, y_new, rtol=1e-5, atol=1e-5)  // float32
```

Performance floor on the sweep: harness also times each case and
stashes p50 dispatch cost. The unified kernel must stay within
1.25× of the legacy path's median time per case. Cases where the
unified kernel is *slower* than 1.25× are allowed individually
but surface a warning — they guide any post-rewrite perf tuning.

Harness dependencies:
- `mlx::fast::scaled_dot_product_attention(...)` public API.
- A new internal-only entry point
  `mlx::fast::scaled_dot_product_attention_unified(...)` that
  forces the Phase-1 kernel. Env-gated so default path is
  unchanged until Phase 4 flips the switch.
- doctest for runner integration; results also dumped to a
  markdown table at
  `benchmarks/notes/sdpa-option-c-regression-YYYY-MM-DD.md`.

Exit gate: **100 % of sweep cases passing correctness against
current legacy output, including every boldface L_k threshold.**
Time budget: ~1 session to write the harness and fixtures
(focus on the Q/K/V factories, RNG seeding, and the comparator).

### Phase 1 — Unified vector-SDPA kernel

New kernel file: `mlx/backend/metal/kernels/sdpa_unified.metal`

Objectives:

1. **Single kernel covers the full (L_q ≤ 8, L_k ∈ [1, 128K])
   decode parameter space.** No CPU-side branch selecting between
   `sdpa_vector` and `sdpa_vector_2pass`.
2. **`blocks` (= K-axis split count) becomes a runtime argument**
   passed through the kernel. CPU computes it from (N,
   gqa_factor, device); kernel uses it to drive the K-axis loop.
3. **Single reduction pattern** — always partial-accumulate per
   threadgroup + a tail merge in the same kernel. For small N
   (where legacy used single-pass), `blocks == 1` short-circuits
   the merge.
4. **No shape-dependent setBytes.** All scalars flow through the
   AB struct in Phase 3.
5. **Function constants retained** for graph-stable choices:
   `has_mask`, `bool_mask`, `float_mask`, `do_causal`, `has_sinks`,
   `query_transposed`. These don't change per step.

Signature sketch:

```metal
template <typename T, int D, int V = D>
[[kernel]] void sdpa_unified_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const constant int& blocks [[buffer(11)]],   // << NEW: runtime block count
    // mask + sinks at 12..17 as today, all function_constant-gated
    ...);
```

Body outline:

```
// Per-block partial accumulate (fmap of sdpa_vector_2pass_1 body):
for (k_block = 0; k_block < blocks; ++k_block) {
    // Load keys/values for this block's K slice
    // Online softmax accumulate with {max, sum, out}
}
// Per-threadgroup merge (equivalent to sdpa_vector_2pass_2 body,
// but always runs in-kernel so no second dispatch):
threadgroup_barrier(mem_flags::mem_threadgroup);
// Merge partial {max, sum, out} across threadgroups
// Write final output
```

The merge step is the key change — instead of a second dispatch
reading intermediate per-block state from a scratch buffer, the
single kernel does both passes via threadgroup memory. For
`blocks == 1` the merge collapses to a direct write.

Scratch buffer: the current 2-pass path allocates a per-call
intermediate. The unified kernel keeps that scratch when `blocks
> 1` but sizes it based on the runtime `blocks` value and writes
it into threadgroup memory where possible. A device-memory
fallback for very large `blocks × n_heads` stays in place.

Estimated: 1–2 sessions to write + correctness-converge against
the Phase 0 harness.

### Phase 2 — Retire legacy vector-SDPA kernels

After Phase 1 passes 100 % on the regression harness:

1. Remove `sdpa_vector`, `sdpa_vector_2pass`, `sdpa_vector_2pass_1`,
   `sdpa_vector_2pass_2` kernels and their instantiations.
2. Delete the `sdpa_vector` / `sdpa_vector_2pass` C++ helpers.
3. Remove the vector-path branch point in
   `ScaledDotProductAttention::eval_gpu`; call
   `sdpa_unified_vector` directly when `q.shape(2) <= 8`.
4. Update any tests that reference legacy kernel names.

Exit gate: regression sweep still 100 %, `./scripts/benchmark.sh
--model gemma4-e2b --method simple` within 2 % of pre-rewrite
baseline.

### Phase 3 — AB migration of `sdpa_unified_vector`

Standard pattern from the prior eight primitives. Args struct:

```metal
struct SdpaUnifiedArgs {
  BufferPtrOffset queries;
  BufferPtrOffset keys;
  BufferPtrOffset values;
  BufferPtrOffset out;
  BufferPtrOffset mask;      // zero when !has_mask
  BufferPtrOffset sinks;     // zero when !has_sinks
  int gqa_factor;
  int N;
  int blocks;
  int _pad;
  size_t k_head_stride;
  size_t k_seq_stride;
  size_t v_head_stride;
  size_t v_seq_stride;
  float scale;
  int mask_kv_seq_stride;    // valid when has_mask
  int mask_q_seq_stride;
  int mask_head_stride;
  int num_q_heads;           // valid when has_sinks
};
```

~128 bytes total. Fits cleanly in the existing
`ArgumentBuffer::Slot` kinds — no new slot type required.

CPU-side branch lives in `ScaledDotProductAttention::eval_gpu`
vector path, gated by `MLX_METAL_AB=1`. Pool + `register_*` +
`use_resource` follow the established pattern.

Exit gate: `MLX_METAL_AB=1` regression sweep matches AB=0 byte-
for-byte on the output. Bench on Gemma 4 E2B + GPT-OSS-20B must
show decode tok/s within ±2 % of AB=0 (the win lands at Phase 4,
not here).

### Phase 4 — ICB integration re-verification

With the unified kernel + AB in place, re-run the shape-
sensitivity diagnostic from [`6f097aa6`](https://github.com/ekryski/mlx/commit/6f097aa6):

```
MLX_METAL_AB=1 MLX_METAL_ICB=1 ./tests/tests --test-case="*shape-sensitivity*"
```

Expected:
- `T_k=1024` and `T_k=1025` recordings produce **identical bytes
  arenas** (the invariant the original diagnostic showed broken).
- Segment topology stable: both at the same command count and
  segment count.
- Stale-T_k replay numerical divergence (E3): must now be **0.0
  across the entire decode T_k range**, not just at the recorded
  T_k.

Exit gate: all three of the above. This is when single-ICB
replay for the full decode loop becomes numerically correct,
unlocking the original adoption-plan goal of ≥ 85 tok/s decode
on GPT-OSS-20B via one-shot-record + many-replay.

### Phase 5 — Full-attention path (follow-on)

Out of this plan's scope; opened as a spawned task when Phase 4
ships clean.

## Test harness invariants (Phase 0 expansion)

The sweep intentionally crosses the known topology flip points.
Additional invariants to explicitly assert in the test harness:

- **Causal masking**: output at position i depends only on
  K/V[0..i]. Flip K/V[i+1..] to random noise; output must not
  change. Exposes any implicit past-sequence read from a
  threadgroup tile overrun.
- **GQA correctness**: run H_q=8, H_k=1 and compare against
  H_q=H_k=8 with K/V replicated. Outputs must match within
  tolerance. Exposes any gqa_factor indexing error.
- **Mask shape broadcasting**: bool + float masks at each of
  (per-head, per-batch, shared, full-4D) shapes.
- **Sinks correctness**: output with sinks present must equal
  output with sinks re-expressed as additional K/V entries with
  the sink learnable-parameter embedding. Exposes sinks accumulator
  math errors.
- **Determinism**: same inputs → same outputs bitwise. Dispatch-
  order sensitivity here would leak through multi-stream replay.

## Risk register

- **Perf regression at small N (where legacy single-pass kernel
  was optimal).** Mitigation: unified kernel short-circuits the
  merge step when `blocks == 1`, keeping the small-N hot path
  identical to the legacy single-pass layout. Validated by the
  perf floor check in the test harness.
- **Threadgroup memory pressure from always-on merge.** Legacy
  `sdpa_vector_2pass` uses device-memory scratch for partials
  because the data is too large for threadgroup memory at high
  blocks × n_heads. Unified kernel falls back to device memory
  when threadgroup memory is insufficient; decision is compile-
  time via template parameter on `D` + runtime `blocks`.
- **Sinks + causal interaction.** Sinks add attention denominator
  contribution without K/V memory; causal masks must not filter
  sinks out. Already handled in legacy, but worth explicit test
  coverage.
- **Function-constant explosion.** 6 bool function constants ×
  5 head dims × 3 dtypes × 2 V-dims = 180 PSOs per decode. Same
  as legacy; no new explosion.

## Branching + commit strategy

- Cut a new sub-branch `ek/sdpa-option-c` off
  `ek/ab-rmsnorm-pilot` in mlx. Keeps the 8-primitive shipped
  state stable as `ek/ab-rmsnorm-pilot`; SDPA work isolates on
  the sibling branch. If the rewrite has to abandon, the current
  measured +2.7–4.4 % decode win stays intact.
- Phase 0 lands as its own commit (test harness + doctest
  integration + baseline capture on legacy).
- Phase 1 as its own commit after the harness passes against the
  new kernel.
- Phase 2 removes legacy code in a separate commit so blame is
  clean.
- Phase 3 AB migration.
- Phase 4 verification + docs.

One primitive per commit, same discipline as the prior eight.

## Time estimate (sessions)

| Phase | Effort |
|---|---|
| 0 — Test harness + fixtures | 1 session |
| 1 — Unified kernel body | 1–2 sessions |
| 2 — Retire legacy kernels | 0.5 session |
| 3 — AB migration | 0.5 session |
| 4 — ICB re-verify + bench | 0.5 session |
| Total | **3.5–4.5 sessions** |

Phase 0 is the most important. Every later phase's correctness
claim is only as strong as the regression sweep, so over-investing
here pays back through the rest.

## Success criterion (end-to-end)

1. SDPA output on every case in the Phase 0 sweep matches the
   pre-rewrite multi-variant output within tolerance.
2. Gemma 4 E2B + GPT-OSS-20B decode tok/s with MLX_METAL_AB=1 on
   the 9-primitive stack meets or exceeds the current 8-primitive
   measurement at n=6 within each model's noise floor.
3. Shape-sensitivity diagnostic `6f097aa6` reports **"ARENAS
   IDENTICAL"** across T_k=1024 ↔ 1025, and stale-T_k replay
   divergence is exactly 0.0 across the full decode T_k range.
4. GPT-OSS-20B decode tok/s with MLX_METAL_AB=1 **AND** ICB
   integration ≥ 85 tok/s (the plan-doc headline goal, currently
   blocked on SDPA variant stability).
