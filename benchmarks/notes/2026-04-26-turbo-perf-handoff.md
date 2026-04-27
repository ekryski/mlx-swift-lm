# TurboQuant Perf Investigation — Session Handoff (2026-04-26)

Picking-up doc for the turbo-quant performance investigation. Captures what's merged, the A/B path baseline numbers, what we ruled out, and the prioritized backlog.

## Branch to start from

**`alpha`** — has all 4 merged PRs from this session (#101, #104, #105, #106) plus the cleanup PRs (#95, #96). All other turbo-related branches (`ek/turbo-kv-fixes`, `ek/turbo-kv-beta`, `ek/turbo-kv-lazy-alpha`, `ek/profile-investigation`) are either fully superseded by alpha or were exploratory branches that landed via PR. Don't rebase from those — they're dead weight.

```bash
git checkout alpha && git pull origin alpha
git log --oneline -6
# 87eafee docs(bench): in-process [MLX-PROFILE] aggregator + xctrace recipes  (#106)
# a0e5d36 refactor(turbo-kv): replace invasive eval-barrier profiling …       (#105)
# e9679d8 fix(turbo-kv): replace eval barrier with dtype cast …               (#104)
# b55c199 perf(turbo-kv): bypass rotation in A path …                         (#101)
# 07ece2a chore: remove native C++ prefill bridge                             (#95)
# e6d0969 chore: remove MLXServer                                             (#96)
```

**Working branch (long-running):** cut a single branch from alpha and stack all the backlog steps on it. Suggested name: `ek/turbo-kv-perf`.

```bash
git checkout alpha && git pull origin alpha
git checkout -b ek/turbo-kv-perf
git push -u origin ek/turbo-kv-perf
```

Each backlog step lands as one or more commits on this branch. Bench each step in place via `[MLX-PROFILE]` smoke; capture the delta in commit messages. Comprehensive 8-model sweep at the very end, and we either open one consolidated PR or split into a small number of focused PRs once we're confident in the wins.

If alpha advances during the session (other PRs merging), `git rebase alpha` periodically to stay current — none of the backlog steps should conflict with unrelated alpha work, but rebasing keeps it clean.


## TL;DR

- **A path** (default since PR #101) matches `--kv none` for memory and decode tok/s. No regression. Shipped.
- **B path** (`useCompressedAttention=true`, opt-in) is single-stream-correct as of PR #104, but **15-49% slower than A** depending on model — encode kernel dominates per-layer cost ~9:1 over the TurboFlash attention kernel itself. Unblocks single-stream B-as-storage; doesn't beat A on decode tok/s.
- **Memory**: peak GPU is dominated by prefill activations, not KV. KV-path optimization (A vs B) doesn't move peak. Real peak savings need a different attack (chunked prefill SDPA, paged KV).
- **Profiling**: `MLX_BENCH_PROFILE=2` now emits an `[MLX-PROFILE]` per-phase CPU table at end of every cell + `os_signpost` intervals readable by xctrace + Instruments. Both paths (A, B, affine, default) are instrumented.
- **Session 2/3 update (2026-04-26 PM)**: After discovering and working around a workflow gotcha (see "Workflow gotcha" section below), retested every hypothesis with the C++/Metal changes ACTUALLY in the compiled binary. **Final answer: the bf16 kernel output ALONE fixes the bug.** Validated on Qwen 0.8B / Qwen 9B / Nemotron 30B at 4k with `useCompressedAttention=true` and the asType cast removed entirely — all three produce coherent output and decode at or near the alpha baseline. H10 (donation_lock_) and H3 (stopGradient) ALSO fix the bug, but redundantly — they were tested only with bf16 active. The asType cast in [TurboQuantKVCache.swift](Libraries/MLXLMCommon/TurboQuantKVCache.swift) is removed. The H12 deep-dive in `device.cpp` (`prev_outputs_ = std::move(next_outputs_)` replaces rather than unions) may still be a real upstream MLX hazard-tracking quirk worth investigating, but is no longer load-bearing for turbo correctness.
- **Session 2/3 workflow gotcha discovered**: SPM compiles C/C++ from `.build/checkouts/mlx-swift/`, NOT from the sibling `Packages/mlx-swift` symlink (which only redirects Swift). And the metallib is built from `Source/Cmlx/mlx-generated/metal/`, NOT from `Source/Cmlx/mlx/mlx/backend/metal/kernels/`. Edits to the sibling kernel/host code are NOT picked up automatically. To take effect for in-place iteration: `chmod u+w` and copy into `.build/checkouts/...` (and `mlx-generated/metal/...` for kernels) before `make clean-cmlx && make`. For shippable changes: commit/push to the sibling's remote and re-pin `Package.resolved`. This was the root cause of the entire confusion in Sessions 2 and the first half of 3.

## Merged PRs (alpha)

| PR | Title | What it does |
|---|---|---|
| #101 | bypass rotation in A path | A default skips rotation+dequant buffer; uses raw FP16 cache directly. Decode +3-23% on small models, peak unchanged. |
| #104 | dtype cast in compressedAttention | Replaces `eval()` barrier (40-60% drag on small-nKVH) with single `output.asType(queries.dtype)`. Fixes Qwen 9B B-path coherency (was emitting `!!!!!`). +50-240% B-path decode vs eval-barrier baseline. |
| #105 | signpost refactor | Removes invasive `MLX_BENCH_PROFILE=3` eval-barrier profiler. Wraps phases with `os_signpost` intervals (zero cost when no tracer). New labels: `kv_update`, `sdpa`, `qsdpa`, `tq_encode`, `tq_score`, `tq_softmax`, `tq_value`, `tq_rotate`. |
| #106 | `[MLX-PROFILE]` aggregator + README recipes | In-process CPU wall-clock per-phase aggregator (CLI-visible without Instruments). Documents `--instrument 'Points of Interest'` requirement for xctrace + 13-row instrument inventory. |

## Open issues

- **Issue #103** — B=16 batched decode crashes with `Invalid Resource` in the prefill encoder when the dtype-cast fix is active. Single-stream is unaffected. Likely command-buffer resource pressure or lazy-graph-shape interaction with batched dispatch. Tracked, not blocking.
- **Issue #83** — TurboQuant KV crashes on the entire Gemma 4 family (E2B / E4B / 26B-A4B / 31B) with `Unable to load kernel turbo_fused_encode_wht_{4,2}_512`. Gemma 4's text decoder uses head_dim=512 on its global-attention layers (in addition to head_dim=256 on the sliding layers); the WHT-fused encode kernel is only instantiated for dim ≤ 256 today. **No open PR across mlx-swift-lm / mlx-swift / mlx / mlx-c.** Need to add `instantiate_turbo_encode_wht(bits, 512, log2(512)=9)` (and the corresponding logdim WHT-butterfly path) for `bits ∈ {2, 3, 4, 8}`. Likely also need the matching `turbo_flash_p1*_512` and pass2 `turbo_flash_p2*_512` instantiations if the rest of the B-path pipeline isn't already covered for that dim. Verify by running the Gemma 4 E2B B-path smoke after adding kernels — it should reach decode, not crash on first update.

## A vs B path baseline (M1 Max 64GB, alpha post #101 + #104)

Summarization, 4-bit weights, 400 generated tokens, `--kv turbo4v2`. Numbers from the 8-model sweep run on 2026-04-26 against alpha + PR #101.

| Model | Ctx | A decode | B decode | Δ | A peak | B peak |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 1024 | 178.9 | 120.8 | -32% | 1.64 GB | 1.64 GB |
| Qwen3.5-0.8B | 4096 | 168.0 | 121.5 | -28% | 1.69 GB | 1.69 GB |
| Qwen3.5-0.8B | 32768 | 102.5 | 52.3 | -49% | 1.98 GB | 1.98 GB |
| Qwen3.5-9B | 1024 | – | 48.5 | – | – | 5.87 GB |
| Qwen3.5-9B | 4096 | – | 44.2 | – | – | 5.98 GB |
| Qwen3.5-9B | 32768 | – | 22.0 | – | – | 6.84 GB |
| Nemotron 30B | 1024 | 71.3 | 60.9 | -15% | 17.69 GB | 17.69 GB |
| Nemotron 30B | 4096 | 73.0 | 59.5 | -18% | 17.72 GB | 17.72 GB |
| Nemotron 30B | 32768 | 59.6 | 43.2 | -28% | 17.91 GB | 17.91 GB |

A path full sweep (8 models × 3 contexts × {none, turbo4v2}): see `benchmarks/m1-max-64gb-2026-04-26.md` (re-generated by the sweep on top of PR #101).

### Per-phase profile snapshot (Qwen 0.8B 4k turbo4v2, 200 max tokens)

```
A path:                                       B path:
decode_step  6123 µs/token                    decode_step  8651 µs/token
  kv_update    22 µs/call (× 2430 calls)        tq_encode    42 µs/call (× 2406 calls)
  sdpa          2 µs/call (× 2430 calls)        tq_value     11 µs/call
                                                tq_rotate     7 µs/call
```

**Key insight from H7 profiling:** CPU dispatch difference (B - A ≈ 83 ms over 200 tokens) is only **8% of the actual decode wall difference (~970 ms)**. The remaining ~887 ms is GPU execution time, not dispatch overhead. So B's bottleneck is the GPU work itself — particularly the encode kernel running on every decode step on every attention layer. CPU dispatch ratio (encode 9:1 flash kernel) mirrors GPU work ratio.

## Hypotheses landscape

| # | Hypothesis | Status |
|---|---|---|
| H1 | `MLX_METAL_PROFILE` env leak | ❌ ruled out — env clean |
| H2 | Replace eval barrier with dtype cast | ✅ shipped (PR #104) |
| H3 | Op-identity fusion barrier (`stopGradient`) | ✅ retested 2026-04-26 PM with bf16 kernel actually compiled — Qwen 9B coherent. But redundant: bf16 kernel alone is sufficient. |
| H4 | MLX submodule bisect for B regression | deprioritized — H6 + H7 explain structurally |
| H5 | Adaptive TurboFlash block size (Tom PR #93) | open — Tom's data: +25-33% on long ctx, but compressed-attention still loses 35-56% to A even with it |
| H6 | Codec-rework commits broke B kernel reads | ❌ ruled out — kernel-facing surface unchanged from April baseline |
| H7 | Profile B path with `MLX_BENCH_PROFILE=3` | ✅ done — encode dominates 9:1, GPU-bound |
| H8 | bf16 kernel output (ditches the `asType` cast) | ✅ **CONFIRMED 2026-04-26 PM** — once the bf16 kernel was actually compiled in (workflow gotcha section), it produced coherent output on every model tested with the asType cast removed. **This is the ship fix.** |
| H9 | Deferred batch encode (encode every N tokens) | open — only real B-path perf lever per H7 |
| H10 | `donatable=false` at mlx-c (proper #87/#92 fix) | ✅ retested 2026-04-26 PM with C++ actually compiled — Qwen 9B coherent. But redundant: bf16 kernel alone is sufficient. |
| H11 | Cmd-buffer tuning commit `5956b6c8` is the trigger | ❌ **REFUTED 2026-04-26 PM** — Qwen 9B B-path (cast off) sweep `MLX_MAX_OPS_PER_BUFFER ∈ {1, 50, 100, 200, 500}` and even `ops=1 mb=1`: bug reproduces at every cap value. Cmd-buffer batching is not the trigger. |
| H12 | Buffer-pointer collision in pooled allocator (hazard tracker fooled by reuse) | ⚠️ partially confirmed: trace clearly shows `prev_outputs_ = std::move(...)` REPLACE-not-UNION causes pass2-output ptrs to fall out of the tracker. **However, the UNION fix tested in isolation (FP32 kernel + UNION + no cast) does NOT fix the bug — Qwen 9B still emits `!!!!!`.** So the move-not-union behavior is observable but isn't the load-bearing cause. Whatever the bf16 kernel does (smaller buffer size? different fusion plan? different hazard scope?) is the actual lever. The hazard tracker mystery remains for future investigation; **bf16 kernel output is the durable, validated fix**. |

## Session 2 deep-dive (2026-04-26 PM): fusion bug isn't what we thought

We attempted Step 1 (H8 bf16 output) and Step 2 (H10 donatable=false) on `ek/turbo-kv-perf` cut from alpha. Both failed in instructive ways and reshaped our understanding of the bug.

### Step 1 results

Three sub-variants on Qwen 0.8B 4k turbo4v2 / Qwen 9B 4k canary:

| Variant | Qwen 9B output | Qwen 0.8B decode | Notes |
|---|---|---:|---|
| Alpha B (PR #104, fp32 kernel + asType→bf16) | ✓ "Thinking..." | 121.5 tok/s | doc baseline |
| **B' (bf16 kernel + asType cast retained as no-op)** | ✓ "Thinking..." | **125.5 tok/s** | tested OK — small win, low confidence (within noise) |
| B (bf16 kernel, drop asType) | ✗ `!!!!!!!` | n/a | **fusion bug returns** — bf16-native output alone doesn't create a sufficient graph boundary |

Per-phase signposts on B' showed `tq_encode` ~42 µs (= alpha baseline) — the bf16 output didn't move per-phase numbers because the bottleneck is GPU encode work, not the dtype cast (consistent with H7).

### Step 2 (H10 donation_lock_) result

Implemented exactly as the doc described: added `mutable std::shared_ptr<array::Data> donation_lock_;` to `TurboFlashPass2` in `fast_primitives.h`, then in `eval_gpu` after `out.set_data(...)` did `donation_lock_ = out.data_shared_ptr();` to bump `data.use_count()` to 2 (so `is_donatable()` returns false for the lifetime of the output array's graph node). Combined with dropping the `asType` cast in Swift.

**Result on Qwen 9B 4k canary: `!!!!!` — fusion bug NOT fixed.**

### H3 stopGradient result

Replaced `output.asType(queries.dtype)` with `stopGradient(output)` (an explicit op-identity / autodiff barrier). With or without `donation_lock_` in place.

**Result on Qwen 9B 4k canary: `!!!!!` — fusion bug NOT fixed.**

### What we learned about the bug

Three things that should "create a graph boundary" do nothing:
1. `donation_lock_` on the producer's output (data refcount ≥ 2).
2. `stopGradient` on the consumer side (explicit op-identity barrier).
3. `bfloat` kernel output (different dtype than what the upstream consumer was expecting on the alpha-prior path).

One thing that does work:
4. `asType(queries.dtype)` — even when source dtype == target dtype (no actual cast needed).

So the bug is not "graph fuses through TurboFlash output" in a generic sense, and it's not "donation aliases the output buffer". It's something more specific to **what `asType` does that `stopGradient` and donation prevention don't**.

### Hypothesis: it's command-buffer batching, not graph fusion

The user's hunch — that this is a side-effect of mlx submodule commit `5956b6c8` "perf(metal): output-byte tracking + tuned cmd-buffer defaults (Max/Ultra)" — fits the evidence well.

**Timeline:**
- 2026-04-23: `5956b6c8` lands. M1/M2/M3 Max bumped from `max_ops_per_buffer=200, max_mb=50` to `ops=500, mb=100`. Output-byte tracking added to `register_output_array`. The commit-heuristic switched from input-bytes-only to (input+output)-bytes.
- 2026-04-24: First eval-barrier fix (`ac4cfc5`) "fix(turbo-kv): eval barrier after TurboFlash on nKVH=2 shapes (closes #87)". **One day later.**
- 2026-04-26: PR #104 swaps the eval barrier for the asType cast.

**MLX hazard-tracking model** (read out of `mlx/backend/metal/device.cpp`):

1. Every compute encoder is created with `MTL::DispatchTypeConcurrent` (line 601). **Within one encoder, kernel dispatches run concurrently on the GPU unless explicit memory barriers are inserted.**
2. Per-dispatch barrier check: `set_input_array` sets `needs_barrier_` if the input buffer pointer matches anything in `prev_outputs_`. `prev_outputs_` accumulates outputs from earlier dispatches in the same encoder. `dispatch_threads` calls `maybeInsertBarrier` first, which emits `memoryBarrier(BarrierScopeBuffers)` and clears the flag.
3. Cross-encoder synchronization is via Metal fences in `prev_ce_outputs_` populated at `end_encoding` (line 487-503).
4. Hazard tracking is **buffer-pointer-based**: `a.buffer().ptr()`. The pooled allocator can hand out the same buffer pointer repeatedly (which is why `5956b6c8`'s output-byte tracking added a separate dedup'd counter for memory-pressure decisions).

**Why this could explain `asType` being uniquely effective:**

`astype(x, x.dtype())` is NOT a pure no-op in MLX — even for matching dtypes, it dispatches a copy kernel that allocates a fresh output buffer. That fresh allocation gives the consumer a *different* `buffer().ptr()` than the pass2 output. If the bug is in pass2-output's hazard tracking (e.g., the buffer ptr collides with something already in `all_outputs_`/`prev_outputs_` so `needs_barrier_` doesn't fire when it should), introducing an intermediate `Y = astype(X)` buffer sidesteps the bad tracking — the consumer reads `Y`, not `X`, and `Y` was tracked correctly because it's a fresh allocation.

`stopGradient` in MLX inference is graph-only metadata (no kernel dispatch). Doesn't allocate a new output buffer. Doesn't change the buffer pointer the consumer sees. → no help.

`donation_lock_` only changes the refcount of `array_desc_->data`. It doesn't change buffer pointers either. The hazard tracker doesn't read `is_donatable()` — it reads buffer ptrs. → no help.

This story is consistent with all four data points.

### H11 experiment results (2026-04-26 PM) — REFUTED

Sweep on Qwen 9B 4k turbo4v2 with the `asType` cast disabled (i.e., bug-prone state). Single arch, single context, 80 generated tokens per cell.

| `MLX_MAX_OPS_PER_BUFFER` | Output | Decode tok/s |
|---|---|---:|
| 1 (commit per dispatch) | `Thinking!!!!!!!` (corrupt) | 27.9 |
| 50 (upstream main default) | `!!!!!` | 39.4 |
| 100 | `!!!!!` | 39.1 |
| 200 | `!!!!!` | 39.2 |
| 500 (current alpha default) | `Thinking!!!!!!!` | 39.1 |
| 1 + `MLX_MAX_MB_PER_BUFFER=1` | `Thinking!!!!!!!` (corrupt) | 26.9 |

**Findings:**

1. **Bug reproduces at every cap value**, including the most aggressive `ops=1 mb=1`. Cmd-buffer batching is not the trigger.
2. **Decode tok/s is essentially flat from ops=50 to ops=500** on Qwen 9B 4k (39.1-39.4). `ops=1` costs ~30% (27.9). So *some* batching helps but increasing the cap above ~50 yields no observable benefit on this workload.
3. **Caveat from prior multi-arch profiling**: an earlier sweep across multiple models and longer contexts (32k) had previously found ops=500 fastest, especially at long ctx. **This single-workload Phase-2 reading at 4k on a small model is not sufficient to overturn that.** Don't change the cap based on this data alone.

### Pivoting: what the failures collectively tell us

After three failed fixes (H8, H10, H3) and one refuted hypothesis (H11), the bug profile is sharper:

- It is **shape-dependent** — Qwen 9B (nKVH=4) reproduces; Qwen 0.8B doesn't visibly garble (output stays coherent even with cast off).
- It is **NOT** about graph fusion in the lazy-compile sense, NOT about donation/aliasing on the output array, NOT about cmd-buffer batching, NOT defeated by op-identity barriers.
- It IS defeated by `asType` even when source==target dtype. The unique thing about `asType` (vs `stopGradient`) is that it dispatches a copy kernel that allocates a fresh output buffer — a different `buffer().ptr()`.

**New leading suspect (H12)**: buffer-pointer collision in MLX's pooled allocator + buffer-pointer-based hazard tracking. If the pass2 output buffer is allocated at a pointer that the allocator has handed out previously for an unrelated array within the same encoder, the `prev_outputs_` / `all_outputs_` sets either falsely register or mis-track the dependency. The asType-allocated copy-output sidesteps this because it gets a fresh allocation at the moment the pool is in a different state.

**Concrete next experiments for H12:**

1. **Instrument `register_output_array` and `set_input_array` to log `(buffer ptr, primitive name)`** on every call within a decode step on Qwen 9B B path. Look for a buffer ptr that's an output of dispatch N AND an input of dispatch M where the producer-consumer dependency is on `pass2 → next-op` but the same ptr appears earlier in the encoder (collision). One pre-encode + cleanup; concrete diagnostic.
2. **Disable the pooled allocator's reuse on bf16 outputs** — force fresh allocation. If the bug disappears, that confirms pool-reuse-with-collision.
3. **xctrace with subsystem `ai.mlx.metal`** during a corrupted run, look at per-kernel-dispatch resource read/write windows on the affected buffer.

### Closed-out experiments (kept for record)

The following were tried and don't help — don't repeat:
- `donation_lock_` in `TurboFlashPass2` (refcount ≥ 2): no effect on the bug.
- `stopGradient(output)` after the kernel: no effect.
- `MLX_MAX_OPS_PER_BUFFER=1` (force commit-per-dispatch): no effect.
- `MLX_MAX_OPS_PER_BUFFER=1 MLX_MAX_MB_PER_BUFFER=1`: no effect.
- Reverting `5956b6c8` only — would have been useful, but the H11 sweep already covers the same ground via the env-var override (which exercises the exact same `commit()` paths). Skippable.

## Session 3 H12 confirmation (2026-04-26 PM, very late)

### Setup

Added an env-gated diagnostic logger (`MLX_HAZARD_TRACE=1`) to `CommandEncoder` in `mlx/backend/metal/device.cpp`. Logs every `set_input_array` / `set_output_array` / `register_output_array` / `maybeInsertBarrier` / `end_encoding` with kernel name, buffer pointer, size, and the critical bits: `prev_hit` for inputs (matches `prev_outputs_`?) and `first_seen` for outputs (first time this ptr was registered as an output in the encoder?).

Test harness gotcha: `swift test --filter benchmark` redirects stderr away from the shell, so `fprintf(stderr, ...)` is invisible. Run `xcrun xctest` directly on the test bundle to capture stderr in the parent shell:

```bash
TURBO_USE_BETA=1 MLX_HAZARD_TRACE=1 \
  MLX_BENCH_MODEL=qwen35-9b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_NGRAM=0 MLX_BENCH_MAX_TOKENS=1 MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit \
  xcrun xctest .build/arm64-apple-macosx/release/mlx-swift-lmPackageTests.xctest 1>/tmp/q9b.out 2>/tmp/q9b.err
```

Trace volume: ~9.8M lines / 1.1 GB for one decode token of Qwen 9B at 4k. Filter to the lines that matter (pass2 outputs and their consumers).

### The smoking gun

Filtered to pass2 dispatches, output buffer, and the next consumer that reads that buffer:

```
PASS2-OUT disp=5080 ptr=0x13c0489a0
  CONSUMER disp=5084 kernel=vv_Multiplyfloat32 prev_hit=0 barriers_after_pass2=3
PASS2-OUT disp=5209 ptr=0x13c0dc3d0
  CONSUMER disp=5213 kernel=vv_Multiplyfloat32 prev_hit=0 barriers_after_pass2=3
PASS2-OUT disp=5338 ptr=0x13c098170
  CONSUMER disp=5342 kernel=vv_Multiplyfloat32 prev_hit=0 barriers_after_pass2=3
... (every pass2 dispatch, all 24 attention layers, all bf16 outputs going to the same broken pattern)
```

Every pass2 output is consumed 4 dispatches later by `vv_Multiplyfloat32` (the attention-output projection). Three intermediate barriers fire in between (for unrelated reasons). **In all cases the consumer's `set_input_array` sees `prev_hit=0` for the pass2 buffer pointer**, even though pass2 wrote to that exact pointer just 4 dispatches earlier. **No barrier is emitted for the pass2→multiply RAW dependency.**

### Root cause

In `mlx/backend/metal/device.cpp:376` (`maybeInsertBarrier`):

```cpp
void CommandEncoder::maybeInsertBarrier() {
  if (needs_barrier_) {
    get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
    needs_barrier_ = false;
    prev_outputs_ = std::move(next_outputs_);   // <-- REPLACES, doesn't UNION
  } else {
    prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  }
  next_outputs_.clear();
}
```

When ANY dispatch's barrier fires (because that dispatch's input matched `prev_outputs_` for an unrelated reason), the existing `prev_outputs_` is **replaced** with just the current dispatch's outputs. All earlier outputs that haven't been consumed yet — including pass2's, which is awaiting its consumer — are dropped from the tracker. When the actual consumer arrives a few dispatches later, the buffer pointer is no longer in `prev_outputs_`, so `set_input_array` returns `prev_hit=0` and `needs_barrier_` stays false. Metal's `MTL::DispatchTypeConcurrent` then runs the consumer concurrently with pass2 → race → garbage output.

The intent of the move-not-union was: "after a barrier, all prior writes are fenced, so we don't need to track them anymore." That's correct in principle for `memoryBarrier(BarrierScopeBuffers)` — the docs say it fences ALL prior memory ops. But empirically, **with `MTL::DispatchTypeConcurrent` the barrier doesn't actually serialize pass2's writes against the multiply that comes after the *next* barrier**. Either Apple's barrier semantics are narrower than the docs imply, or there's a Metal driver bug specific to this kind of chained "barrier-then-more-dispatches-then-consumer-of-pre-barrier-output" pattern.

### Why `asType` is uniquely effective

`output.asType(queries.dtype)` (even when source==target dtype) inserts a fresh kernel between pass2 and the multiply. That kernel's `set_input_array(pass2_output)` runs *immediately* after pass2 (before any other barrier could replace `prev_outputs_`), so it sees `prev_hit=1` and triggers a barrier directly tied to pass2's output. The asType output is then a fresh allocation with a different buffer pointer, which the multiply consumes safely.

`stopGradient` and `donation_lock_` don't help because:
- `stopGradient` doesn't dispatch a kernel; it's pure graph metadata. No `set_input_array` call ever sees the pass2 output → no barrier opportunity.
- `donation_lock_` only changes refcounts. The hazard tracker is buffer-pointer-keyed, not refcount-keyed. No effect.

`MLX_MAX_OPS_PER_BUFFER=1` doesn't help because cross-encoder fences (`prev_ce_outputs_`) are also keyed by buffer pointers and have an analogous "the fence list is current-encoder-scoped" structure. The pass2 output's fence is registered at end_encoding, but if the multiply runs in a SUBSEQUENT encoder, the fence wait WOULD be added — yet the bug reproduces. So either (a) the multiply runs in the same single-dispatch encoder as pass2 even with `ops=1` (the encoder created at start of "buffer N+1" still has pass2's fence in `prev_ce_outputs_`, and the multiply's `set_input_array` should add the fence wait at end_encoding), or (b) the fence-wait mechanism has the same kind of bug.

### Proposed fix

In `maybeInsertBarrier`, **union don't replace** in the barrier branch:

```cpp
if (needs_barrier_) {
  get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
  needs_barrier_ = false;
  prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());  // UNION
}
```

Slightly more conservative — `prev_outputs_` will keep growing across barriers within an encoder until end_encoding. Memory cost is bounded by encoder lifetime. This is correct because the next barrier only needs to fire if a NEW input matches; the existing fenced writes are still "candidates for needing a barrier" until proven otherwise (the next barrier WILL include them in its scope, but the cost of an extra barrier insertion is small compared to a race).

Even better fix: track barriers per-buffer, not as a global flag. Each barrier should remove from `prev_outputs_` only the buffers actually fenced by Metal at that point. But Apple's API doesn't expose that granularity, so the union approach is the practical fix.

### Concrete next experiments

1. **Apply the union fix locally and re-run Qwen 9B 4k with cast OFF.** If output is coherent, fix is validated. Should be a 5-line edit + rebuild.
2. **Quantify cost**: re-run B-path bench with the fix applied. Expect ~0% perf change since extra barrier insertions are rare and cheap.
3. **Drop the `asType` cast** in `compressedAttention` — the bug is now actually fixed at the source. Same for any other turbo-related "cast-as-fence-barrier" workarounds.
4. **Upstream PR to ml-explore/mlx**: this is a generic MLX bug. Anyone using `MTL::DispatchTypeConcurrent` with a write→long-chain→read pattern can hit it. Worth a focused report + fix PR.

## Workflow gotcha (workflow notes)

This section caused multiple hours of confusion before being unraveled. Future-Eric / future-Claude: read this carefully.

The `Packages/mlx-swift` symlink to `../mlx-swift` only overrides **Swift** sources; SPM compiles native code (Cmlx) from `.build/checkouts/mlx-swift/Source/Cmlx/`. And the metallib is built by `scripts/build-metallib.sh` from `Source/Cmlx/mlx-generated/metal/`, NOT from `Source/Cmlx/mlx/mlx/backend/metal/kernels/`.

So edits to the sibling's:
- **Swift sources** — picked up by `make spm` automatically (Packages symlink).
- **Metal kernel sources at `mlx/backend/metal/kernels/*.metal`** — NOT picked up. The metallib build looks at `mlx-generated/metal/` instead. To take effect, edits must be propagated via `tools/update-mlx.sh` (which runs cmake on the mlx submodule and copies kernels into `mlx-generated/`).
- **C++ host code under `mlx/...`, `mlx-c/...`** — NOT picked up. SPM compiles from `.build/checkouts/mlx-swift/Source/Cmlx/...`. To take effect either:
  - `chmod u+w` and copy/edit the `.build/checkouts/` files in place, then `make clean-cmlx && make`. Ephemeral — `swift package resolve` wipes them.
  - OR commit/push to the sibling's remote and re-pin Package.resolved.

For one-off investigation work (like H12 instrumentation), the chmod-and-copy approach is fine. For shippable changes, do it properly via a remote PR.

**Caveat applied retroactively**: the "Step 1 (B' bf16 kernel + asType retained) +3% on Qwen 0.8B" finding was alpha baseline measured against itself with run-to-run noise — the bf16 kernel never compiled. The bf16 kernel change is still a sensible cleanup but is currently UNVALIDATED in compiled form. To actually validate, copy the modified `turbo_quant.metal` to `Source/Cmlx/mlx-generated/metal/turbo_quant.metal` and the modified `fast.cpp` to `.build/checkouts/...`, then re-run.

### Status of the working branch

`ek/turbo-kv-perf` on all three repos. Final state of changes after Step 2 cleanup:
- **mlx**: `turbo_quant.metal` pass2 kernels write `device bfloat* output` (kept — clean change). `fast.cpp` output arrays are `bfloat16` (kept). `donation_lock_` member + assignment **reverted**.
- **mlx-swift**: clean (no edits).
- **mlx-swift-lm**: `compressedAttention` keeps the `asType(queries.dtype)` cast with an updated comment explaining it's load-bearing as something other than a graph-fusion barrier (likely cmd-buffer hazard-tracking workaround). Plus an uncommitted local env-var hack `TURBO_USE_BETA` in `init()` for B-path smoke testing.

If we ship B' as a tiny "kernel writes bf16 directly + better-comment-on-the-cast" cleanup PR, the diff is small and zero-risk. The `asType` cast stays as a documented workaround until H11 is investigated and fixed at the right layer.



For each step: **make change → small targeted bench → check `[MLX-PROFILE]` for signal → xctrace if needed**. Avoid full sweeps mid-iteration; final comprehensive sweep at the end.

### 1. bf16 kernel output (H8)

Touch points:
- `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/turbo_quant.metal:245` — `turbo_flash_pass2` kernel currently writes `device float* output [[buffer(3)]]`. Same for `turbo_flash_pass2_fused_rot:296`.
- `mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp:1655-1690` — `turbo_flash_pass2` and `turbo_flash_pass2_fused` host functions construct the output array with `float32` dtype.
- `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/turbo_quant.cpp:343-394` — `TurboFlashPass2::eval_gpu` allocates the output buffer.

Plan (original): template the kernel on output dtype (bf16 first), thread the dtype through the dispatch, drop the `output.asType(queries.dtype)` cast in `compressedAttention`. Watch for fusion-bug recurrence — without the cast, the graph fuser may alias the output buffer again.

**Update 2026-04-26 PM**: tested as Option B (hardcoded bf16 instead of templated, since the API surface change is multi-repo). Result: kernel writes bf16 cleanly, but **the asType cast cannot be dropped** — Qwen 9B emits `!!!!!`. The fp32→bf16 conversion was not the load-bearing thing about the cast; some other property of the asType primitive (most likely: that it allocates a fresh output buffer and emits a copy dispatch) is what defeats the bug. See "Session 2 deep-dive" above for the full investigation.

**Status now**: bf16 kernel output is a clean, small, optional cleanup (kernel + fast.cpp + comment update on the cast). The `asType` cast stays. **Decision**: keep the bf16 kernel change (slightly faster + no observed quality regression on the cells we tested), and document it clearly in the kernel/host code so it isn't mistaken for the cause of any future bug.

**Future work (quality validation)**: before any user-visible promotion of the bf16-output kernel, run a proper A/B comparison of perplexity (PPL) and KL-divergence (KLD) between the fp32-output baseline and the bf16-output kernel across the 8 representative models (qwen35-0.8b, gemma4-e2b, gpt-oss-20b, nemotron-30b-a3b, gemma4-26b-a4b, qwen35-27b, qwen35-35b-a3b, gemma4-31b) at 1k / 4k / 32k context. The current evidence is "produces coherent output on Qwen 9B with cast retained" — that's a weak signal. PPL/KLD against an fp32-reference checkpoint on a held-out corpus (wikitext, etc.) is the right gate.

Risks (closed-out):
- Numerical: kept accumulators (`m`, `l`, `o`) fp32 inside the kernel; only cast at final write. Confirmed correct on canary cells. Full PPL/KLD eval pending (see "Future work" above).
- Fusion bug recurrence: confirmed — yes, it does, regardless of bf16 output. Therefore the cast must be retained for now.

### 2. `donatable=false` at mlx-c (H10)

**Update 2026-04-26 PM: tested and FAILED.** Implemented as `mutable std::shared_ptr<array::Data> donation_lock_;` member on `TurboFlashPass2`, set in `eval_gpu` after `out.set_data(...)`. Bumped `data.use_count()` to 2, making `is_donatable()` return false. Qwen 9B B-path canary still emits `!!!!!`. Also tried `stopGradient` (H3) as the consumer-side barrier — same result.

**Conclusion**: the bug is not about the upstream graph fuser donating/aliasing the TurboFlash output buffer based on `is_donatable()`. The hazard tracker doesn't read `is_donatable()` — it reads `a.buffer().ptr()` for both barrier insertion (`prev_outputs_`) and for cross-encoder fence registration (`prev_ce_outputs_`). Refcount tricks don't change buffer pointers, so they don't change hazard-tracking behavior.

**Pivot to H11 (cmd-buffer batching investigation)** before any further attempt at "ship without the cast." See "Session 2 deep-dive" above for concrete experiments.

### 3. Deferred batch encode (H9)

Tom's PR #93 had this idea. Encode every N=64 decode tokens instead of every step. Cuts `tq_encode` dispatch count by ~64×.

Touch points:
- `Libraries/MLXLMCommon/TurboQuantKVCache.swift` — `compressedAttention` calls `encodeNewToken` every step; change to defer + flush every N.
- Need a "pending raw" buffer to hold un-encoded tokens between flushes.
- `trim()` must flush before modifying offsets (per Tom's PR 93 notes).

Plan:
- Add `pendingRawKeys` / `pendingRawValues` buffers.
- `compressedAttention` writes new K/V to pending + bumps offset.
- Every N steps, batch-encode pending into the packed buffer.
- For attention, score against (compressed prefix + pending raw suffix) — needs either two-stage attention or fallback to dense for the suffix.

Risks:
- Attention correctness when reading from compressed + pending mix.
- State-snapshot semantics — `state` would round-trip stale compressed prefix.

Targeted bench: Qwen 0.8B 4k + 32k turbo4v2, B path. Expected: significant `tq_encode` dispatch count reduction, decode tok/s improvement proportional to encode share of step time.

### 4. Anything that surfaces from debug

Open slot. Likely candidates from H7 data we haven't explored:
- The `tq_value` / `tq_rotate` GPU kernels — could they fuse?
- Sliding-window optimizations — most B-path bench cells aren't sliding (Qwen 0.8B/9B). Test on GPT-OSS / Gemma 4 to see if sliding-window B is dispatching efficiently. (Note: Gemma 4 E2B B-path currently fails to load `turbo_fused_encode_wht_2_512` Metal kernel — separate kernel-coverage issue.)

### 5. Adaptive block size (H5, Tom's PR #93)

Cherry-pick `87822f5` from PR #93. Drop-in additive change to `TurboQuantKernels.swift` — adds `adaptiveBlockSize(tokenCount:)` that picks 16 @ 1k, 64 @ 4k, 256 @ 32k. Tom measured +25-33% on Qwen 0.8B/2B/9B B path at 32k.

Caveats from Tom's profiling:
- 4K context is mildly worse with adaptive (-21% on 0.8B at 4K).
- Even with adaptive enabled, B path still loses 35-56% to A at every cell tested. So adaptive shrinks the gap, doesn't close it.
- Order this last so the earlier steps (which attack the encode bottleneck) don't get masked by block-size noise.

Targeted bench: same set, focused on long-context cells (32k where the win is biggest).

## Workflow for rapid iteration

### Quick smoke (per step, ~30-60 s)

```bash
MLX_BENCH_PROFILE=2 \
  MLX_BENCH_MODEL=qwen35-0.8b \
  MLX_BENCH_METHOD=summarization \
  MLX_BENCH_QUANT=4bit \
  MLX_BENCH_KV=turbo4v2 \
  MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_NGRAM=0 \
  MLX_BENCH_MAX_TOKENS=200 \
  swift test --skip-build -c release --filter benchmark
```

Watch the `[MLX-PROFILE]` table at end of output. Compare per-phase avg µs / total ms vs the baseline numbers in this doc.

### B path opt-in for benching

`useCompressedAttention` is a constructor flag, not env-gated on alpha. To smoke B path, either:
1. Re-add the env var hack locally (1 line, don't commit):
   ```swift
   // Libraries/MLXLMCommon/TurboQuantKVCache.swift, in init():
   let envBeta = (ProcessInfo.processInfo.environment["TURBO_USE_BETA"] ?? "") == "1"
   self.useCompressedAttention = envBeta || useCompressedAttention
   ```
   Then `TURBO_USE_BETA=1 swift test ...`.
2. Or hardcode `useCompressedAttention=true` in a model factory for the duration of the experiment.

### Full xctrace recipe (when phase data needs GPU correlation)

```bash
xcrun xctrace record \
  --template 'Time Profiler' \
  --instrument 'Points of Interest' \
  --instrument 'Metal Application' \
  --output /tmp/profile.trace \
  --launch -- /usr/bin/env \
    MLX_BENCH_PROFILE=2 MLX_METAL_PROFILE=1 \
    MLX_BENCH_MODEL=qwen35-0.8b MLX_BENCH_METHOD=summarization \
    MLX_BENCH_QUANT=4bit MLX_BENCH_KV=turbo4v2 \
    MLX_BENCH_CONTEXT=4096 MLX_BENCH_MAX_TOKENS=60 MLX_BENCH_NGRAM=0 \
    /usr/bin/swift test --skip-build -c release --filter benchmark
open /tmp/profile.trace
```

In Instruments: filter Points of Interest by subsystem `ai.mlx.bench` for our phases; filter another track by `ai.mlx.metal` for per-kernel-dispatch labels (gated on `MLX_METAL_PROFILE=1`). Metal Application timeline shows GPU execution windows correlated with the CPU-side intervals.

### Common bench env vars

- `MLX_BENCH_MODEL` — model alias (qwen35-0.8b, gemma4-e2b, gpt-oss-20b, nemotron-30b-a3b, gemma4-26b-a4b, qwen35-27b, qwen35-35b-a3b, gemma4-31b)
- `MLX_BENCH_METHOD` — `summarization` for our investigation
- `MLX_BENCH_QUANT` — `4bit` (typical)
- `MLX_BENCH_KV` — `none` / `turbo4v2` / `turbo4` / `affine4` etc.
- `MLX_BENCH_CONTEXT` — 1024 / 4096 / 32768
- `MLX_BENCH_MAX_TOKENS` — 200 (smoke), 60 (xctrace, keeps trace small), 400 (full bench)
- `MLX_BENCH_NGRAM=0` — disable n-gram speculative decoding (always set this for clean numbers)
- `MLX_BENCH_PROFILE=2` — emit `[MLX-PROFILE]` aggregator + os_signpost intervals
- `MLX_METAL_PROFILE=1` — adds per-kernel-dispatch signposts (subsystem `ai.mlx.metal`); paired with `=2`
- `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` — override the cmd-buffer commit caps from `5956b6c8`. Use `MLX_MAX_OPS_PER_BUFFER=1` for the H11 force-commit-per-dispatch experiment.

### Regression bench (after we're happy with fixes)

Goal: catch unintended regressions across **all model architectures**, the three baseline-relevant KV schemes, and four representative context sizes — without paying for the full matrix. One model per architecture (the smallest within each family that exposes the architecture's behavior); 4-bit weights only (bf16 is the "ideal" precision and 8-bit is a separate axis covered in comprehensive).

Architectures covered:

| Model | Architecture | Why this one |
|---|---|---|
| `qwen35-2b` | Qwen3.5 dense (GatedDeltaNet) | Smaller than the 4B/9B/27B siblings; same architecture |
| `qwen35-35b-a3b` | Qwen3.5 MoE (GatedDeltaNet + MoE) | Only Qwen MoE in the lineup |
| `gpt-oss-20b` | GPT-OSS (attention sinks) | Sinks-using path; B-path opt-out |
| `nemotron-30b-a3b` | Nemotron-H (Mamba/attention hybrid) | Hybrid cache layout |
| `gemma4-e2b` | Gemma 4 dense (sliding + global window) | Mixed window types; head_dim=256 + 512 (see #83) |
| `gemma4-26b-a4b` | Gemma 4 MoE | KV-sharing donor/acceptor pattern |

Sweep: 6 models × 3 KV ({`none`, `turbo4v2`, `affine4`}) × 4 contexts ({1024, 4096, 16384, 65536}) = **72 cells**. Estimate: ~3-5 hours on M1 Max 64GB depending on which 64k cells fit (some 30B+ models will OOM at 64k — those cells fail cleanly, the rest finish).

```bash
./scripts/benchmark.sh \
  --model qwen35-2b,qwen35-35b-a3b,gpt-oss-20b,nemotron-30b-a3b,gemma4-e2b,gemma4-26b-a4b \
  --method summarization \
  --context 1024,4096,16384,65536 \
  --quant 4bit \
  --kv none,turbo4v2,affine4
  --ppl
```

Use this between merges of any of the 5 backlog steps to confirm "no regression on architectures we didn't smoke directly during iteration." Output goes to `benchmarks/m1-max-64gb-{date}.md`.

### Comprehensive bench (only at the end of the whole effort)

Full matrix — all models × all weight quantizations × all KV schemes × all 11 standard contexts. Use as the final "ship-readiness" capture once all backlog steps are merged and the regression bench is clean.

```bash
./scripts/benchmark.sh \
  --model qwen35-0.8b,qwen35-2b,qwen35-4b,qwen35-9b,qwen35-27b,qwen35-35b-a3b,gpt-oss-20b,nemotron-30b-a3b,gemma4-e2b,gemma4-e4b,gemma4-26b-a4b,gemma4-31b \
  --method summarization \
  --quant all \
  --kv all
  --ppl
  --kld
  --think
```

Notes / caveats:
- `--quant all` covers `bf16`, `8bit`, `4bit`. The bf16 cells of bigger models (27B+) may not fit on 64GB; expect those to fail cleanly. They run on M5 Max 128GB.
- `--kv all` expands to every supported scheme: `none`, `affine4`, `affine8`, `turbo3`, `turbo4`, `turbo4v2`, `turbo4v3`, `turbo3v2`, `turbo8`, `turbo8v2`, `turbo8v4`. Some are turbo+ variants and some are alpha-only — the script gates on model support.
- `--context` defaults to all 11 sizes (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072) when unset; the explicit list is omitted above so the script can apply per-model context caps where needed.
- Expect ~12 hours+ on M1 Max 64GB depending on how many cells OOM. M5 Max 128GB is the right machine for the full matrix; M1 Max gets a partial baseline that compares like-for-like against the existing M1 Max history files.

Output: `benchmarks/m1-max-64gb-{date}.md` (or `m5-max-128gb-{date}.md` from a different machine — both are checked in for cross-machine comparisons).

## Key files and lines

| Path | Why |
|---|---|
| `Libraries/MLXLMCommon/TurboQuantKVCache.swift` | A path (`updateAndDequant`), B path (`compressedAttention`), encode (`encodeNewToken`), codec lifecycle |
| `Libraries/MLXLMCommon/AttentionUtils.swift` | A/B/affine/default routing in `attentionWithCacheUpdate`; signpost wrapping for `kv_update`/`sdpa`/`qsdpa` |
| `Libraries/MLXLMCommon/BenchmarkSignpost.swift` | Profiling infrastructure, `[MLX-PROFILE]` aggregator, phase labels |
| `Libraries/MLXLMCommon/WiredMemoryUtils.swift` | KV memory budget estimate (PR #101 fixed turbo undersizing) |
| `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/turbo_quant.metal` | Pass2 kernel — output dtype hardcoded fp32 here |
| `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/turbo_flash.metal` | Pass1 kernel (score + partial softmax) |
| `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/turbo_quant.cpp` | `TurboFlashPass2::eval_gpu` (host dispatch) |
| `mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp:1655-1690` | `turbo_flash_pass2` host function (where output array dtype is set) |
| `mlx-swift/Source/Cmlx/mlx/mlx/array.h:294` | `is_donatable()` — refcount only today |
| `benchmarks/README.md` | Profiling recipes, instrument inventory, A vs B comparison guide |

## Ground rules

### Bench tier progression: smoke → regression → comprehensive

Three tiers, picked deliberately by where you are in the iteration:

| Tier | When | Scope | Time |
|---|---|---|---|
| **Smoke bench** | Inside the iteration loop, after every code change | 1 model + 1 ctx + 200 tokens, `[MLX-PROFILE]` aggregator on. Use Qwen 0.8B 4k turbo4v2 as the default (matches our baseline numbers in this doc). Switch to Qwen 9B 4k turbo4v2 when sanity-checking the fusion-bug fix specifically. | ~30-60s |
| **Regression bench** | Between merges of any backlog step (after a step looks good in smoke, before opening a PR) | 6 models (1 per architecture) × 3 KV (`none`, `turbo4v2`, `affine4`) × 4 contexts (1k/4k/16k/64k) = 72 cells. See the [Regression bench](#regression-bench-after-were-happy-with-fixes) section above for the exact command. | ~3-5 hours |
| **Comprehensive bench** | Once at the very end of the whole effort, after all 5 backlog steps merged + regression bench clean | All 12 models × all weight quants × all KV schemes × all 11 contexts via `--quant all --kv all`. See the [Comprehensive bench](#comprehensive-bench-only-at-the-end-of-the-whole-effort) section above. | ~12+ hours; M5 Max 128GB preferred |

**Don't skip tiers** — running a regression bench after every smoke wastes hours per iteration; running comprehensive between every step is a multi-day commit. Stick to the table.

### Other rules

- **Always grab `[MLX-PROFILE]` data** for any benched change — it's the cheapest signal we have. Compare per-phase totals against the baseline numbers in this doc to see whether the change moved what we expected.
- **Watch for `!!!!!` in output samples** — Qwen 9B B path is the canary for fusion bug regressions.
- **Don't merge bf16 kernel change without testing fusion-bug recurrence** on Qwen 9B specifically.
- **When B=16 batched comes up**, defer to issue #103 — it's a separate problem space.

## Quick-reference: what regressed (and what didn't) from old branch

- ✅ Recovered: turbo4v2 decode parity with `--kv none` on alpha (A path beats April-15 B path absolute decode tok/s on most cells).
- ✅ Recovered: peak GPU regression from PR #94 fixed by PR #101.
- ❌ Not recovered: B path is still 15-49% slower than A (April baseline B was at 96-129% of `--kv none`). The old branch's "B-as-default" speed isn't reproducible on current MLX submodule pin without further work.
- ❌ Open: real peak GPU savings vs `--kv none` (KV optimization can't reach this; needs prefill-side work).

## Where to start

```bash
# Pull latest alpha (should include PRs 101, 104, 105, 106)
git checkout alpha && git pull origin alpha
git log --oneline -6

# Sanity check: smoke A path
MLX_BENCH_PROFILE=2 MLX_BENCH_MODEL=qwen35-0.8b MLX_BENCH_METHOD=summarization \
  MLX_BENCH_QUANT=4bit MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=4096 \
  MLX_BENCH_NGRAM=0 MLX_BENCH_MAX_TOKENS=100 \
  swift test --skip-build -c release --filter benchmark | grep -E "Generation|MLX-PROFILE"

# Confirm A path matches: decode ≈ 168 tok/s, sdpa avg ≈ 2 µs, kv_update avg ≈ 22 µs

# Then start step 1 (bf16 kernel output) — see plan above for files to edit
```
