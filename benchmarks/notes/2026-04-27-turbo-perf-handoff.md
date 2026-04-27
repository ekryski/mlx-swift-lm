# TurboQuant Perf — Next Session Handoff (2026-04-27)

Picking-up doc for the next round. The 2026-04-26 doc captures the long correctness investigation (turbo decode `!!!!!` bug). This doc captures **what's shipped, what's in flight, and what's still on the table for performance work**.

## Where we are

### Branches and PRs (open as of 2026-04-27)

| Repo | Branch | PR | Status |
|---|---|---|---|
| ekryski/mlx | `ek/turbo-kv-perf` | [#18](https://github.com/ekryski/mlx/pull/18) | bf16 pass2 output + dim=512 kernels |
| ekryski/mlx-swift | `ek/turbo-kv-perf` | [#19](https://github.com/ekryski/mlx-swift/pull/19) | submodule bumps + mlx-generated metal updates |
| ekryski/mlx-swift-lm | `ek/turbo-kv-perf` | [#107](https://github.com/ekryski/mlx-swift-lm/pull/107) | drop asType cast + handoff docs + wikitext2 corpus |

Merge order: ekryski/mlx → ekryski/mlx-swift → ekryski/mlx-swift-lm (Package.resolved bump). All three target `alpha`.

### Parallel ongoing work

- [**ekryski/mlx-swift-lm#99**](https://github.com/ekryski/mlx-swift-lm/pull/99) — draft β CLI surface and **attention-sinks plumbing through β** (Tom's branch `ek/turbo-kv-beta`). Wires sinks through the TurboFlash pass2 Metal kernel + C++ + mlx-c + Swift wrappers. Coherent end-to-end on Qwen3.5 / Nemotron / Gemma 4. **GPT-OSS-20B with sinks via β still produces incoherent output** — math is consistent on paper but actual behavior diverges; sliding-window and graph-fuser ruled out. Currently routes sinks-using models through α as a fallback. Once #99 is debugged and lands, the B-path sinks limitation we documented in the 2026-04-26 doc is closed.

### What we shipped this session

- **bf16 pass2 output** — fixes the `!!!!!` decoding corruption (mlx-swift-lm #87/#92). 7-model A/B PPL sweep proved bf16 produces coherent output on every cell tested while fp32 corrupts qwen35-0.8b and qwen35-35b-a3b at ctx=1024 / 400 decode tokens. **Correctness fix, not just perf.**
- **dim=512 kernel coverage** — fixes mlx-swift-lm #83 (Gemma 4 31B's global-attention layers crashed on `Unable to load kernel turbo_fused_encode_wht_4_512`).
- **Workflow gotcha documented** — SPM compiles native code from `.build/checkouts/`, not the `Packages/<name>` symlink (which only redirects Swift). Metallib builds from `mlx-generated/metal/`. Both must be kept in sync; the symlink alone is not enough. Caused several hours of confusion when C++ "fixes" silently never compiled.
- **Hazard-tracker mystery filed at [ekryski/mlx#17](https://github.com/ekryski/mlx/issues/17)** — observable `prev_outputs_ = std::move(...)` REPLACE-not-UNION behavior in `maybeInsertBarrier` is real but isn't the root cause of the bf16-vs-fp32 corruption. Refined hypothesis: the bug is sensitive to specific pooled-allocator pointers / size classes. Parked.

## TL;DR for the next session

The **correctness work is shipped**. What remains is **performance** — the original Step 3 / H9 from the 2026-04-26 doc, plus several smaller items that didn't get touched. None of these are blocking; they're net wins to chase once the perf work resumes.

Decode tok/s on B path is still **15–60% slower than A path** depending on model (per the 2026-04-26 baseline table). Per the H7 profiling that doc captured, the bottleneck is GPU encode work (encode kernel runs every decode step, on every attention layer; dominates ~9:1 over the TurboFlash attention kernel itself). **Step 3 (H9 deferred batch encode) is the only known lever that can move headline tok/s materially.**

## Backlog (in priority order)

### 1. Deferred batch encode (H9) — biggest perf lever

**Encode every N=64 decode tokens instead of every step.** Cuts `tq_encode` dispatch count by ~64×. Tom's PR #93 had this idea. Per the 2026-04-26 H7 profiling, encode dominates B-path GPU time ~9:1 over the flash kernel itself, so this should move headline decode tok/s significantly.

**Touch points** (carried over from the prior doc):
- `Libraries/MLXLMCommon/TurboQuantKVCache.swift::compressedAttention` calls `encodeNewToken` every step; change to defer + flush every N.
- Add `pendingRawKeys` / `pendingRawValues` buffers to hold un-encoded tokens between flushes.
- `trim()` must flush before modifying offsets (per Tom's #93 notes).
- For attention: score against (compressed prefix + pending raw suffix) — needs either two-stage attention or fallback to dense for the suffix.

**Risks**:
- Attention correctness when reading from compressed + pending mix. Two-stage attention requires careful softmax merging.
- State-snapshot semantics — `state` round-trips would carry stale compressed prefix.

**Targeted bench**: Qwen 0.8B 4k + 32k turbo4v2 B path. Expected: significant `tq_encode` dispatch count reduction, decode tok/s improvement proportional to encode share of step time.

### 2. Adaptive TurboFlash block size (H5, Tom's #93 already has the implementation)

Drop-in additive change to `TurboQuantKernels.swift` — adds `adaptiveBlockSize(tokenCount:)` that picks 16 @ 1k, 64 @ 4k, 256 @ 32k. Tom measured +25–33% on Qwen 0.8B/2B/9B B path at 32k.

**Caveats** (from Tom's profiling):
- 4K context is mildly worse with adaptive (-21% on 0.8B at 4K).
- Even with adaptive enabled, B path still loses 35-56% to A at every cell tested. So adaptive shrinks the gap, doesn't close it.
- Cherry-pick `87822f5` from #93 if we want to land it cleanly on alpha.

**Order it after H9**, since the encode-dominated time profile means block-size tuning on the flash kernel is small relative to encode savings.

### 3. Lazy α / β CLI surface (Tom's #98 / #99)

Tom's #99 adds a `-compact` suffix to the `--kv turbo*` strings to opt into β at the CLI level (replaces the temporary `TURBO_USE_BETA` env hack we used during this investigation). Lands β as a memory-first opt-in, ~30–60% of α decode tok/s but significant KV memory savings on long contexts (Gemma 4 31B 245→56MB at ctx=1024, -77%; Gemma 4 26B-A4B 96→48MB at ctx=1024, -50%).

**Status**: draft, working for Qwen3.5 / Nemotron / Gemma 4. **Blocked on the GPT-OSS sinks-incoherence debug** (kernel + plumbing wired but actual behavior diverges from α). Once #99 lands, the B-path sinks limitation closes.

### 4. B-path sinks debug (carried over from #99's open follow-up)

Sinks math is implemented in the pass2 kernel + plumbed end-to-end through C++/mlx-c/Swift. On paper the formula reduces to standard SDPA with an extra log-sum-exp term. Smoke at ctx=128 fails where the buffer doesn't wrap (so it's not sliding-window-related), and the eval-barrier doesn't change behavior (so it's not a graph-fuser issue). The divergence must be in either the kernel-internal accumulation or the dispatch ordering.

Worth a focused day with the hazard tracer (`MLX_HAZARD_TRACE=1` in the mlx fork from this session, see commit `7fa48b0f`) to see whether sinks dispatches go through the same prev_outputs_ behavior as the original bug. Could be the same family of issue we chased in #87/#92 manifesting differently.

### 5. The H12 hazard-tracker mystery (filed at ekryski/mlx#17)

The observable `prev_outputs_ = std::move(...)` REPLACE-not-UNION behavior in `maybeInsertBarrier` is real (hazard trace is unambiguous) but **the UNION fix tested in isolation does NOT defeat the original `!!!!!` bug**. The bf16 output kernel does. So the move-not-union is a separate latent issue that may someday bite a different consumer.

Worth a focused investigation with a turbo-free reproducer to file upstream. Open in the issue with full evidence; left parked here.

### 6. Unrelated but adjacent

- **Issue #103** (carried over) — B=16 batched decode crashes with `Invalid Resource` in the prefill encoder. Single-stream is unaffected. Not addressed this session; still open.
- **Real peak GPU savings vs `--kv none`** — KV optimization (A vs B vs anything we did this session) doesn't move peak GPU. Peak is dominated by prefill activations. Needs a different attack (chunked prefill SDPA, paged KV).

## Repo state at start of next session

After the three PRs merge:
- mlx alpha will have the bf16 + dim=512 kernels
- mlx-swift alpha will have the submodule bump + mlx-generated metal updates
- mlx-swift-lm alpha will have the cast removal + the canonical wikitext2 corpus + the 2026-04-26 + 2026-04-27 handoff docs

Cut a fresh `ek/turbo-kv-perf-2` (or similar) from alpha. The investigation infrastructure from this session — `MLX_HAZARD_TRACE=1` env-gated logger in `device.cpp`, the wikitext2 corpus, the bench harness's `MLX_BENCH_PPL=1` / `MLX_BENCH_KLD=1` flags — is all already on alpha post-merge.

## How to reproduce the bf16 vs fp32 A/B sweep

```bash
# Start with bf16 (committed state)
cd /Users/eric/Development/personal/ai/mlx-swift-lm
make
# Run sweep using the driver in /tmp/ab_sweep.sh on the prior session machine,
# or recreate it: summarization ctx=1024 turbo4v2 B path, MLX_BENCH_PPL=1,
# MLX_BENCH_QUANT=4bit, TURBO_USE_BETA=1 (or use the Tom's `-compact` CLI
# suffix once #99 lands).

# To flip to fp32 in the working tree (don't commit):
# 1. revert ONLY the bf16 hunks from mlx commit b5346117 (kernel + fast.cpp),
#    keeping the dim=512 commit 7fa48b0f intact
# 2. revert ONLY the bf16 hunks in mlx-swift's mlx-generated/metal/turbo_quant.metal
# 3. sync to .build/checkouts/.../fast.cpp
# 4. clean-cmlx clean-metal && make
# Verify: strings .../mlx.metallib | grep 'turbo_flash_pass2_fused_rotILi64'
# should show PU9MTLdevicefR (fp32) instead of PU9MTLdeviceDF16bR (bf16)
```

Sweep results captured in `/tmp/ab_sweep.tsv` on the prior session machine; if needed, reconstruct via the matrix in PR #107's body or run `/tmp/ab_sweep.sh bf16` and `fp32` on the next machine.

## Where to start

```bash
# Pull alpha (which should include this session's PRs once merged)
git checkout alpha && git pull origin alpha

# Sanity check: 4-model Gemma 4 B-path smoke (was the canary for dim=512 fix)
TURBO_USE_BETA=1 \
  MLX_BENCH_MODEL=gemma4-31b MLX_BENCH_KV=turbo4v2 MLX_BENCH_CONTEXT=1024 \
  MLX_BENCH_NGRAM=0 MLX_BENCH_METHOD=summarization MLX_BENCH_QUANT=4bit \
  MLX_BENCH_MAX_TOKENS=20 \
  swift test --skip-build -c release --filter benchmark | grep 'BENCH'

# Should see ~10 tok/s and coherent output. If it crashes on
# `turbo_fused_encode_wht_4_512`, the merge didn't pick up the kernel changes.

# Then start H9 — see "1. Deferred batch encode" above.
```
