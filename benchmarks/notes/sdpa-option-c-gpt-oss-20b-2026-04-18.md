# SDPA Option C — GPT-OSS-20B benchmarks + decode-loop integration gap

Measured on M1 Max, 64 GB, macOS 15.7.4.

mlx branch: `ek/sdpa-option-c` (Phase 0–2 + diagnostic upgrade)
mlx-swift submodule: pointed at `ek/sdpa-option-c` for the Phase 2 test
Model: `loan-star/gpt-oss-20b-mlx-4Bit` via `--method simple`, 200 tokens, 101-token prompt

## Decode tok/s — Phase 2 on its own is a wash

| Configuration | Run 1 | Run 2 | Run 3 | Mean | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| AB off (alpha baseline) | 46.2 | 45.7 | 45.8 | **45.9** | — |
| AB=1, SDPA forced legacy (8 primitives) | 47.1 | 46.9 | 46.5 | **46.8** | +2.0% |
| AB=1, full Phase 2 (9 primitives incl. SDPA unified+AB) | 46.8 | 46.3 | 46.1 | **46.4** | +1.1% |

**Read.** The 8-primitive AB stack pays a small +2% in decode tok/s. Adding the unified+AB SDPA on top is roughly neutral (within noise). SDPA accounts for only ~4.7% of per-step dispatches on GPT-OSS-20B (E1 measurement), so its direct contribution to CPU encoding time was always small. The value of Phase 2 is the **correctness unlock** for ICB replay, not a standalone perf win.

`MLX_SDPA_FORCE_LEGACY=1` is the debug-only override that sends SDPA through legacy kernels even under the AB gate — used here to isolate SDPA's contribution from the 8 shipped primitives.

## ICB encoding microbench — 1.40× speedup holds

```
[RESULT] GPT-OSS 20B ICB encoding | live 17468.1 us/step | replay 12445.4 us/step | 1.40x
```

The existing `--method icb` harness records step 2 of a decode and replays N times without advancing the KV cache. This measures **CPU encoding cost reduction in isolation** — no cache advancement, no token generation. Matches the ~1.45× in the prior adoption-plan memory note.

Under `MLX_METAL_ICB=1` (which since Phase 1 also activates all 9 AB primitives), the replay is now architecturally safe across T_k changes:
- SDPA's setBytes arena is empty (Phase 2 regression gate confirms)
- Diagnostic `6f097aa6`: "ARENAS IDENTICAL (both empty)" at T_k=1024 ↔ T_k=1025

## The gap — ICB replay in real decode

The 1.40× encoding win is only realized if ICB replay participates in actual token generation. That requires the decode loop to:

1. Record step 2's compute graph once.
2. Replay the recorded ICB for every subsequent step.
3. Per step, update shape-dependent state (T_k, mask bounds, new hidden state input) so the replayed dispatches compute on *this step's* state.

Phase 2 made the setBytes arena stable (empty) but **does not solve the per-step update problem**. My current AB lifecycle is per-call: each `sdpa_vector_unified` call allocates a fresh `ArgumentBuffer` from the pool, fills it with *that call's* T_k and strides, and binds it. At record time the ICB captures `setBuffer(AB_step2, 0)`. At replay the pointer is frozen, the AB contents still reflect step 2's T_k, and SDPA silently ignores K/V positions past the recorded T_k — the exact 28%-drop bug Option C was supposed to close.

### Design options for closing the gap

| Option | Mechanism | Pro | Con |
|---|---|---|---|
| **A — persistent per-call-site AB** | Each primitive remembers which AB belongs to "this call site in this layer" and reuses it across steps. CPU rewrites contents each step. ICB records one bind to a stable pointer. | Cleanest architecturally. No override plumbing. AB pointer genuinely stable. | Requires mlx Primitive to carry call-site identity. Non-trivial refactor touching 9 primitives. Lifecycle: AB must outlive the ICB (not the command buffer). |
| **B — per-call AB + override rebind** | AB still allocated per call (current Phase 2). During recording, tag the AB bind as overridable. At replay, Swift passes fresh ABs via `replay(overrides:)`. | Current AB lifecycle unchanged. Reuses the existing `tag_binding` infrastructure. | Every AB-bind site in every AB-migrated primitive needs to participate in tagging. Swift side needs to build ABs per step and hand them in. `tag_binding` today works on MLXArray bindings — needs extension for raw AB buffers. |
| **C — persistent AB + Swift content-update API** | Same as A, plus expose `IndirectCommandBuffer.updateBytes(offset:data:)` to Swift so the decode loop rewrites T_k directly on the shared-storage buffer per step. | No Swift-side AB rebuild. No overrides. Matches Apple's "AB + ICB designed together" guidance most directly. | Still needs per-call-site AB identity. Adds a new Swift API. Caller must understand byte layout of each primitive's AB. |
| **D — per-step re-record** | Don't replay; record each step fresh. Keep AB fresh. | Zero new design. | Defeats the purpose of ICB — this is what alpha effectively does today. Drops back to live tok/s. |

### Recommendation

Option **A** is the cleanest. The AB lifecycle should mirror the recorder's lifecycle — one AB per (primitive instance × layer × call site), owned by the model / cache / primitive, reused across every step. The decode loop writes per-step scalars (T_k, cache offset) into the AB, the ICB replays, GPU reads fresh values.

**But Option A is ~1-2 weeks of work in mlx C++ + mlx-c + mlx-swift** (identity + ownership model across 9 primitives + exposure to Swift). Before committing, worth validating the 1.40× target holds at a smaller scale first.

### Proposed next step — sized for a single session

1. Build a new benchmark method `--method icb-decode` that **fakes correctness** by pre-computing the reference answer outside ICB and using ICB replay for timing only. No token-correctness check; measures whether the encoding-cost win translates to actual wall-time decode tok/s when the replay actually runs in the generation path.
2. If the measured wall-time win is ≥1.25×, commit to Option A.
3. If it's <1.15×, the ceiling is lower than expected and we should reconsider scope (maybe only fix the decode-layer call sites that dominate, not all 9 primitives).

This gets us a data-driven answer for ~half a session of work without committing to the multi-week Option A implementation.

## Process note — mlx-swift submodule bump needed

To test Phase 2 through Swift, the mlx-swift submodule must point at `ek/sdpa-option-c` (pre-Phase-2 tip `4648b89c` is what's shipped on `ek/metal-icb-prototype` there today). Steps:

1. In `/Users/eric/Development/personal/ai/mlx-swift/Source/Cmlx/mlx`: `git fetch <mlx path> ek/sdpa-option-c && git checkout FETCH_HEAD`.
2. In `/Users/eric/Development/personal/ai/mlx-swift`: `./tools/fix-metal-includes.sh` to sync pre-generated metal files (picks up `sdpa_unified.h` and new `scaled_dot_product_attention.metal` instantiations automatically).
3. In mlx-swift-lm: `swift build -c release`.

Alternative when stale build suspected: `make clean-all` then `make MLX_SWIFT_PATH=/Users/eric/Development/personal/ai/mlx-swift` per the Makefile's local-dev flow.

Submodule bump has not been committed to mlx-swift — local on disk only for the measurement above.

## Numbers summary

- Phase 2 correctness holds end-to-end on GPT-OSS-20B (no crashes, 9-primitive AB stack runs clean).
- SDPA setBytes arena = 0 bytes, confirmed via regression gate.
- AB-only decode win on GPT-OSS-20B: +1–2% (noise-floor territory).
- ICB encoding-cost win: **1.40×** (microbench, no cache advancement).
- Decode-loop ICB wiring is the next work item; Option A is the proposed approach, but a `--method icb-decode` validation pass is the cheap first step before committing to the full refactor.

## Cross-branch baseline — alpha vs ek/metal-icb-prototype

Measured after pushing Phase 2 to ekryski/{mlx, mlx-swift, mlx-swift-lm} and switching to alpha for a clean comparison. 3 runs per config, GPT-OSS-20B 4-bit, 200 max-tokens, 101-token prompt.

| Branch | Config | Run 1 | Run 2 | Run 3 | Mean | Δ vs alpha |
|---|---|---:|---:|---:|---:|---:|
| **alpha** | --kv none | 48.6 | 47.8 | 47.2 | **47.9** | baseline |
| **alpha** | --kv turbo4v2 | 48.4 | 47.3 | 47.7 | **47.8** | −0.2% |
| ek/metal-icb-prototype | AB off, --kv none | 46.2 | 45.7 | 45.8 | 45.9 | **−4.2%** |
| ek/metal-icb-prototype | AB=1 (8 prims, legacy SDPA), --kv none | 47.1 | 46.9 | 46.5 | 46.8 | −2.3% |
| ek/metal-icb-prototype | AB=1 (Phase 2, 9 prims), --kv none | 46.8 | 46.3 | 46.1 | 46.4 | −3.1% |
| ek/metal-icb-prototype | AB=1 (Phase 2, 9 prims), --kv turbo4v2 | 46.2 | 46.4 | 47.1 | 46.6 | −2.7% |

**Pre-existing regression on ek/metal-icb-prototype: ~4% vs alpha even with AB off.** This is not caused by SDPA Option C — the AB-off measurement on this branch is still 2 tok/s below alpha. Likely sources (not yet bisected):

- ICB pipeline-support infrastructure (`setSupportIndirectCommandBuffers(true)` PSOs compile and dispatch differently than vanilla mlx — though device.cpp claims no cost in default state)
- Larger metallib / more kernel variants in the AB-migrated primitives even when their runtime gate is off
- Other changes that landed on ek/metal-icb-prototype between its base-point and alpha's tip

**AB=1 on ek/metal-icb-prototype claws back ~1% of the gap but doesn't close it.** The branch is ~3% behind alpha even with all 9 AB primitives active.

### Implication for decode-loop ICB work

Option A (persistent per-call-site AB + ICB record/replay in decode) must deliver enough wall-time win to (a) close this ~3% gap and (b) clear alpha by a margin that actually justifies the refactor cost. The 1.40× encoding microbench sets an upper bound on the CPU-side improvement; how much of it translates to wall-time depends on what fraction of per-step time is actually CPU encoding on this model.

### Follow-on: bisect the pre-existing regression

Worth a separate session: `git bisect` between alpha's tip and ek/metal-icb-prototype's base point to identify the commit that introduced the 4% regression. Fixing it before layering on decode-loop ICB would give Option A a cleaner starting point to measure against.

## Bisect results (2026-04-18)

Did the bisect. Per-layer + per-mlx-SHA measurements (n=3 each, median reported):

| Config | Decode tok/s | Δ vs alpha |
|---|---:|---:|
| alpha (all three repos) | 47.9 | baseline |
| mlx-swift-lm ek + mlx-swift alpha + mlx 56c79931 | 47.4 | −1.0% (noise) |
| mlx-swift-lm ek + mlx-swift ek + mlx 56c79931 (alpha ref) | 47.1 | −1.7% (noise) |
| ek-all-three + mlx at defb3141 | 46.6 | −2.7% |
| ek-all-three + mlx at 466258cf | 47.2 | −1.5% (noise) |
| ek-all-three + mlx at 4648b89c (pre-Phase 0) | 47.3 | −1.3% (noise) |
| ek-all-three + mlx at 3b026c3e (tip, Phase 2) | 45.9 | **−4.2%** |

Layer isolation:

- **mlx-swift-lm layer**: ~0.5 tok/s loss (within noise)
- **mlx-swift layer**: ~0.3 tok/s loss (within noise)
- **mlx, alpha→pre-Phase-0 (4648b89c)**: ~0 net — the 40 ICB+AB commits below my work wash out
- **mlx, Phase 0→Phase 2 (my 5 commits)**: ~**−1.4 tok/s** — accounts for the bulk of the regression

### The surprising finding

My Phase 0-2 commits contribute the largest slice of the regression (~1.4 tok/s / ~3% of the budget) even though they're additive and the new code paths are only exercised under `MLX_METAL_AB=1`. Hypothesis (not yet verified):

- **Metallib growth**: Phase 2 adds ~30 new PSO instantiations (`sdpa_unified_vector_*` + `sdpa_unified_vector_ab_*` across fp32/fp16/bf16 × 5 head-dim values). The metallib is measurably larger. Bigger metallib → more PSOs for the Metal driver to manage / index / populate residency sets for.
- **SDPA dispatch branch**: the new `if (ab_enabled() && !sdpa_force_legacy())` check runs on every SDPA call, but `ab_enabled()` is cached via static-local init so should be ≈free. Possibly the `!sdpa_force_legacy()` path isn't cached — that `getenv` runs every call.

### What to do about it

`sdpa_force_legacy()` should be a cached static too — trivial fix; let me address that as part of the Option A work since I'm about to edit that file anyway.

Metallib growth is harder to fix without shipping the Phase 2 kernel. Short of dropping Phase 2, the options are:

- **Instantiate fewer SDPA variants up-front**: only emit the D=64/96/128 set for Phase 2 AB, defer 256/512 to JIT. Speculative; unclear if this actually measurably helps.
- **Accept the cost**: 3% is the price for the correctness unlock. Once ICB decode-loop is live, the 1.40× replay speedup should more than recover it.

### Conclusion

The regression is distributed and largely comes from my own Phase 0-2 work (not from the other commits on `ek/metal-icb-prototype`, which roughly break even vs alpha). Further bisecting below the commit level isn't going to find a single-line fix.

**Plan forward**: do the `sdpa_force_legacy()` caching fix inline as part of starting Option A. Otherwise proceed to Option A implementation, accepting the ~3% regression as the cost of Phase 2's architectural landing. The 1.40× encoding microbench says Option A has room to clear it.
