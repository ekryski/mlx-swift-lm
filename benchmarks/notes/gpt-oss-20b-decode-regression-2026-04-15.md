# GPT-OSS-20B Decode Regression — Post-Mortem

**Date:** 2026-04-15
**Branch:** `ek/p0-gpt-oss-decode-regression`
**Severity:** P0 — blocked all Phase A–E work per `benchmarks/notes/final-performance-plan-04-15-2026.md`
**Status:** Fixed

## Summary

GPT-OSS-20B 4-bit turbo4v2 decode throughput on M1 Max collapsed from ~50 tok/s to ~3.6 tok/s (a **-93% regression**) between commits `0445a6d` and `062a628`. The cause was **debug logging accidentally committed in `5cf64e8`** ("Fix native prefill bridge for dense and MoE models") — an `eval()` + `.item()` GPU→CPU sync chain inside `GPTOSS.swift`'s attention hot path, gated by a counter that never stopped the logging.

Fix: delete the dead debug code (`-32/+0` lines in `Libraries/MLXLLM/Models/GPTOSS.swift`).

## Reproduction

Controlled conditions: `MLX_SWIFT_PATH=/Users/eric/Development/personal/ai/mlx-swift` (pinned to `1d11c3d`) — identical mlx-swift source for every run. Model: GPT-OSS-20B 4-bit, `--kv turbo4v2 --method summarization --context 128,1024`, N=3 runs per commit.

| Commit | 128 ctx median (worst) | 1024 ctx median (worst) | Prefill 128 ctx |
|--------|:-:|:-:|:-:|
| `0445a6d` (good, 2026-04-13) | 50.4 (50.4) tok/s | 50.2 (49.8) | 293–376 tok/s |
| `062a628` (regressed, 2026-04-15) | 3.6 (3.5) | 3.6 (3.5) | 113–119 |
| **After fix (this branch)** | **49.8 (49.2)** | **49.8 (49.2)** | 316–376 |

Gemma4 E2B negative control (1 run, post-fix): 99.2 tok/s @ 128 ctx, 92.0 @ 1024 ctx — healthy, matches pre-regression expectations.

**Delta from plan-doc baseline (62.1 tok/s @ 128 ctx).** Today's `0445a6d` measures ~50 tok/s rather than 62.1. This ~19% shortfall is *not* part of this regression — it's a separate mlx-swift drift between `79a1dcd` (when the plan doc measured baseline) and the current `1d11c3d`, which includes `923e61d perf: enable NAX Metal dispatch`. Tracked as a follow-up.

## Investigation

Followed the P0 plan at `~/.claude/plans/greedy-snacking-barto.md`:

1. **Step 1 reproduce baseline (N≥3).** Confirmed 93% regression, tight distributions on both commits (<3% spread), zero run-to-run variance in my environment. The plan-doc's observed "30% variance" on `a744cb5` was environmental (thermal/memory pressure in the prior session), not intrinsic. Controlled reproduction removed it.

2. **Step 3 targeted revert.** The plan's leading hypothesis was `062a628`'s `ensureBuffer` pre-allocation in `RotatingKVCache.updateMultiToken`. I reverted the `totalLen < maxCacheSize` branch back to the pre-062a628 form (`self.keys = fullK.contiguous()`). Result: decode still 3.5–3.8 tok/s across 3 runs — **no change**. `062a628`'s KVCache changes were definitively ruled out.

3. **Step 4 bisect.** `git diff --stat 0445a6d..062a628` surfaced that `Libraries/MLXLLM/Models/GPTOSS.swift` had `+86/-1` lines across two commits: `5cf64e8` (the primary change) and `8dee146` (NATIVE_PREFILL opt-in, already ruled out). Reading `5cf64e8`'s GPT-OSS diff revealed the debug logging.

## Root cause

Commit `5cf64e8` added **four** blocks of debug logging in `GPTOSS.swift`:

| Block | Location | Guard | Per-token cost |
|-------|----------|:-:|:-:|
| pre-RoPE Q/K log | `AttentionBlock.callAsFunction` ~line 223 | `blockLogCount <= 1` | eval×2 + .item()×4 + stderr |
| post-RoPE Q/K log | `AttentionBlock.callAsFunction` ~line 239 | `blockLogCount <= 1` | eval×2 + .item()×4 + stderr |
| SDPA-out log | `AttentionBlock.callAsFunction` ~line 252 | `blockLogCount <= 1` | eval×2 + .item()×3 + stderr |
| FWD L0 post_attn | `GPTOSSTransformerBlock.callAsFunction` ~line 332 | `blockLogCount == 0` | one-shot |

The logic bug:

```swift
static var blockLogCount = 0

// In GPTOSSTransformerBlock.callAsFunction:
if Self.blockLogCount == 0 {
    // ... log FWD L0 ...
    Self.blockLogCount += 1   // ← ONLY incrementer
}

// In AttentionBlock.callAsFunction (all three blocks):
if GPTOSSTransformerBlock.blockLogCount <= 1 {
    // ... log Q/K/SDPA with eval+.item() ...
    // ← no increment; guard is permanently true after first pass
}
```

Timeline per run: first pass sets `blockLogCount = 1` from the `== 0` branch. After that, the `== 0` branch is skipped (the FWD L0 log does stop). But the three `<= 1` blocks inside `AttentionBlock` **never increment the counter**, so their guard `1 <= 1` stays true forever. GPT-OSS-20B has 24 transformer blocks; every decode token fires 24 × 3 = **72 per-token GPU→CPU sync boundaries**. Each `.item(Float.self)` forces a stream synchronization and blocks the GPU from pipelining the next kernel. At ~50 µs per sync × 72 syncs × 400 tokens, decode becomes CPU-sync-bound instead of bandwidth-bound.

The bounded counterpart — `fwdCount < 2` logging in `GPTOSSModelInner.callAsFunction` (lines 359–419) — was also added in `5cf64e8` but is self-limiting: `fwdCount` increments every forward, so after 2 passes all the `doLog`-gated logging stops. Perf-neutral on any run >2 tokens. Left in place for now (follow-up cleanup below) to keep the P0 diff minimal.

## Fix

Delete the four unbounded debug blocks and the `static var blockLogCount`:

```
 Libraries/MLXLLM/Models/GPTOSS.swift | 32 --------------------------------
 1 file changed, 32 deletions(-)
```

The fix makes no behavioral change — the debug code was pure telemetry into stderr, not used by any consumer. All `062a628` regression tests pass (`testRotatingCache*`, `testSingleTokenDecodeConsistency`, `testRotatingCachePrefillThenDecodeDoesNotCorrupt`, etc.).

## Why the plan's leading hypothesis was wrong

The plan pointed at `062a628` because:
- It was the most recent commit before the regressed benchmark.
- Its commit message "fixing buffer overflow bug in rotating KV cache" suggested KV-cache-adjacent perf risk.
- It landed 1,011 lines of new tests, suggesting the fix had teeth.

What was missed: the `ensureBuffer` + prefix-fill change only affects `updateMultiToken`'s `totalLen < maxCacheSize` branch. For GPT-OSS-20B sliding-attention layers with `maxCacheSize=128` under the 128-ctx summarization benchmark, `totalLen = 128` — the `>=` branch runs, which was **unchanged**. A `git diff --stat 0445a6d..062a628` earlier would have surfaced the `+86/-1` GPTOSS.swift change (from `5cf64e8`, committed 2 days before `062a628`) and redirected the investigation immediately. Lesson: before committing to a suspect based on commit-message reading, diff-stat the full range for files in the regressed model's path.

Also, the plan doc described the regression as affecting both NATIVE_PREFILL=0 and =1 equally. That observation should have been a stronger signal that the cause was *in the model code itself* (a decode-path concern), not in the native-bridge adapter layer — which is what most of the commits between `0445a6d` and `062a628` are.

## What the fix does not cover — follow-up tickets

1. **Remove bounded `fwdCount < 2` debug logging in `GPTOSSModelInner.callAsFunction`** (lines 359–419 and `static var fwdCount`). Same commit origin (`5cf64e8`). Perf-neutral past the first 2 forward passes, so not blocking P0. Cleanup PR against `ek/tom-eric-moe-tuning`.

2. **`testFusedPrecisionDiagnostic` failing on `main` / current branch tip.** Gemma4 fused RMSNorm+RoPE kernel produces `normMaxDiff = 0.03125` vs reference; test threshold is `< 0.01`. Fails on unmodified `062a628` (not caused by this fix). Warrants its own investigation — the test comment warns: "even small per-element diffs compound over 30+ layers." Unclear whether this is a real correctness concern or a too-tight test threshold.

3. **`testCacheSerialization` fatal crash** when run under the broad `KVCacheTests|Gemma4Tests` filter. Error: `[broadcast_shapes] Shapes (1,8,6,64) and (1,8,4,64) cannot be broadcast` during what appears to be a concurrent TurboQuant serialization test. Reproducible on unmodified code. Likely concurrency / test-interaction bug rather than a product bug. Investigate and isolate the offending test case.

4. **mlx-swift `1d11c3d` vs `79a1dcd` 19% decode regression on GPT-OSS-20B.** Plan-doc baseline at `0445a6d` + mlx-swift@79a1dcd was 62.1 tok/s; today at `0445a6d` + mlx-swift@1d11c3d is 50.4 tok/s. Two mlx-swift commits in the gap: `923e61d perf: enable NAX Metal dispatch` and `1d11c3d` (merge). NAX is M5-only; on M1 Max the guard should make the new code path inert. Bisect mlx-swift separately to confirm or isolate.

5. **`NATIVE_PREFILL=1` slower than `=0` on GPT-OSS-20B** (P0-followup in the plan doc). Separate code path. File a ticket; do not fold into this fix.

6. **Per-commit GPT-OSS smoke-benchmark in CI** (plan doc P0.5 second half). This is the second recent instance of GPT-OSS decode regressing silently in a short window. A minimal 128/1024 ctx smoke check per PR would have caught this within hours instead of days. Needs a stable benchmark runner and a variance budget; Phase-A infra work.

## Signal for future investigations

- **Tight run-to-run variance in one environment and loose variance in another is information.** The plan doc reported ~30% same-commit variance on `a744cb5`; my controlled reproduction showed <3% on the same commit. That gap tells you the earlier environment had a confound (thermal, another process, memory pressure) — not that the code is noisy.
- **`git diff --stat` across the full regression range is usually the fastest first investigative step** even when you have a leading hypothesis. It costs nothing, fits in one screen, and would have surfaced the GPTOSS change immediately.
- **Debug code that writes via `.item(Float.self)` or `eval()` inside any per-token loop is a GPU-sync tax that will never show up in a linter.** Worth a CI grep for `\.item\(.*\)` or `eval\(` inside hot-path `.swift` files under `Libraries/MLXLLM/Models/`.
