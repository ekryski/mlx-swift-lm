# Final Performance Plan — 2026-04-15

> Audit + implementation plan for the next round of perf work. No code changes accompany this note — implementation happens in follow-up sessions on feature branches off `ek/tom-eric-moe-tuning`.

## Context

Picking up from the 2026-04-07 plan after two weeks of perf + safety work. Goals today:

1. Audit memory-leak and thread-unsafe MLX-array concerns against the current state of all four repos.
2. Re-verify every item in the "still open" list against actual code — several were already tried-and-regressed *or* already implemented.
3. Decide what remains that's worth chasing. Downstream sequence: ~~fix P0 GPT-OSS-20B decode regression~~ ✅ done → **resolve P0-B Qwen3.5 prefill regression (§0.5)** → graph-caching re-evaluation + Metal ICB experiment → batch decode → n-gram speculative → diffusion speculative.

All benchmark numbers cited are **M1 Max 64 GB**. M5 Max numbers are excluded from headline comparisons.

Repos audited:

| Repo | Path | Branch |
|------|------|--------|
| `mlx-swift-lm` | `/Users/eric/Development/personal/ai/mlx-swift-lm/` | `ek/tom-eric-moe-tuning` |
| `mlx-swift` | (dependency / local) | `ek/speed-improvements-2` |
| `mlx` (submodule) | under mlx-swift | `ek/perf-improvements` |
| `mlx-c` (submodule) | under mlx-swift | `ekryski/perf-improvements` |

---

## 0. Priority Zero — GPT-OSS-20B Decode Regression (~92%) — ✅ **COMPLETE**

**Resolved 2026-04-15** by `5002e44 fix: restore GPT-OSS-20B decode perf by removing leftover debug logging` (merged in [ekryski/mlx-swift-lm#29](https://github.com/ekryski/mlx-swift-lm/pull/29)). Root cause was leftover debug logging inside `GPTOSS.AttentionBlock.callAsFunction` committed accidentally by `5cf64e8`: three `.asType(float32).reshaped(-1) + eval() + MLX.sum(.abs()).item() + stderr.write()` blocks per attention call (pre-RoPE Q/K, post-RoPE Q/K, SDPA output) plus one block + `static blockLogCount` counter — each block forced a GPU→CPU sync per call. Bisect of `062a628` / `ensureBuffer` / `NATIVE_PREFILL` was a dead end; the regression was purely the sync-per-dispatch log spam.

### Verification (M1 Max, 2026-04-15 13:57, commit `5002e44`)

| Config | Context | Prefill tok/s | Decode tok/s | vs. 04-13 baseline |
|--------|:--:|:--:|:--:|:--:|
| turbo4v2 | 128 | 376.2 | 59.8 | prefill -3%, decode -3.7% |
| turbo4v2 | 1024 | 610.8 | 57.5 | decode -3.8% |
| turbo4v2 | 4096 | 689.1 | 56.4 | decode +2.9% |
| turbo4v2 | 32768 | 544.3 | 38.9 | decode -2.3% |

Source: `benchmarks/gpt-oss-20b/gpt-oss-20b-4bit-turbo4v2-summarization-benchmark-2026-04-15-1357.md`.

**Success gate status:** decode target was ≥ 55 tok/s @ 128 ctx turbo4v2. Delivered 59.8 tok/s. ✅

**Residual gap — P0-followup (non-blocking):** we're still ~3-4% below the 04-13 baseline on decode at short contexts and -3% on prefill at 128 ctx. Short-list of possibilities worth a cheap A/B before Phase A starts:
- Tom's `withGenerationStream` wrapper still wraps each GPT-OSS forward call — confirm via trace that the stream is actually warm (first-token overhead may be eating the short-context delta).
- `062a628` `ensureBuffer` pre-allocates the full `[B, H, maxCacheSize=128, D]` RotatingKVCache buffer on the first multi-token update — for GPT-OSS with `maxCacheSize=128` this is small in absolute terms but adds one allocation on the first token that wasn't there before. Re-profile if the 3% matters.
- `NATIVE_PREFILL=1` is slower than `=0` on GPT-OSS today (tracked as P0-followup in earlier version of this note). Separate issue from P0; still open.

### Original investigation (preserved for history)

### Evidence (M1 Max, GPT-OSS-20B 4-bit, no `--ppl --kld --think`)

| Config | Date | Commit | 128 ctx | 1024 ctx | 4096 ctx | 32768 ctx |
|--------|------|--------|:--:|:--:|:--:|:--:|
| turbo4v2 (baseline) | 2026-04-13 12:09 | `0445a6d` | **62.1** | 59.8 | 54.8 | 39.8 |
| turbo4v2 (regressed) | 2026-04-15 11:15 | `062a628` | **4.9** | 4.6 | 4.5 | 4.1 |
| no-quant (regressed) | 2026-04-15 11:28 | `a744cb5` | 5.6 | 5.3 | 5.2 | 3.9 |
| no-quant (regressed) | 2026-04-15 11:37 | `a744cb5` | **3.9** | 4.0 | 3.6 | 3.9 |

Sources: `benchmarks/gpt-oss-20b/gpt-oss-20b-4bit-turbo4v2-summarization-benchmark-2026-04-13-1209.md`, `-2026-04-15-1115.md`, `benchmarks/gpt-oss-20b/gpt-oss-20b-4bit-no-quant-summarization-benchmark-2026-04-15-1128.md`, `-1137.md`.

**Magnitude: -92% decode at 128 ctx, -91% at 4k, -90% at 32k.** Prefill on turbo4v2 also regressed (136.2 vs 387.9 at 128 ctx = -65%). Short-context decode regression (where this should be most bandwidth-bound) is the worst — a strong hint that something in the per-token CPU path or KV-cache init is the culprit, not a GPU-compute issue.

Two consecutive runs on the same commit (11:28 vs 11:37, both `a744cb5`) disagree by ~30% at 128 ctx — 5.6 → 3.9 — so there's also run-to-run instability on top of the regression. Suggests thermal / memory-pressure / something-else-on-the-machine noise sitting on top of a hard regression. Repeat each bisect step.

### Sanity checks already run

- **NATIVE_PREFILL=0 vs =1**: tested. Both regressed; `NATIVE_PREFILL=1` is **even slower**. Conclusion: the native-bridge opt-in toggle (`8dee146`) is not the primary cause of the GPT-OSS decode regression. But it does mean the native bridge itself has its own perf issue on GPT-OSS that is separate and worth fixing after P0 lands — track as P0-followup below.

### Bisect candidates (commits touching GPT-OSS decode hot path between `0445a6d` and `062a628`)

```
062a628  fixing buffer overflow bug in rotating KV cache   ← primary suspect
5cf64e8  Fix native prefill bridge for dense and MoE models
8dee146  config: make native prefill bridge opt-in (NATIVE_PREFILL=1)   ← ruled out as primary
```

`8dee146` is down-ranked because the regression reproduces with the bridge off. `062a628`'s `ensureBuffer()` change is the strongest candidate — it now pre-allocates the full `[B, H, maxCacheSize, D]` buffer *and* clamps trim idx via `max(0, idx - trimmed)`. For GPT-OSS with `maxCacheSize = 128` (RotatingKVCache), the write pattern on decode changed: every decode token writes at `idx`, then reads back the full 128-slot buffer. If that read got a different memory-layout or dispatch path than before, we'd see exactly this shape of regression (bandwidth-bound decode, unchanged GPU baseline memory, unchanged prefill-compute share).

Indirect suspects worth checking only if the three above don't account for it:
- `00547bc perf: dlopen bridge + NAX enable` — even with NATIVE_PREFILL=0, something about the dlopen path could have changed stream / scheduler defaults.
- `d406cfe fix: restore .contiguous() on bridge KV injection` — bridge-only on paper, but verify.

### Priority Zero plan

**P0.1 — Reproduce and baseline**
- Re-run 2026-04-13-1209 config on `0445a6d` to confirm ~60 tok/s on current hardware. This rules out thermal/background noise (note the 11:28 vs 11:37 same-commit 30% delta — run each benchmark twice).

**P0.2 — Trace before bisect** (cheap, high-signal)
- Dump a single-token decode trace with `MLX_METAL_TRACE=1` on `062a628` vs `0445a6d`. Compare:
  - Per-token dispatch count — if `062a628` shows a massive dispatch explosion (e.g. per-token Metal pipeline rebuild or re-encoded TurboQuant kernels), that's the regression class.
  - Kernel names in the trace — look for reintroduction of a JIT kernel or a buffer-alloc kernel that wasn't there on the baseline.
- Check `GPU Baseline` memory across contexts: both `0445a6d` and `062a628` show `10.41GB` → not a weight-loading miss. Pure runtime path regression.

**P0.3 — Bisect**
- `git bisect` with `0445a6d` as good, `062a628` as bad, test = `benchmark.sh --model gpt-oss-20b --quant 4bit --kv turbo4v2 --method summarization --context 128,1024`. Bisect criterion: decode < 20 tok/s at 128 ctx = bad.
- Start the bisect at `062a628`: attempt a targeted revert of *just* the `ensureBuffer()` call in `updateMultiToken` (keep the `trim()` idx clamp) and re-run. If that restores decode, the `ensureBuffer` pre-allocation is the cause and we need a GPT-OSS-aware fix (maybe skip pre-allocation when `maxCacheSize` is small).

**P0.4 — Fix + verify**
- Once identified, choose: revert, targeted fix, or env-flag gate.
- Safety regression suite must pass: `KVCacheTests`, `Gemma4Tests`, `SpeculativeDecodingTests` — these were added with `062a628`, so we cannot naively revert it. A targeted fix that keeps the test-validated safety behavior but avoids the perf regression is the goal.
- Re-run GPT-OSS and Gemma4 benchmarks to confirm no collateral regression on the fix.

**P0.5 — Post-mortem** at `benchmarks/notes/gpt-oss-20b-decode-regression-2026-04-15.md` regardless of outcome. This is the second recent instance of GPT-OSS decode regressing silently (per earlier session notes); it warrants a standing regression check in the benchmark CI — at minimum, a per-commit smoke benchmark of GPT-OSS decode at 128/1024 ctx.

**P0-followup (after P0 merges) — native bridge perf on GPT-OSS**
- NATIVE_PREFILL=1 is slower than NATIVE_PREFILL=0 on GPT-OSS today. This is a separate issue from P0 but directly degrades any user who flips the flag on.
- Profile `NATIVE_PREFILL=1` GPT-OSS prefill + decode to identify the extra cost (stream sync? contiguous copy? bridge/MLX boundary?).
- Decide: fix the native-bridge path for GPT-OSS, or gate NATIVE_PREFILL=1 away from GPT-OSS until the perf gap closes.

### Success gates

- Decode ≥ 55 tok/s at 128 ctx turbo4v2 on M1 Max (within 10% of 04-13 baseline of 62.1).
- No regression on Gemma4 E2B / Qwen3.5-35B decode.
- Post-mortem written.

---

## 0.5 Priority Zero-B — Qwen3.5 Prefill Regressed ~15× vs. 2 Weeks Ago

**New P0 surfaced after GPT-OSS fix landed.** Qwen3.5 hybrid models (0.8B dense representative; also affects the 35B-A3B MoE) have lost **~84-94% of prefill throughput** since 2026-04-02. This is not a subtle regression and not caused by `max_ops_per_buffer`; it's the accumulated cost of two **intentional memory-tradeoff commits** (`2a695c0` undo + `0445a6d`) plus a small additional hit from the `max_ops` bump.

### Evidence (M1 Max, Qwen3.5-0.8B bf16, turbo4v2, PPL/KLD ON in both runs)

| Context | 2026-04-02 (pre-tradeoff) | 2026-04-16 (current) | Prefill Δ | 32K Peak |
|---|:--:|:--:|:--:|:--:|
| 128 | 1,064 | 238 | **-78%** | — |
| 1024 | **3,246** | 232 | **-93%** | — |
| 4096 | **4,090** | 242 | **-94%** | 1.81 GB (vs 3.43 GB: -47%) |
| 32768 | **3,448** | 230 | **-93%** | 2.44 GB (vs 4.82 GB: -49%) |
| 131072 | 1,230 | 193 | -84% | 4.75 GB (vs 8.29 GB: -43%) |

Sources: `benchmarks/qwen3.5-0.8b/qwen3.5-0.8b-bf16-turbo4v2-summarization-benchmark-2026-04-02-2341.md` vs `-2026-04-16-0257.md`.

**Observation**: April-2 prefill *scales with context size* (1k → 4k → 32k sees 3.2→4.0→3.4 thousand tok/s), i.e. GPU batching is working and amortizing dispatch. Current prefill is **flat ~230 tok/s regardless of context**, meaning the chunked prefill + per-layer eval path has serialized work into ~512-token batches with an eval sync between every layer. Throughput is now bounded by sync count, not GPU.

The 4-bit benchmarks tell a gentler version of the same story because 4-bit peak memory never ballooned to begin with (1.1 GB at 32k), so the tradeoff isn't visible as strongly — the 17-24% prefill drop on 4-bit is the `max_ops` bump layered on top of the already-reduced absolute throughput.

### Root cause — three changes stacked between 04-02 and 04-16

1. **`2a695c0` (2026-04-04, "perf: 5.7x prefill speedup")** — added `prefillStepSize = max(windowSize ?? 512, 4096)` to `Qwen35Model.prepare()`. Forced 4096-token minimum chunks. Commit msg: "Context 1024 prefill: 84.4 → 478.2 tok/s (5.7x), TTFT: 12.5s → 2.4s". This was **the** prefill win for Qwen3.5 hybrids.
2. **`0445a6d` (2026-04-13, "perf: fix prefill memory bloat for SSM/GDN hybrid models")** — **reverted** 2a695c0's 4096 floor (`prefillStepSize = windowSize ?? 512`), added per-layer `eval(hiddenStates, c)` during prefill, added per-chunk `MLX.Memory.clearCache()`. Commit msg: "Results (Qwen3.5 0.8B, 4096 ctx): peak 5.59GB → 1.34GB (-76%)". Memory win; prefill cost implicit.
3. **`max_ops_per_buffer_ = 100 → 200`** (working tree as of 04-16, now committed in mlx at `d5e8c0de` on `ek/qwen35-prefill-speedup` — see below). Additional +~17-24% prefill hit on Qwen3.5 (but +3-5% decode on Gemma4 and pure-attention models). Net-neutral across the fleet; net-negative on Qwen3.5.

**Effective current prefill path (Qwen3.5 hybrid, T tokens)**:
- Chunk loop in `Qwen35.swift:728-737`: `chunkSize = min(prefillStepSize, T-1)` with `prefillStepSize = windowSize ?? 512` at line 722 (no floor — today's benchmarks pass `windowSize=2048`, so chunks are 2048; a Swift caller that omits windowSize gets 512-token chunks). eval(cache) + clearCache() between chunks → each chunk is a hard GPU/CPU sync.
- Inside a chunk, `Qwen35TextModelInner.callAsFunction` at `Qwen35.swift:593-596` does `eval(hiddenStates, c)` **after every single layer** (24 layers for 0.8B). Each is another full CPU→GPU sync.
- Net: for a 4096-token prompt at chunk=2048, prefill completes ~2 chunks × 24 layers = **48 per-layer syncs** before any tokens are decoded. Pre-0445a6d the same prompt had ~1 sync (end of prefill).

The per-layer eval is the dominant cost. The chunk-size shrink is the secondary cost. `max_ops_per_buffer` is a distant third on Qwen3.5 specifically.

### Decisions already taken this session

- **`max_ops_per_buffer = 200` is now committed**: mlx `d5e8c0de` on `ek/qwen35-prefill-speedup`, with the tradeoff documented in the commit body. Confirms the user's intent ("we should be using 200") and ends the uncommitted-working-tree ambiguity. mlx-swift's submodule pointer needs to follow (see P0-B.5 below).

### Plan — Priority Zero-B (revised)

The honest framing is: we have a stack of three deliberate memory wins that, collectively, cost us ~93% of prefill. The user wants the speed back. We need to unwind selectively, keeping the memory wins that matter at long context and dropping the ones with the worst speed cost.

**P0-B.1 — Quantify each layer of the stack** (fast, no code changes)
- Bench matrix: `{100, 200}` × `{per-layer eval on, off}` × `{chunk=512, 2048, 4096, 8192}` for Qwen3.5-0.8B bf16 turbo4v2 at 1k / 4k / 32k / 131k ctx. 32 runs. 
- Output: a table of (prefill tok/s, peak GPU) per cell. Fill in `benchmarks/notes/qwen35-prefill-tradeoff-matrix-2026-04-XX.md`.
- This replaces guesswork with data. We will probably find that per-layer eval is the single biggest cost at short context, and chunk size the biggest cost at long context.

**P0-B.2 — Land the obvious fix: decouple chunk size from memory policy**
- Today, 2a695c0's `max(windowSize ?? 512, 4096)` floor is gone. That single line is worth most of the "5.7x prefill speedup" commit. Restoring **a floor** (not forcing 4096 unconditionally — floor) recovers a large fraction at minimal memory cost:
  ```swift
  let prefillStepSize = max(windowSize ?? 512, 2048)  // or 4096
  ```
  With the chunk floor and per-layer eval still enabled, peak memory stays bounded (per-layer eval still clamps activation bloat *within* a chunk) but sync count drops by 2-8x.
- Guard with an env flag during bring-up (`MLX_QWEN35_PREFILL_CHUNK_FLOOR=4096`) so the matrix from B.1 can tune it cleanly before becoming the default.

**P0-B.3 — Per-layer eval: narrow its scope**
- Per-layer eval is defensible when chunk is *huge* (e.g. 131k in a single chunk would blow memory). It is wasteful at chunk=2048 for a 0.8B model. Two options:
  - (a) Gate per-layer eval behind chunk size: `if isPrefill && chunkSize >= 4096`.
  - (b) Eval every N layers instead of every layer (e.g. N=4 or N=8). Turns 24 syncs into 3-6.
- (b) is what mlx-swift-lm's own bridge already does (`9cdd178 Bridge: batch eval every 4 layers instead of per-layer (2x faster prefill)`). Same idea, applied to the Swift prefill path.
- Expected gain: **2-5x prefill** at 1k-32k ctx with ≤ +20% peak GPU. Matrix in B.1 sizes this.

**P0-B.4 — MoE verify (Qwen3.5-35B-A3B)**
- Same chunk-floor + every-N-layer eval change tested on the MoE. The MoE has different peak behavior (gatherQuantizedMM creates its own intermediates), so peak gain has to be re-measured.
- Acceptance: Qwen3.5-35B-A3B prefill at 4k ctx ≥ 1,500 tok/s (vs April-8 baseline of ~4,400; vs today's 85.8). We're not expecting to fully recover the April-8 numbers because some of the April-8 speed came from the native prefill bridge (now opt-in); the gates here are "enough of it back that the user doesn't feel a regression".

**P0-B.5 — Move mlx-swift submodule pointer forward**
- mlx `ek/qwen35-prefill-speedup` now has the `max_ops=200` commit (`d5e8c0de`). mlx-swift currently pins `690cf46b` (pre-commit).
- Bump the mlx-swift submodule pointer on branch `ek/speed-improvements-2` to `d5e8c0de` and commit. Small, no behavior change from current benchmark runs (they already build against this file as a working-tree edit).
- Also update §4 "Items Already Implemented" row: remove the ⚠️ status; `max_ops_per_buffer=200` for Max/Ultra is now the committed default, with the tradeoff documented in the mlx commit body.

**P0-B.6 — Adaptive command-buffer commit threshold (§5 A2) — keep in backlog, not P0-B**
- The real long-term fix for the Qwen3.5-vs-Gemma4 tension is MB-aware commits per `spec-adaptive-command-buffer-management.md`. That work is still the right move, but it's Phase A2 — do not block P0-B on it.

### Success gates (P0-B)

- Qwen3.5-0.8B bf16 turbo4v2 prefill at 4k ctx ≥ **1,500 tok/s** (vs current 242, vs April-2 4,090). Full recovery is not the goal; recovering >50% of the lost prefill is.
- Qwen3.5-0.8B bf16 32K ctx peak GPU ≤ **3.5 GB** (vs current 2.44, vs April-2 4.82). Some peak regression is acceptable; the April-13 memory win was worth ~2 GB and we want to keep most of it.
- Qwen3.5-35B-A3B 4-bit turbo4v2 prefill at 4k ctx ≥ **1,500 tok/s** (vs current 85.8).
- **No regression on Gemma4 E2B decode/prefill, GPT-OSS-20B decode/prefill, Nemotron prefill memory.** The whole reason the memory tradeoffs were made was to un-OOM these sibling hybrid models.

### Verification artifacts to write

- `benchmarks/notes/qwen35-prefill-tradeoff-matrix-2026-04-XX.md` — P0-B.1 data.
- `benchmarks/notes/qwen35-prefill-recovery-2026-04-XX.md` — post-fix benchmarks and analysis, with a direct April-2 vs post-fix comparison table matching this section.

---

## 1. Safety Audit — Memory Leaks and Thread-Unsafe MLX Arrays

### Memory leaks — all known issues fixed

| Issue | Status | Evidence |
|-------|--------|----------|
| `mlx_vector_array` leaks in turbo/GDN/SSM bindings | ✅ Fixed | mlx-swift `79a1dcd` |
| Prefill chunks retaining 5.4 GB in buffer pool | ✅ Fixed | `clearCache()` in `LLMModel.swift:45`, `Qwen35.swift:736`, `GPTOSS.swift:544`; post-loop for `Gemma4.swift:1189` (intra-loop clear intentionally dropped for Gemma4 per comment at 1183) |
| `RotatingKVCache.trim()` idx going negative | ✅ Fixed today | `062a628` — `idx = max(0, idx - trimmed)` + ensureBuffer + 1,011 test lines |
| `RotatingKVCache` circular write OOB when buffer < maxCacheSize | ✅ Fixed today | `062a628` — ensureBuffer pre-allocates full maxCacheSize |
| int16 overflow in SDPA NAX mask for KV seq > 32K | ✅ Fixed | mlx `a33b7916` |

### Thread safety — no known live bugs

| Issue | Status | Evidence |
|-------|--------|----------|
| Wired memory race | ✅ Fixed | mlx-swift `2755373` |
| Server concurrent request crashes | ✅ Fixed | `9fe61e7` — `ServerPromptCache` → actor |
| Cache trim crash on interrupted generation | ✅ Fixed | `a065e9f` |
| Thread join hang on process exit | ✅ Fixed | mlx `520cea2b` |

**Audit-only items** (theoretical, no known trigger):

- `TokenIterator` + `SendableBox` across Task boundaries — by design sound; `SerialAccessContainer` wraps with `AsyncMutex` at `SerialAccessContainer.swift:44`.
- `asyncEval()` thread affinity — MLX has internal sync; Swift Task migration is theoretical.
- Module-level `@Sendable` compiled closures in Gemma4 — MLX compile cache is thread-local; collisions theoretical.

Re-examine these *only* if sporadic concurrency crashes surface. Batch decode (Phase C) will stress the concurrent path hardest — do a 30-minute sanity pass before starting C.

**Bottom line**: no open memory leaks, no open thread-safety bugs.

---

## 2. Performance: What's Done (M1 Max numbers, no `--ppl --kld --think`)

### Peak decode on M1 Max — Gemma4 E2B 4-bit, turbo4v2, *without* overhead flags

| Context | Swift decode (tok/s) | Python mlx-vlm (tok/s) |
|---------|:--:|:--:|
| 128 | **104.1** | (n/a) |
| 1024 | **100.4** | (n/a) |
| 4096 | **97.3** | 96.3 |
| 8192 | — | 85.1 |
| 16384 | — | 82.9 |
| 32768 | **75.2** | 72.3 |

Source: `optimization-change-log.md:28-35` and `incept5-benchmark-comparison.md`.

**Swift is at or above Python mlx-vlm on M1 Max for Gemma4 E2B turbo4v2** at every comparable context. The ~20% "gap" in earlier notes was `--ppl --kld --think` overhead, not a real perf gap.

### What landed (ordered)

**Prefill**
- **2.3x Gemma prefill** (`00547bc` + mlx-swift `923e61d`): `dlopen` bridge + NAX framework dispatch (where supported). Gemma4 512 tok: 8,342 → 19,583 tok/s.
- **5.7x Qwen3.5 MoE prefill** (`2a695c0`): adaptive chunk 4096 for MoE, skip GPU→CPU sync in non-thinking. TTFT 12.5s → 2.4s at 1024.
- **Qwen bridge eval-barrier removal** (`aa1aa60`), async KV + contiguous removal (`48ce0f5`, `ded3387`).

**Decode**
- All 14 custom kernels migrated to C framework dispatch — zero JIT in hot path (TurboQuant ×10, GatedDelta ×2, SSM, rmsNormResidual).
- Fused RMSNorm+RoPE (+1-3% decode, -7% TTFT), RMSNorm+Residual (+1.2% decode, 90 dispatches/token saved), TurboQuant buffer-arg dispatch (+2-5% decode), FusedGateUpSwitchGLU (+5-6% decode, MoE), v_norm→MLXFast.rmsNorm (+2-4% decode), peek() caching (+3-5% for shared layers), PPL deferred phase tracking (+2-3% with flags), `trackPerplexity` default false.

**Correctness / memory**
- Dtype-leak fixes — peak memory 7.7 → 3.4 GB at 32K.
- Empirical kernel-option study in `mlx-kernel-options-empirical-2026-04-08.md`.

---

## 3. Experiments Already Run and Scientifically Ruled Out

These were in earlier plans as "open" but have been **implemented and definitively regressed on M1 Max**. Do **not** reopen without new evidence.

| Experiment | Status | Regression (M1 Max) | Code state |
|------------|--------|---------------------|------------|
| **Batched Q+K+V GEMV fusion** (weight concat, sequential shared-x, z-parallel) | Tested, all 3 regressed | -3% to -5% decode | Scaffolding preserved in mlx-swift. See `gemv-fusion-analysis-2026-04-12.md`. |
| **Warp Decode MoE** (output-neuron parallelism, `simd_shuffle_xor`) | Implemented, regressed | -20% to -25% decode (Qwen3.5-35B) | Full kernel chain present (mlx-swift-lm `caddf7a` + mlx-swift `630a160` + mlx `16e72e60` + mlx-c `c22b168`). Dispatch gated with `false && env(WARP_MOE_DECODE)` at `SwitchLayers.swift`. See `warp-decode-moe-analysis-2026-04-12.md`. |
| **`compile()` of large sub-computations** on *unfused* graph (compiled MLP, compiled QKV, compiledNormResidual) | Tested, neutral/regressed | ~0% to -1.6% decode | Reverted. Small-element-wise compiles (`compiledGeglu`, `compiledLogitSoftcap`) kept — they help. |
| **Steel SDPA BD=512 on M1 Max & M5 Max** | Proven infeasible | — | Both GPUs have 208 KB register budget; BD=512 can't fit. BD=256 is the practical ceiling and is already active. **Do not retest.** |

### Root cause from `gemv-fusion-analysis-2026-04-12.md`

> Fusion wins when it reduces **memory traffic**, not when it reduces **dispatch count**. On unified memory, in-buffer dispatch overhead is <0.5 ms on a ~10 ms decode token.

This explains why both the batched-QKV and Warp-MoE experiments failed. It does **not** rule out graph caching or Metal ICBs — see §5.

---

## 4. Items Already Implemented (double-checked today)

| Item | Status | Evidence |
|------|--------|----------|
| **Circular KV cache (logical circular indexing)** | ✅ Done | mlx-swift-lm `ebb63cf` (2026-04-12). `RotatingKVCache` uses pre-allocated buffer + `idx = end % maxCacheSize`. `temporalOrder()` and old `trim()` helper removed. Buffer-overflow corner-case fixed by today's `062a628`. See `KVCache.swift:525`+. |
| **Steel SDPA BD=256** | ✅ Active | mlx-swift `ca654b1` — BD=256 instantiated for float16/bfloat16. Float32 TGP overflow fixed in `8cab71d`. NAX guard (`594bfb0b`) restricts NAX dispatch to BD≤128. |
| **Metal Residency Sets (`MTLResidencySet`)** | ✅ Done | mlx-swift `resident.h/.cpp`, Metal3+ gated. Upstream fixes `8fe1d092`, `60c41543`, `1a28b69e`, `f98ce25a`. |
| **Dtype-leak fix** | ✅ Done | `b6cfca9`. Peak 7.7 → 3.4 GB at 32K. |
| **Generation-dedicated Metal stream (`withGenerationStream`)** | ✅ Done via Tom's pattern | mlx-swift-lm `Evaluate.swift:10-31`. Earlier attempts (`setAsDefault` at module init, `withNewDefaultStream` globally) regressed 6x because they replaced the default before it was initialized. Tom's pattern creates the stream **lazily on first access** and wraps each bridge/forward call in `withGenerationStream { ... }` with `defer { MLX.Stream.gpu.setAsDefault() }`. Applied to both prefill (`Qwen2.swift:60`) and decode (`Evaluate.swift:1183-1186`). Prefill gains 3-4x from this; decode gain is smaller but it's wired up. Remove this from any "ruled-out" list. |
| **`max_ops_per_buffer` tuning** | ✅ Done | **200 for Max/Ultra** is now the committed default: mlx `d5e8c0de` on `ek/qwen35-prefill-speedup`. Helps Gemma4/pure-attention decode +3-5%; hurts Qwen3.5 hybrid prefill ~17-24% and 32K peak ~28% (see §0.5 — `max_ops` is only one of three stacked causes for the Qwen3.5 prefill regression). Env override (`MLX_MAX_OPS_PER_BUFFER`) remains the per-run escape hatch. mlx-swift submodule pointer still needs bumping (P0-B.5). |

---

## 5. Items Genuinely Still Open

Ranked by expected value × confidence, M1 Max targeted.

| # | Item | Confidence | Category | Expected gain |
|---|------|:---:|----------|---------------|
| 1 | **Symbolic sliding-window mask** (Gemma4) — stop materializing [B,H,L,L] masks in sliding layers. Add `.slidingWindow(size:)` case to `ScaledDotProductAttentionMaskMode` and framework-dispatch it. | Medium | Memory traffic | 5-10% decode Gemma4 E2B at 4k+; reduces sliding-layer mask allocation (~3.7 GB at 4k × 28 layers per `ca654b1`) |
| 2 | **Graph caching re-evaluation on post-fusion code** — see §5.1. Prior `compile()` regression was on unfused graphs; today's hot path has 14+ larger fused ops, so tape-replay overhead is proportionally smaller. Worth a focused re-test. | Medium | CPU encoding overhead | TBD — kill-fast experiment |
| 3 | **Metal Indirect Command Buffers (ICB)** — pre-encode the decode dispatch sequence once, execute per token with updated bindings. Headers present in metal-cpp bindings; no dispatch code yet. See §5.1.1. | Medium (evidence from llama.cpp / research but unvalidated here) | Scheduler | Eliminates per-token CPU Metal encoding (targets the ~5-6 ms of per-token CPU encoding from the Metal System Trace in `metal-trace-decode-profile-2026-04-05.md`) |
| 4 | **Adaptive command-buffer (MB-aware commit threshold)** — `spec-adaptive-command-buffer-management.md`. Current commit logic tracks input bytes only; output allocations (e.g. 256 MB prefill score matrices) are invisible. | High | Scheduler / memory | +3-5% decode at ops=500 without prefill peak-memory increase |
| 5 | **NAX PR audit + BD=256 zero-length array fix** ([ekryski/mlx#1](https://github.com/ekryski/mlx/pull/1)) — see §5.2. NAX dispatch is currently restricted to head_dim ≤ 128 because of a known zero-length-array bug at BD=256. Fixing the bug unlocks NAX for head_dim=256 models (Gemma4) on M5 hardware. | High (if willing to touch kernel) | Kernel | M5-only; M1 Max unaffected |
| 6 | **VLM GPU→CPU roundtrip elimination** — 9 remaining sites across 5 VLMs per `gpu-cpu-gpu-roundtrips-remaining.md`. Replace `.asArray()` loops with `MLX.argWhere()` / `which()`. | High | Bandwidth / TTFT | TTFT reduction on VLMs |
| 7 | **Batch decoding** (Phase 4 from 04-07 plan — never started) | Medium | Throughput | 1.5-2.5x aggregate at batch=2; 2-3.5x at batch=4 |
| 8 | **N-gram speculative decoding improvements** (Phase 5 — never started, fully spec'd) | Medium | Latency | Variable; depends on workload acceptance rate |
| 9 | **Diffusion-based speculative decoding** (port DFlash / DDTree / Mirror-SD to Swift) — see §5.3 | Medium (external evidence) | Latency | 2-4x per external MLX-Python benchmarks |
| 10 | **ANE / Orion offloading** — planning only, no code | Low | Heterogeneous compute | Variable |

### 5.1 Graph caching — revised motivation

- Prior `compile()` regression was on the *unfused* graph: hundreds of small element-wise ops per layer. Tape-replay overhead (lock + closure alloc + cache lookup per op) dominated the graph-build savings.
- Today's hot path: 14 framework-dispatched fused kernels + a handful of element-wise ops. The tape is **much shorter**, and per-op tape-replay overhead is proportionally smaller.
- The 04-12 GEMV-fusion finding ("memory traffic, not dispatch count") applies to *GPU-side* dispatch overhead. Graph caching attacks *CPU-side* graph-walking + encoding (the 10.73 ms `asyncEval` dominant time in the Phase 8 profile from `final-performance-optimization-plan-04-07-2026.md:345-355`).
- Complementary to Metal ICBs but achievable without MLX framework changes.

Experiment (Phase B): wrap one Gemma4 decoder layer in `compile(shapeless: true)` or the `compiled()` API, A/B decode at contexts 1024 / 4096. Kill-fast rules: abandon if decode Δ < +3% after two attempts.

### 5.1.1 Metal ICBs — complementary experiment

Graph caching attacks the MLX-side graph walk. Metal Indirect Command Buffers attack the layer below: the actual Metal command encoding that `asyncEval` still has to do even after an MLX tape replay.

Per the Metal System Trace in `metal-trace-decode-profile-2026-04-05.md`, ~5-6 ms per decode token is spent in CPU-side Metal encoding — a quantity neither `compile()` nor any of the fusion work addresses. ICBs let us encode a decoder layer's dispatch sequence **once** at generation start, then execute it per token by updating only the handful of bindings that actually change (token-ID buffer, KV cache offsets, maybe position-encoding state).

ICB viability on Apple Silicon:
- Available on all M1+ hardware (`MTLIndirectCommandBuffer`).
- Supports compute dispatches.
- 16,384-command limit per ICB (our decode path is ~200 dispatches — plenty of headroom).
- Individual command bindings are mutable without re-encoding.
- Headers present in metal-cpp at `mlx-swift/Source/Cmlx/metal-cpp/Metal/MTLIndirectCommandBuffer.hpp`; no dispatch code yet.

Scope / risk:
- Requires MLX-framework changes in `mlx/backend/metal/` (scheduler + command encoder need to support an "encode-once, execute-many" mode for the decode hot path). Not a pure Swift experiment.
- Buffer sizes: KV cache grows every token. Allocate the ICB against the max-size pre-allocated KV buffer (Circular KV already guarantees this) and update the offset binding per token.
- Any op that lazily allocates intermediate buffers breaks the ICB approach. First task in the experiment is to audit which ops in a all model decode layers are re-encoding-safe.

Experiment: Phase B'. Kill-fast bar: same as Phase B (+3% decode). If both graph caching AND ICBs clear the bar, try stacking them (graph cache Swift side, ICB for the Metal side).

### 5.2 NAX PR audit ([ekryski/mlx#1](https://github.com/ekryski/mlx/pull/1))

The PR changes **1 file** (`mlx/backend/metal/scaled_dot_product_attention.cpp`) and adds:

```
q.shape(3) <= 128 && // NAX BD=256 has zero-length array bug
```

Effects:
- NAX dispatch is **enabled by default** when hardware supports it (`metal::is_nax_available()`). For M5 this means NAX is on; for M1 Max the branch is inert (safe fallback).
- The new guard **restricts NAX to head_dim ≤ 128**. Gemma4 (head_dim = 256) falls back to non-NAX Steel BD=256 on M5 until the zero-length-array bug is fixed.
- No tests disabled by the PR. M1 Max path is unaffected because NAX isn't available there.

Action items (Phase A4):
- Confirm the guard is in place on current mlx `ek/perf-improvements` tip and not skipped in downstream Swift bindings.
- Verify M5 fallback path produces numerically identical output (unit test or bit-exact A/B vs NAX-disabled build).
- File a tracking issue for the BD=256 zero-length-array bug so NAX can eventually handle head_dim=256 (Gemma4).

### 5.3 Diffusion-based speculative decoding — external references

Three MLX repos implementing diffusion-assisted drafting:

| Repo | Technique | Benchmarks (per README) | Language |
|------|-----------|-------------------------|----------|
| [humanrouter/ddtree-mlx](https://github.com/humanrouter/ddtree-mlx) | Tree-based draft + parallel tree verification on top of DFlash. Heap-based optimal draft tree. | 1.52x over autoregressive on Qwen3.5-27B; 85%+ accept rate for code | Python / MLX |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | Block-diffusion drafting: small (~1B) draft model emits 16-token blocks in parallel; target verifies in one pass. Tape-replay rollback via custom Metal kernels. | **4.10x on Qwen3.5-4B @ 2048 ctx; 4.13x on Qwen3.5-9B @ 2048 ctx; 1.98x on Qwen3.5-27B 4-bit @ 1024 ctx**. Acceptance > 87%. | Python / MLX |
| [0xClandestine/mirror-sd](https://github.com/0xClandestine/mirror-sd) | Heterogeneous accelerator dispatch: target on GPU, draft on ANE (currently non-viable due to bf16→f16 precision loss; GPU-only path works). Builds on DFlash. | 2.19x on M4 Max Qwen3.5-27B @ 2048; 3.55x on Qwen3-8B w/ block size 16 | Python / MLX |

These are real, working implementations — Phase E is no longer "research-only". Port plan:
1. Port DFlash block-diffusion draft to Swift (prerequisite for both DDTree and Mirror-SD).
2. Evaluate tree verification (DDTree) vs flat verification.
3. Reconsider Mirror-SD's GPU/ANE split once Orion investigation (§5 #10) has data.

---

## 6. Execution Plan

Each phase lands on its own feature branch off `ek/tom-eric-moe-tuning`, merged back after benchmark verification (`--ppl --kld --think` OFF for headline numbers; ON for regression checking).

### Phase A — Memory-traffic wins + infra audits

**A1. Symbolic sliding-window mask**
- Framework change in mlx-swift: add `.slidingWindow(size:)` case to `ScaledDotProductAttentionMaskMode` (`MLXFast.swift`) + dispatch in `scaled_dot_product_attention.cpp`.
- Consumer change in mlx-swift-lm: `Gemma4.swift:1054-1060` swap `makeAttentionMask()` for the new mode.
- Success: ≥ +5% decode on Gemma4 E2B at 4k context without flags; peak-memory reduction visible on sliding-layer allocations.

**A2. Adaptive command-buffer commit (MB-aware)**
- Implement per `spec-adaptive-command-buffer-management.md`.
- Track output allocation bytes in Metal command-buffer build loop; commit when input+output MB exceeds threshold, not just op count.
- File: mlx `mlx/backend/metal/command_buffer.cpp` (and/or `device.cpp`).
- Success: raise `max_ops_per_buffer` from 200 → 500 on M1 Max, observe +3% decode with no prefill peak-memory increase.

**A3. VLM GPU→CPU roundtrip elimination**
- 9 sites across 5 VLMs (Qwen VL, LFM2VL, FastVLM, Mistral3, Pixtral; Qwen3VL ×2, Qwen35, Gemma3, Gemma4). Replace `.asArray(Int.self).enumerated()` and `.asArray(Bool.self)` with `MLX.argWhere()` / `MLX.which()`.
- Skip Qwen25VL seq-length accumulation (lower priority per `gpu-cpu-gpu-roundtrips-remaining.md`).
- Success: TTFT regression-free on LLM benches + measurable TTFT improvement on VLMs.

**A4. NAX PR audit ([ekryski/mlx#1](https://github.com/ekryski/mlx/pull/1))**
- Confirm on latest mlx `ek/perf-improvements` tip that the `q.shape(3) <= 128` guard is in place and that M5 hardware actually takes the NAX path for head_dim ≤ 128 (validate via trace or env log).
- Verify bit-exact output between NAX-enabled and NAX-disabled builds at head_dim=64, 80, 128 for one representative model.
- File tracking issue for the BD=256 zero-length-array bug so NAX can one day handle Gemma4's head_dim=256 on M5.
- M1 Max: no code change needed; confirm the fallback is inert.

### Phase B — Graph caching re-evaluation (bounded, kill-fast)

Motivation in §5.1.

**B1. Per-layer `compile()` on decode** — compile one Gemma4 E2B decoder layer with `compile(shapeless: true)` (or the newer `compiled()` API). A/B at contexts 1024, 4096 without flags.

**B2. Per-layer `compile()` on prefill** — same harness, measure prefill tok/s.

**Kill rules**:
- If decode Δ < +3% after two attempts → abandon for decode.
- If prefill Δ < +5% → abandon for prefill.
- If either clears the bar, expand to a second model (GPT-OSS-20B) before generalizing.

**B3. Write results** to `graph-caching-re-evaluation-2026-04-XX.md` with the actual benchmark tables. This either validates the approach or gives us a stronger "ruled out post-fusion" citation.

### Phase B' — Metal ICB experiment (CPU-encoding overhead)

Runs in parallel with Phase B or immediately after, depending on engineering bandwidth. Motivation and scope in §5.1.1.

**B'1. Audit the Gemma4 decode dispatch list** — enumerate every Metal compute dispatch emitted per token for a Gemma4 E2B decoder layer. Classify each dispatch:
- Stable bindings (weight buffers, constants) — candidate for pre-encoding.
- Mutable bindings (input token buffer, KV cache slot offset, position offset) — must be updatable via ICB binding rewrite.
- Dynamic-shape / lazy-allocation ops — blockers; must find alternatives or accept they break the ICB path.

Output: `icb-dispatch-audit-2026-04-XX.md` with the full list and a go/no-go recommendation before touching MLX.

**B'2. Prototype ICB encoding for one layer** — behind an env flag (e.g. `MLX_METAL_ICB=1`), change MLX's Metal scheduler to:
- On first decode token of a generation, encode the decoder-layer dispatch sequence into an `MTLIndirectCommandBuffer`.
- On subsequent tokens, update only the mutable bindings identified in B'1 and call `executeCommandsInBuffer`.
- Fall back to the normal encoding path if any guard condition fails (dtype mismatch, shape mismatch, etc.).

Files: mlx `mlx/backend/metal/` (scheduler, command encoder), plus Swift/C dispatch wrapper in mlx-swift.

**B'3. Benchmark** — same harness as B: Gemma4 E2B at 1024, 4096 without flags, compare `MLX_METAL_ICB=0` vs `=1`.

**Kill rules**:
- If decode Δ < +3% after the prototype works → abandon, document findings.
- If it clears the bar but requires unacceptably invasive framework changes → document as long-term work, keep prototype behind env flag.
- If it clears the bar and is clean enough to merge → wire up for a second model (GPT-OSS-20B, then Qwen3.5-35B) before promoting to default.

**B'4. Write results** to `metal-icb-experiment-2026-04-XX.md`.

Safety consideration: ICBs bypass a lot of MLX's normal safety nets. Before any default-on flip, run the full safety regression suite — especially `KVCacheTests` and `Gemma4Tests`, which exercise cache update/read paths that an ICB prototype is most likely to break.

### Phase C — Batch decoding (Phase 4 from 04-07 plan)

Gate on Phase A completion. Scope to pure-attention models (Gemma4 E2B/26B, GPT-OSS-20B). Skip GatedDelta hybrids (Qwen3.5) — sequential O(T) needs batched-kernel changes.

**C1. `BatchKVCache` wrapper** — `KVCache.swift`. Wraps B independent `KVCache` instances; stacks K/V for batched attention. TurboQuant already flattens batch via `totalQ = B·nQHeads·L`, no kernel change.

**C2. `BatchTokenIterator`** — `Evaluate.swift`. Parallel to `TokenIterator`; single forward pass returns `[B, vocab]`. Reuse `SerialAccessContainer` pattern.

**C3. Batch-aware sampler** — extend `Sampler` protocol to accept `[B, vocab]` with per-sequence penalties.

**C4. Benchmark**:
```bash
benchmark.sh --model gemma4-e2b --quant 4bit --kv none,turbo4v2 --method summarization \
  --context 1024,4096 --batch 2
benchmark.sh --model gemma4-e2b --quant 4bit --kv none,turbo4v2 --method summarization \
  --context 1024,4096 --batch 4
```
Success: ≥ 1.5x aggregate at batch=2; ≥ 2x at batch=4. Peak memory within turbo4v2 budget at 4k.

**Risk check**: run the thread-safety audit items from §1 as a 30-minute sanity pass before starting C.

### Phase D — N-gram speculative decoding (Phase 5 from 04-07 plan)

Full spec already at §5 of `final-performance-optimization-plan-04-07-2026.md`.

- **D1. Hash-based n-gram lookup** — FNV-1a, O(1) replaces O(n·m).
- **D2. Multi-size n-gram (2-5)** — longest match wins.
- **D3. Dynamic draft length** — 8 at >70% accept, 3 at 20-40%, disable <20%.
- **D4. Fix acceptance-rate metric** — divide by actual proposals, not slot capacity.
- **D5. Expose metrics** — `ngramProposed`, `ngramAccepted`, `ngramAcceptanceRate` in `GenerateCompletionInfo`.

Reuse existing `SpeculativeTokenIterator` verification path.
Success: ≥ 30% acceptance on a summarization corpus with dynamic length; zero regression when off.

### Phase E — Diffusion-based speculative decoding (port from MLX-Python references)

Gate on Phase D. Design note before code.

**E1. Survey note** (`diffusion-spec-decoding-survey-2026-04-XX.md`) — compare DFlash, DDTree, Mirror-SD per §5.3 against each other and against the existing `SpeculativeTokenIterator`. Identify what's portable (block-diffusion draft) vs what needs new infra (tree verification, target-aware attention for draft).

**E2. Port DFlash draft-model path** — block diffusion, 16-token parallel draft. Smallest unit of work that can show a speedup. Requires:
- A trained block-diffusion draft model matched to a target we support (Qwen3.5-27B-A3B is the existing DFlash pairing — confirm availability or need to train).
- Custom Metal kernel for tape-replay rollback (per DFlash README).
- Integration with existing `SpeculativeTokenIterator` verification.

**E3. Evaluate tree verification (DDTree)** only if DFlash port clears 1.5x end-to-end on a representative Qwen model on M1 Max.

**E4. Mirror-SD** deferred until Orion/ANE investigation has data on bf16 precision on M-series ANE.

---

## 7. Critical Files

| Path | Purpose | Phase |
|------|---------|:---:|
| `Libraries/MLXLMCommon/KVCache.swift` | BatchKVCache wrapper | C1 |
| `Libraries/MLXLMCommon/Evaluate.swift` | BatchTokenIterator, n-gram speculative, diffusion draft hooks | C2–C3, D, E |
| `Libraries/MLXLLM/Models/Gemma4.swift` | Sliding-window mask call site | A1 |
| mlx-swift `MLXFast.swift` | Add `.slidingWindow(size:)` case | A1 |
| mlx `scaled_dot_product_attention.cpp` | Sliding-window mask dispatch, NAX guard | A1, A4 |
| mlx `command_buffer.cpp`, `device.cpp` | Adaptive MB-aware commit; ICB scheduler changes | A2, B' |
| mlx `mlx/backend/metal/` (scheduler, command encoder) | ICB encode-once/execute-many path | B' |
| VLM model files (Qwen VL, LFM2VL, FastVLM, Mistral3, Pixtral, Qwen3VL ×2, Qwen35, Gemma3, Gemma4) | Replace `.asArray()` | A3 |

Related specs to re-read before each phase: `spec-layer-level-kernel-fusion.md`, `spec-adaptive-command-buffer-management.md`, `gpu-cpu-gpu-roundtrips-remaining.md`.

---

## 8. Verification

Per-phase benchmark harness (M1 Max, headline numbers *without* `--ppl --kld --think`; rerun with flags for regression coverage):

```bash
benchmark.sh --model <model> --quant 4bit --kv none,turbo4v2 \
  --method summarization --context 1024,4096,16384
```

Primary models: Gemma4 E2B, Gemma4 26B-A4B, Qwen3.5-35B-A3B, GPT-OSS-20B.

Success gates:
- **P0** — GPT-OSS-20B decode ≥ 55 tok/s @ 128 ctx turbo4v2 on M1 Max; Gemma4 / Qwen35 decode regression-free; post-mortem written. **This must pass before any other phase merges.**
- **A1** — ≥ +5% decode Gemma4 E2B 4k; peak memory for sliding layers reduced.
- **A2** — +3% decode with ops=500; peak memory unchanged or lower at prefill.
- **A3** — TTFT improvement on a VLM; LLM benches regression-free.
- **A4** — NAX guard verified; bit-exact output; tracking issue filed for BD=256 bug.
- **B** — kill-fast per §B (decode +3% or prefill +5%); results doc written either way.
- **B'** — kill-fast per §B'; dispatch audit written before any MLX change; results doc written either way.
- **C** — batch=2 ≥ 1.5x; batch=4 ≥ 2x; peak memory within budget.
- **D** — ≥ 30% acceptance on summarization; zero regression when off.
- **E** — survey written; if port attempted, ≥ 1.5x e2e on Qwen-family target.

Safety regression suite per phase before merge: `Tests/MLXLMTests/KVCacheTests.swift`, `Tests/MLXLMTests/Gemma4Tests.swift`, `Tests/MLXLMTests/SpeculativeDecodingTests.swift`, `Tests/MLXLMTests/Gemma4ChatTemplateTests.swift`.

---

## 9. Single-line Summary

**P0 complete (GPT-OSS-20B decode restored by `5002e44` — debug-logging GPU→CPU syncs in the attention hot path); P0-B now active: Qwen3.5 hybrid prefill is ~15× slower than 2026-04-02 — three stacked causes, chiefly the 04-13 revert of the "5.7x prefill speedup" `max(chunk, 4096)` floor plus the per-layer prefill eval added in the same commit; `max_ops=200` is the smallest contributor and is now committed at mlx `d5e8c0de`. Recovery plan in §0.5: restore chunk-size floor, eval every N layers instead of every layer, preserve the peak-memory win from 04-13.** After both P0s: no open memory / thread-safety bugs; Circular KV, Steel BD=256, Residency Sets, and the generation-dedicated stream are all live; Warp MoE + batched QKV fusion are implemented and regressed (stay off); graph caching AND Metal ICBs get paired CPU-overhead experiments on today's fused graph; genuinely open work is symbolic sliding-window mask → adaptive cmd-buffer → VLM roundtrips → batch decode → n-gram speculative → DFlash diffusion-based speculative port.
