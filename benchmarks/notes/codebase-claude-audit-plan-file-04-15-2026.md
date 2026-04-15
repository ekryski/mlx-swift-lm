# Final Performance Plan — 2026-04-15

> **Deliverable**: on approval this plan is written to `benchmarks/notes/final-performance-plan-2026-04-15.md` in `mlx-swift-lm` on `ek/tom-eric-moe-tuning`. Audit + plan only — no code implemented in this session.

## Context

Picking up from the 2026-04-07 plan after two weeks of perf + safety work. Goals today:

1. Audit memory-leak and thread-unsafe MLX-array concerns against the current state of all four repos.
2. Re-verify every item in the "still open" list against actual code (user correctly noted several were already tried-and-regressed *or* already implemented).
3. Decide what remains that's worth chasing. User's requested downstream sequence: **graph-caching re-evaluation → batch decode → n-gram speculative → diffusion speculative**.

All benchmark numbers cited are **M1 Max 64 GB** (Eric's machine). M5 Max numbers (Tom's machine) are excluded.

Repos audited:

| Repo | Path | Branch |
|------|------|--------|
| `mlx-swift-lm` | `/Users/eric/Development/personal/ai/mlx-swift-lm/` | `ek/tom-eric-moe-tuning` |
| `mlx-swift` | (dependency / local) | `ek/speed-improvements-2` |
| `mlx` (submodule) | under mlx-swift | `ek/perf-improvements` |
| `mlx-c` (submodule) | under mlx-swift | `ekryski/perf-improvements` |

This planning worktree (`.claude/worktrees/focused-murdock`, based on `main` @ 826fbc9) does not contain the perf work — implementation must happen in the main checkout on the branches above.

---

## 1. Safety Audit — Memory Leaks and Thread-Unsafe MLX Arrays

### Memory leaks — all known issues fixed

| Issue | Status | Evidence |
|-------|--------|----------|
| `mlx_vector_array` leaks in turbo/GDN/SSM bindings | ✅ Fixed | mlx-swift `79a1dcd` |
| Prefill chunks retaining 5.4 GB in buffer pool | ✅ Fixed | `clearCache()` in [LLMModel.swift:45](Libraries/MLXLLM/LLMModel.swift:45), [Qwen35.swift:736](Libraries/MLXLLM/Models/Qwen35.swift:736), [GPTOSS.swift:544](Libraries/MLXLLM/Models/GPTOSS.swift:544); post-loop for [Gemma4.swift:1189](Libraries/MLXLLM/Models/Gemma4.swift:1189) (intra-loop clear intentionally dropped for Gemma4 per comment at 1183) |
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

- `TokenIterator` + `SendableBox` across Task boundaries — by design sound; `SerialAccessContainer` wraps with `AsyncMutex` at [SerialAccessContainer.swift:44](Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift:44).
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

Source: [optimization-change-log.md:28-35](benchmarks/notes/optimization-change-log.md) and [incept5-benchmark-comparison.md](benchmarks/notes/incept5-benchmark-comparison.md).

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
- Empirical kernel-option study in [mlx-kernel-options-empirical-2026-04-08.md](benchmarks/notes/mlx-kernel-options-empirical-2026-04-08.md).

---

## 3. Experiments Already Run and Scientifically Ruled Out

These were in earlier plans as "open" but have been **implemented and definitively regressed on M1 Max**. Do **not** reopen without new evidence.

| Experiment | Status | Regression (M1 Max) | Code state |
|------------|--------|---------------------|------------|
| **Batched Q+K+V GEMV fusion** (weight concat, sequential shared-x, z-parallel) | Tested, all 3 regressed | -3% to -5% decode | Scaffolding preserved in mlx-swift. See [gemv-fusion-analysis-2026-04-12.md](benchmarks/notes/gemv-fusion-analysis-2026-04-12.md). |
| **Warp Decode MoE** (output-neuron parallelism, `simd_shuffle_xor`) | Implemented, regressed | -20% to -25% decode (Qwen3.5-35B) | Full kernel chain present (mlx-swift-lm `caddf7a` + mlx-swift `630a160` + mlx `16e72e60` + mlx-c `c22b168`). Dispatch gated with `false && env(WARP_MOE_DECODE)` at [SwitchLayers.swift](Libraries/MLXLMCommon/SwitchLayers.swift). See [warp-decode-moe-analysis-2026-04-12.md](benchmarks/notes/warp-decode-moe-analysis-2026-04-12.md). |
| **`compile()` of large sub-computations** on *unfused* graph (compiled MLP, compiled QKV, compiledNormResidual) | Tested, neutral/regressed | ~0% to -1.6% decode | Reverted. Small-element-wise compiles (`compiledGeglu`, `compiledLogitSoftcap`) kept — they help. |
| **Steel SDPA BD=512 on M1 Max & M5 Max** | Proven infeasible | — | Both GPUs have 208 KB register budget; BD=512 can't fit. BD=256 is the practical ceiling and is already active. **Do not retest.** |

### Root cause from [gemv-fusion-analysis-2026-04-12.md](benchmarks/notes/gemv-fusion-analysis-2026-04-12.md)

> Fusion wins when it reduces **memory traffic**, not when it reduces **dispatch count**. On unified memory, in-buffer dispatch overhead is <0.5 ms on a ~10 ms decode token.

This explains why both the batched-QKV and Warp-MoE experiments failed. It does **not** rule out graph caching — see §5.

---

## 4. Items Already Implemented (user double-checks confirmed)

| Item | Status | Evidence |
|------|--------|----------|
| **Circular KV cache (logical circular indexing)** | ✅ Done | mlx-swift-lm `ebb63cf` (2026-04-12). `RotatingKVCache` uses pre-allocated buffer + `idx = end % maxCacheSize`. `temporalOrder()` and old `trim()` helper removed. Buffer-overflow corner-case fixed by today's `062a628`. See [KVCache.swift:525](Libraries/MLXLMCommon/KVCache.swift:525)+. |
| **Steel SDPA BD=256** | ✅ Active | mlx-swift `ca654b1` — BD=256 instantiated for float16/bfloat16. Float32 TGP overflow fixed in `8cab71d`. NAX guard (`594bfb0b`) restricts NAX dispatch to BD≤128. |
| **Metal Residency Sets (`MTLResidencySet`)** | ✅ Done | mlx-swift `resident.h/.cpp`, Metal3+ gated. Upstream fixes `8fe1d092`, `60c41543`, `1a28b69e`, `f98ce25a`. |
| **Dtype-leak fix** | ✅ Done | `b6cfca9`. Peak 7.7 → 3.4 GB at 32K. |
| **Generation-dedicated Metal stream (`withGenerationStream`)** | ✅ Done via Tom's pattern | mlx-swift-lm [Evaluate.swift:10-31](Libraries/MLXLMCommon/Evaluate.swift:10). Earlier attempts (`setAsDefault` at module init, `withNewDefaultStream` globally) regressed 6x because they replaced the default before it was initialized. Tom's pattern creates the stream **lazily on first access** and wraps each bridge/forward call in `withGenerationStream { ... }` with `defer { MLX.Stream.gpu.setAsDefault() }`. Applied to both prefill ([Qwen2.swift:60](Libraries/MLXLLM/Models/Qwen2.swift:60)) and decode ([Evaluate.swift:1183-1186](Libraries/MLXLMCommon/Evaluate.swift:1183)). Prefill gains 3-4x from this; decode gain is smaller but it's wired up. Remove this from any "ruled-out" list. |
| **`max_ops_per_buffer` tuning** | ✅ Partial | 200 for M1 Max/Ultra. Env override available. (Not 300 as the 04-07 plan table implied.) |

---

## 5. Items Genuinely Still Open

Ranked by expected value × confidence, M1 Max targeted.

| # | Item | Confidence | Category | Expected gain |
|---|------|:---:|----------|---------------|
| 1 | **Symbolic sliding-window mask** (Gemma4) — stop materializing [B,H,L,L] masks in sliding layers. Add `.slidingWindow(size:)` case to `ScaledDotProductAttentionMaskMode` and framework-dispatch it. | Medium | Memory traffic | 5-10% decode Gemma4 E2B at 4k+; reduces sliding-layer mask allocation (~3.7 GB at 4k × 28 layers per `ca654b1`) |
| 2 | **Graph caching re-evaluation on post-fusion code** — see §5.1 below. User's argument: prior `compile()` regression was measured against unfused graphs with hundreds of small ops; today's hot path has 14+ larger fused ops, so tape-replay overhead is proportionally smaller. Worth a focused re-test. | Medium | CPU encoding overhead | TBD — kill-fast experiment |
| 3 | **Adaptive command-buffer (MB-aware commit threshold)** — [spec-adaptive-command-buffer-management.md](benchmarks/notes/spec-adaptive-command-buffer-management.md). Current commit logic tracks input bytes only; output allocations (e.g. 256 MB prefill score matrices) are invisible. | High | Scheduler / memory | +3-5% decode at ops=500 without prefill peak-memory increase |
| 4 | **NAX PR audit + BD=256 zero-length array fix** ([ekryski/mlx#1](https://github.com/ekryski/mlx/pull/1)) — see §5.2. Today the NAX dispatch is restricted to head_dim ≤ 128 because of a known zero-length-array bug at BD=256. Fixing the bug unlocks NAX for head_dim=256 models (Gemma4) on M5 hardware. | High (if willing to touch kernel) | Kernel | M5-only; M1 Max unaffected |
| 5 | **VLM GPU→CPU roundtrip elimination** — 9 remaining sites across 5 VLMs per [gpu-cpu-gpu-roundtrips-remaining.md](benchmarks/notes/gpu-cpu-gpu-roundtrips-remaining.md). Replace `.asArray()` loops with `MLX.argWhere()` / `which()`. | High | Bandwidth / TTFT | TTFT reduction on VLMs |
| 6 | **Batch decoding** (Phase 4 from 04-07 plan — never started) | Medium | Throughput | 1.5-2.5x aggregate at batch=2; 2-3.5x at batch=4 |
| 7 | **N-gram speculative decoding improvements** (Phase 5 — never started, fully spec'd) | Medium | Latency | Variable; depends on workload acceptance rate |
| 8 | **Diffusion-based speculative decoding** (port DFlash / DDTree / Mirror-SD to Swift) — see §5.3 | Medium (external evidence) | Latency | 2-4x per external MLX-Python benchmarks |
| 9 | **ANE / Orion offloading** — planning only, no code | Low | Heterogeneous compute | Variable |
| 10 | **Metal Indirect Command Buffers (ICB)** — pre-encode the decode dispatch sequence once, execute per token with updated bindings. Headers present in metal-cpp bindings; no dispatch code yet. See §5.4. | Medium (evidence from llama.cpp / research but unvalidated here) | Scheduler | Eliminates per-token CPU Metal encoding (targets the ~5-6 ms of per-token CPU encoding from the Metal System Trace in [metal-trace-decode-profile-2026-04-05.md](benchmarks/notes/metal-trace-decode-profile-2026-04-05.md)) |

### 5.1 Graph caching — revised motivation

Prior reasoning (and my earlier draft) treated graph caching as ruled out. User's pushback is correct and prompts a re-test:

- Prior `compile()` regression was on the *unfused* graph: hundreds of small element-wise ops per layer. Tape-replay overhead (lock + closure alloc + cache lookup per op) dominated the graph-build savings.
- Today's hot path: 14 framework-dispatched fused kernels + a handful of element-wise ops. The tape is **much shorter**, and per-op tape-replay overhead is proportionally smaller.
- The 04-12 GEMV-fusion finding ("memory traffic, not dispatch count") applies to *GPU-side* dispatch overhead. Graph caching attacks *CPU-side* graph-walking + encoding (the 10.73 ms `asyncEval` dominant time in the Phase 8 profile from [final-performance-optimization-plan-04-07-2026.md:345-355](benchmarks/notes/final-performance-optimization-plan-04-07-2026.md)).
- Complementary to Metal ICBs (which pre-encode the Metal command sequence) but achievable today without MLX framework changes.

Experiment (see Phase B below): wrap one Gemma4 decoder layer in `compile(shapeless: true)` or the `compiled()` API, A/B decode at contexts 1024 / 4096. Kill-fast rules: abandon if decode Δ < +3% after two attempts.

### 5.1.1 Metal ICBs — complementary experiment

Graph caching attacks the MLX-side graph walk. Metal Indirect Command Buffers attack the layer below: the actual Metal command encoding that `asyncEval` still has to do even after an MLX tape replay.

Per the Metal System Trace in [metal-trace-decode-profile-2026-04-05.md](benchmarks/notes/metal-trace-decode-profile-2026-04-05.md), ~5-6 ms per decode token is spent in CPU-side Metal encoding — a quantity neither `compile()` nor any of the fusion work addresses. ICBs let us encode a decoder layer's dispatch sequence **once** at generation start, then execute it per token by updating only the handful of bindings that actually change (token-ID buffer, KV cache offsets, maybe position-encoding state).

ICB viability on Apple Silicon:
- Available on all M1+ hardware (`MTLIndirectCommandBuffer`).
- Supports compute dispatches.
- 16,384-command limit per ICB (our decode path is ~200 dispatches — plenty of headroom).
- Individual command bindings are mutable without re-encoding.
- Headers present in metal-cpp at `mlx-swift/Source/Cmlx/metal-cpp/Metal/MTLIndirectCommandBuffer.hpp`; no dispatch code yet.

Scope / risk:
- Requires MLX-framework changes in `mlx/backend/metal/` (scheduler + command encoder need to support an "encode-once, execute-many" mode for the decode hot path). Not a pure Swift experiment.
- Buffer sizes: KV cache grows every token. Allocate the ICB against the max-size pre-allocated KV buffer (Circular KV already guarantees this) and update the offset binding per token.
- Any op that lazily allocates intermediate buffers breaks the ICB approach. First task in the experiment is to audit which ops in a Gemma4 decode layer are reencoding-safe.

Experiment: see Phase B' below. Kill-fast bar: same as Phase B (+3% decode). If both graph caching AND ICBs clear the bar, try stacking them (graph cache Swift side, ICB for the Metal side).

### 5.2 NAX PR audit ([ekryski/mlx#1](https://github.com/ekryski/mlx/pull/1))

The PR changes **1 file** (`mlx/backend/metal/scaled_dot_product_attention.cpp`) and adds:

```
q.shape(3) <= 128 && // NAX BD=256 has zero-length array bug
```

Effects:
- NAX dispatch is **enabled by default** when hardware supports it (`metal::is_nax_available()`). For M5 this means NAX is on; for M1 Max the branch is inert (safe fallback).
- The new guard **restricts NAX to head_dim ≤ 128**. Gemma4 (head_dim = 256) falls back to non-NAX Steel BD=256 on M5 until the zero-length-array bug is fixed.
- No tests disabled by the PR. M1 Max path is unaffected because NAX isn't available there.

Action items for this audit (Phase A4 below):
- Confirm the guard is correct against current mlx `ek/perf-improvements` tip and not skipped in downstream Swift bindings.
- Verify M5 fallback path produces numerically identical output (unit test or bit-exact A/B vs NAX-disabled build).
- File a tracking issue for the BD=256 zero-length-array bug so NAX can eventually handle head_dim=256 (Gemma4).

### 5.3 Diffusion-based speculative decoding — external references

User supplied three MLX repos implementing diffusion-assisted drafting:

| Repo | Technique | Benchmarks (per README) | Language |
|------|-----------|-------------------------|----------|
| [humanrouter/ddtree-mlx](https://github.com/humanrouter/ddtree-mlx) | Tree-based draft + parallel tree verification on top of DFlash. Heap-based optimal draft tree. | 1.52x over autoregressive on Qwen3.5-27B; 85%+ accept rate for code | Python / MLX |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | Block-diffusion drafting: small (~1B) draft model emits 16-token blocks in parallel; target verifies in one pass. Tape-replay rollback via custom Metal kernels. | **4.10x on Qwen3.5-4B @ 2048 ctx; 4.13x on Qwen3.5-9B @ 2048 ctx; 1.98x on Qwen3.5-27B 4-bit @ 1024 ctx**. Acceptance > 87%. | Python / MLX |
| [0xClandestine/mirror-sd](https://github.com/0xClandestine/mirror-sd) | Heterogeneous accelerator dispatch: target on GPU, draft on ANE (currently non-viable due to bf16→f16 precision loss; GPU-only path works). Builds on DFlash. | 2.19x on M4 Max Qwen3.5-27B @ 2048; 3.55x on Qwen3-8B w/ block size 16 | Python / MLX |

These are real, working implementations — Phase E is no longer "research-only". Port plan:
1. Port DFlash block-diffusion draft to Swift (prerequisite for both DDTree and Mirror-SD).
2. Evaluate tree verification (DDTree) vs flat verification.
3. Reconsider Mirror-SD's GPU/ANE split once Orion investigation (Phase §5 #9) has data.

---

## 6. Plan for Today (Audit Deliverable)

On approval, written to `benchmarks/notes/final-performance-plan-2026-04-15.md`. No code in this session.

Downstream execution (separate sessions), each phase on its own feature branch off `ek/tom-eric-moe-tuning`, merged back after benchmark verification (`--ppl --kld --think` OFF for headline numbers; ON for regression checking).

### Phase A — Memory-traffic wins + infra audits

**A1. Symbolic sliding-window mask**
- Framework change in mlx-swift: add `.slidingWindow(size:)` case to `ScaledDotProductAttentionMaskMode` (`MLXFast.swift`) + dispatch in `scaled_dot_product_attention.cpp`.
- Consumer change in mlx-swift-lm: [Gemma4.swift:1054-1060](Libraries/MLXLLM/Models/Gemma4.swift:1054) swap `makeAttentionMask()` for the new mode.
- Success: ≥ +5% decode on Gemma4 E2B at 4k context without flags; peak-memory reduction visible on sliding-layer allocations.

**A2. Adaptive command-buffer commit (MB-aware)**
- Implement per [spec-adaptive-command-buffer-management.md](benchmarks/notes/spec-adaptive-command-buffer-management.md).
- Track output allocation bytes in Metal command-buffer build loop; commit when input+output MB exceeds threshold, not just op count.
- File: mlx `mlx/backend/metal/command_buffer.cpp` (and/or `device.cpp`).
- Success: raise `max_ops_per_buffer` from 200 → 500 on M1 Max, observe +3% decode with no prefill peak-memory increase.

**A3. VLM GPU→CPU roundtrip elimination**
- 9 sites across 5 VLMs (Qwen VL, LFM2VL, FastVLM, Mistral3, Pixtral; Qwen3VL ×2, Qwen35, Gemma3, Gemma4). Replace `.asArray(Int.self).enumerated()` and `.asArray(Bool.self)` with `MLX.argWhere()` / `MLX.which()`.
- Skip Qwen25VL seq-length accumulation (lower priority per [gpu-cpu-gpu-roundtrips-remaining.md](benchmarks/notes/gpu-cpu-gpu-roundtrips-remaining.md)).
- Success: TTFT regression-free on LLM benches + measurable TTFT improvement on VLMs.

**A4. NAX PR audit ([ekryski/mlx#1](https://github.com/ekryski/mlx/pull/1))**
- Confirm on latest mlx `ek/perf-improvements` tip that the `q.shape(3) <= 128` guard is in place and that M5 hardware actually takes the NAX path for head_dim ≤ 128 (validate via trace or env log).
- Verify bit-exact output between NAX-enabled and NAX-disabled builds at head_dim=64, 80, 128 for one representative model.
- File tracking issue for the BD=256 zero-length-array bug so NAX can one day handle Gemma4's head_dim=256 on M5.
- M1 Max: no code change needed; confirm the fallback is inert.

### Phase B — Graph caching re-evaluation (bounded, kill-fast)

Newly elevated from "optional later" per user feedback — motivation in §5.1.

**B1. Per-layer `compile()` on decode** — compile one Gemma4 E2B decoder layer with `compile(shapeless: true)` (or the newer `compiled()` API). A/B at contexts 1024, 4096 without flags.

**B2. Per-layer `compile()` on prefill** — same harness, measure prefill tok/s.

**Kill rules**:
- If decode Δ < +3% after two attempts → abandon for decode.
- If prefill Δ < +5% → abandon for prefill.
- If either clears the bar, expand to a second model (GPT-OSS-20B) before generalizing.

**B3. Write results** to `benchmarks/notes/graph-caching-re-evaluation-2026-04-XX.md` with the actual benchmark tables. This either validates the approach or gives us a stronger "ruled out post-fusion" citation.

### Phase B' — Metal ICB experiment (CPU-encoding overhead)

Runs in parallel with Phase B or immediately after, depending on engineering bandwidth. Motivation and scope in §5.1.1.

**B'1. Audit the Gemma4 decode dispatch list** — enumerate every Metal compute dispatch emitted per token for a Gemma4 E2B decoder layer. Classify each dispatch:
- Stable bindings (weight buffers, constants) — candidate for pre-encoding.
- Mutable bindings (input token buffer, KV cache slot offset, position offset) — must be updatable via ICB binding rewrite.
- Dynamic-shape / lazy-allocation ops — blockers; must find alternatives or accept they break the ICB path.

Output: `benchmarks/notes/icb-dispatch-audit-2026-04-XX.md` with the full list and a go/no-go recommendation before touching MLX.

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

**B'4. Write results** to `benchmarks/notes/metal-icb-experiment-2026-04-XX.md`.

Safety consideration: ICBs bypass a lot of MLX's normal safety nets. Before any default-on flip, run the full safety regression suite — especially `KVCacheTests` and `Gemma4Tests`, which exercise cache update/read paths that an ICB prototype is most likely to break.

### Phase C — Batch decoding (Phase 4 from 04-07 plan)

Gate on Phase A completion. Scope to pure-attention models (Gemma4 E2B/26B, GPT-OSS-20B). Skip GatedDelta hybrids (Qwen3.5) — sequential O(T) needs batched-kernel changes.

**C1. `BatchKVCache` wrapper** — [KVCache.swift](Libraries/MLXLMCommon/KVCache.swift). Wraps B independent `KVCache` instances; stacks K/V for batched attention. TurboQuant already flattens batch via `totalQ = B·nQHeads·L`, no kernel change.

**C2. `BatchTokenIterator`** — [Evaluate.swift](Libraries/MLXLMCommon/Evaluate.swift). Parallel to `TokenIterator`; single forward pass returns `[B, vocab]`. Reuse `SerialAccessContainer` pattern.

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

**E1. Survey note** (`benchmarks/notes/diffusion-spec-decoding-survey-2026-04-XX.md`) — compare DFlash, DDTree, Mirror-SD per §5.3 against each other and against the existing `SpeculativeTokenIterator`. Identify what's portable (block-diffusion draft) vs what needs new infra (tree verification, target-aware attention for draft).

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
| [Libraries/MLXLMCommon/KVCache.swift](Libraries/MLXLMCommon/KVCache.swift) | BatchKVCache wrapper | C1 |
| [Libraries/MLXLMCommon/Evaluate.swift](Libraries/MLXLMCommon/Evaluate.swift) | BatchTokenIterator, n-gram speculative, diffusion draft hooks | C2–C3, D, E |
| [Libraries/MLXLLM/Models/Gemma4.swift](Libraries/MLXLLM/Models/Gemma4.swift) | Sliding-window mask call site | A1 |
| mlx-swift `MLXFast.swift` | Add `.slidingWindow(size:)` case | A1 |
| mlx `scaled_dot_product_attention.cpp` | Sliding-window mask dispatch, NAX guard | A1, A4 |
| mlx `command_buffer.cpp`, `device.cpp` | Adaptive MB-aware commit | A2 |
| VLM model files (Qwen VL, LFM2VL, FastVLM, Mistral3, Pixtral, Qwen3VL ×2, Qwen35, Gemma3, Gemma4) | Replace `.asArray()` | A3 |

Related specs to re-read before each phase: [spec-layer-level-kernel-fusion.md](benchmarks/notes/spec-layer-level-kernel-fusion.md), [spec-adaptive-command-buffer-management.md](benchmarks/notes/spec-adaptive-command-buffer-management.md), [gpu-cpu-gpu-roundtrips-remaining.md](benchmarks/notes/gpu-cpu-gpu-roundtrips-remaining.md).

---

## 8. Verification

Per-phase benchmark harness (M1 Max, headline numbers *without* `--ppl --kld --think`; rerun with flags for regression coverage):

```bash
benchmark.sh --model <model> --quant 4bit --kv none,turbo4v2 \
  --method summarization --context 1024,4096,16384
```

Primary models: Gemma4 E2B, Gemma4 26B-A4B, Qwen3.5-35B-A3B, GPT-OSS-20B.

Success gates:
- **A1** — ≥ +5% decode Gemma4 E2B 4k; peak memory for sliding layers reduced.
- **A2** — +3% decode with ops=500; peak memory unchanged or lower at prefill.
- **A3** — TTFT improvement on a VLM; LLM benches regression-free.
- **A4** — NAX guard verified; bit-exact output; tracking issue filed for BD=256 bug.
- **B** — kill-fast per §B (decode +3% or prefill +5%); results doc written either way.
- **B'** — kill-fast per §B'; dispatch audit written before any MLX change; results doc written either way.
- **C** — batch=2 ≥ 1.5x; batch=4 ≥ 2x; peak memory within budget.
- **D** — ≥ 30% acceptance on summarization; zero regression when off.
- **E** — survey written; if port attempted, ≥ 1.5x e2e on Qwen-family target.

Safety regression suite per phase before merge: [KVCacheTests.swift](Tests/MLXLMTests/KVCacheTests.swift), [Gemma4Tests.swift](Tests/MLXLMTests/Gemma4Tests.swift), [SpeculativeDecodingTests.swift](Tests/MLXLMTests/SpeculativeDecodingTests.swift), [Gemma4ChatTemplateTests.swift](Tests/MLXLMTests/Gemma4ChatTemplateTests.swift).

---

## 9. Single-line Summary

No open memory / thread-safety bugs; Circular KV, Steel BD=256, Residency Sets, and the generation-dedicated stream are all live; Warp MoE + batched QKV fusion are implemented and regressed (stay off); graph caching AND Metal ICBs get paired CPU-overhead experiments on today's fused graph; genuinely open work is symbolic sliding-window mask → adaptive cmd-buffer → VLM roundtrips → batch decode → n-gram speculative → DFlash diffusion-based speculative port.