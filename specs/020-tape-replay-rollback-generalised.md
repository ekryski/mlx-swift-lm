# 020 — State-replay rollback for non-DFlash speculative decoders

**Status:** **Phases 1 + 2 + 3 consolidated and shipped** in PR [#143](https://github.com/ekryski/mlx-swift-lm/pull/143) (plus the cross-repo native-kernel chain [mlx#26](https://github.com/ekryski/mlx/pull/26) → [mlx-c#14](https://github.com/ekryski/mlx-c/pull/14) → [mlx-swift#25](https://github.com/ekryski/mlx-swift/pull/25) → mlx-swift-lm#143). GDN coverage for Qwen 3.5 / 3.6 is live. Phases 4 (DFlash iterator wiring) and 5 (PrefixKVCache integration) deferred to follow-up PRs once their parent specs land. **Mamba / Mamba 2 kernel support deferred** — Nemotron-H + Jamba cleanly opt out via per-cache `canStateReplay = false` and route through vanilla `TokenIterator`; future-work options ranked in §"Mamba / Mamba 2 follow-up (post-MVP)" below.

**Branch:** `ek/020-tape-replay-phase1` (PR #143)
**Depends on:** —. Originally specced as needing spec 015 (DFlash) phase 3's per-layer Mamba kernel first; that ordering inverted during implementation — this PR ships the state-replay primitive **first** and spec 015 phase 3 will refactor onto it.

## The insight

Today's `NGramSpeculativeTokenIterator` refuses to run on hybrid models (Qwen 3.5 / 3.6 GatedDeltaNet / Nemotron-H / Jamba / Granite-MoE-Hybrid / FalconH1) because their Mamba / SSM layers have non-trimmable cache state — there's no positional rollback. dflash-mlx solves this for DFlash with **state-replay rollback**: during the verify forward, record the per-step state-update deltas into a log; on partial acceptance, replay only the accepted prefix's deltas onto a snapshot taken at round entry.

**This trick is decoder-agnostic.** Once we have a state-replay primitive in the cache layer, *any* speculative decoder using that cache can do partial-accept rollback, including the n-gram iterator. The mamba constraint that has been blocking PLD on the entire Qwen 3.5 / 3.6 family becomes a non-issue.

This is the highest-leverage technical change in the speculative-decoding stack for this codebase.

(Note on naming: this was originally written using "innovation tape" — the Kalman-filter term for `B_t * x_t`. Renamed to "**delta log**" + "**state replay**" 2026-05-08 for readability. The math is unchanged.)

## What "state replay" means

A linear recurrent layer evolves its hidden state via `s_{t+1} = f(s_t, x_t, params)`. A typical SSM state update looks like:

```
s_{t+1} = A_t * s_t + B_t * x_t      (linear; can be batched / parallel-scanned)
y_t     = C_t * s_t + D_t * x_t
```

Standard cache: store `s_t` after every token. Standard rollback: snapshot `s_{t_0}` at speculation start; on partial accept of `k` tokens, snapshot becomes invalid, restore from `s_{t_0}` and redo `k` updates.

State-replay rollback: don't snapshot the **full state** at every speculation; record the per-step **state-update deltas** (`B_t * x_t`, written into a "delta log") during verify, plus the **start state** `s_{t_0}` once. On accept of `k` tokens, the post-rollback state is `s_{t_0 + k} = A^{(k)} * s_{t_0} + ∑_{i=0..k-1} A^{(k-1-i)} * (B_{t_0+i} * x_{t_0+i})`. The deltas were already computed during verify; we just re-fold them through the linear recurrence.

For non-linear recurrences (Mamba's gated delta is partially nonlinear), the same idea applies if the nonlinearity factorises through the per-step inputs — which is the case for Mamba-style layers per the dflash-mlx implementation.

## Design

### 1. New cache protocol

```swift
public protocol StateReplayCache: KVCache {
    /// Begin recording a delta log. Caller is committing to one of:
    ///   - `commitFull()` — accept all log entries; cache state advances to end-of-log.
    ///   - `rollback(acceptedPrefix:)` — accept first `k` entries; entries `k..<count` discarded.
    ///   - `cancel()` — reject all entries; cache state restored to pre-record snapshot.
    func beginRecord()

    /// Append the next per-step recurrence tensors (for GDN: `[delta_t, k_t, g_t]`).
    /// Called from inside `update(...)` during a recording session.
    func recordStep(_ tensors: [MLXArray])

    func commitFull()
    func rollback(acceptedPrefix k: Int)
    func cancel()

    /// Cost: replay-time complexity. Used by the iterator to decide whether
    /// state-replay is cheaper than re-running the layer from snapshot.
    var replayCost: StateReplayCost { get }  // .o1 (linear SSM), .ok (Mamba gated delta), .reforward (worst case)
}
```

`update(...)` becomes mode-aware: outside a recording session it behaves as today; inside one, it also calls `recordStep(...)` with the per-step recurrence tensors from the current input.

### 2. SSMStateCache conformance

This is the load-bearing change. Today `SSMStateCache.isTrimmable` is `false`. After this spec lands:

- `SSMStateCache: StateReplayCache` — supports beginRecord / recordStep / rollback.
- `isTrimmable` stays false (semantic: positional trim is not supported), but `canStateReplay` is true.

The Metal kernel for `rollback(acceptedPrefix:)` is the `state_replay` kernel in `mlx/mlx/backend/metal/kernels/gated_delta_replay.metal` (canonical, in mlx repo) / `mlx-swift/Source/Cmlx/mlx-generated/metal/gated_delta_replay.metal` (SPM mirror). Body matches dflash-mlx's `state-replay rollback` Metal kernel where possible, with the masked-timestep correctness fix (`3217e15`) and branchless `metal::select` pattern (`c9f992e`) adopted from day 1.

### 3. New cache trimmability predicate

```swift
public func canRollbackPromptCache(_ cache: [KVCache]) -> Bool {
    cache.allSatisfy { $0.isTrimmable || ($0 as? StateReplayCache) != nil }
}
```

This replaces `canTrimPromptCache` in the `MLXLMCommon.generate(...)` auto-routing decision. Pure-attention models still take the trim path (cheap, no log recording). Hybrid models take the state-replay path.

### 4. Iterator changes

`NGramSpeculativeTokenIterator.speculateRound` becomes:

```swift
beginCacheRecord(mainCache)        // recording mode: only fires on state-replay caches
let mainResult = mainModel(verifyInput, cache: mainCache, state: mainState)
// ... compute accepted ...
// Single dispatch handles both partial and full accept — the helper
// internally routes full-accept (rejected == 0) to commitFull() on
// state-replay layers and skips trim on trimmable ones.
rollbackPromptCache(mainCache, acceptedPrefix: accepted, numDraft: numDraft)
```

For pure-attention layers, `beginCacheRecord` is a no-op; `rollbackPromptCache` translates to the existing `trim(numTokens: rejected)`.

For state-replay layers (GDN), the helper dispatches the `state_replay` Metal kernel; full-accept skips the kernel entirely (the verify forward already advanced state through all T entries).

**Note on translation:** the helper passes `acceptedPrefix + 1` to the protocol's `rollback(acceptedPrefix:)` (NOT `acceptedPrefix`) because the verify forward records `T = numDraft + 1` entries — one for the y baseline (always kept) plus one per draft. This `+1` mapping is encapsulated in the helper; the iterator stays unaware of the entry count.

### 5. Memory budget

Originally this section called for capping the delta log at `MLX_STATE_REPLAY_MAX_STEPS=32` per round to bound memory at long context × wide verify windows. The Option A perf commit (`50ba77f`) inverted the recording shape: instead of `T = numDraft + 1` per-step entries per layer per round, we record **one per-round entry per layer** with the T-axis tensors intact. The actual storage is references into the kernel's existing output buffers — no fresh allocation. The cap was never needed and is not implemented. See Risk #2 below.

## Implementation phases

1. **Phase 1 ✅ DONE — Protocol + dispatch helpers (no concrete conformance).** Landed the protocol; concrete trimmable `KVCache`s (`StandardKVCache` / `AffineQuantizedKVCache`) get the `rollbackPromptCache(...)` free-function dispatch which routes to existing `trim(...)` semantics on `isTrimmable` caches. Iterators unchanged at this phase. Validates the API contract on workloads that already work.

2. **Phase 2 ✅ DONE — SSMStateCache + native Metal kernels.** Ported dflash-mlx's recurrent-rollback Metal kernel as `state_replay`, plus the companion forward-with-record kernel `gated_delta_step_record` that captures per-step `(delta_t, k_t, g_t)` into a delta log buffer. Native-C plumbing (mlx Primitive → mlx-c C ABI → mlx-swift Swift wrapper) — not JIT. Per-token equivalence tests against a sequential reference live in `Tests/MLXLMTests/StateReplayCacheTests.swift::SSMStateCacheStateReplayTests`.

3. **Phase 3 ✅ DONE — Wire into NGramSpeculativeTokenIterator.** Replaced the `canTrimPromptCache` gate with `canRollbackPromptCache` in `NgramSpeculativeDecoding.swift` (init guard) and `Evaluate.swift` (auto-route + asymmetric speculative init). `speculateRound` body wraps the verify forward in `beginCacheRecord` → `rollbackPromptCache` (or `commitCacheRecord` on full-accept). Bench validates the whole chain on Qwen 3.5 9B, 35B-A3B, and 3.6 27B; numbers in PR #143's bench-summary comment.

4. **Phase 4 — DEFERRED — Wire into DFlashSpeculativeTokenIterator.** Spec 015 (DFlash) phase 1 hasn't merged yet. When it does, that branch refactors onto the protocol from this spec rather than its own private path. One implementation, two consumers.

5. **Phase 5 — DEFERRED — Wire into PrefixKVCache.** With state replay, `SSMStateCache` becomes serialisable for prefix snapshots: snapshot the state, plus enough history to rebuild it. Spec 017 phase 3 depends on this. Will be picked up on spec 017's branch when this PR lands.

## Implementation status (phases 1 + 2 + 3 consolidated in PR #143)

**Multi-repo PR chain** (in dependency order, all on `ekryski/*` forks):

| # | Repo | PR | Title | Status |
|---|---|---|---|---|
| 1 | mlx | [#26](https://github.com/ekryski/mlx/pull/26) | `feat(fast): GatedDeltaStepRecord + StateReplay primitives` | Open |
| 2 | mlx-c | [#14](https://github.com/ekryski/mlx-c/pull/14) | `feat(fast): C ABI bridges for state-replay primitives` | Open |
| 3 | mlx-swift | [#25](https://github.com/ekryski/mlx-swift/pull/25) | `feat: native GDN state-replay primitives` | Open |
| 4 | mlx-swift-lm | [#143](https://github.com/ekryski/mlx-swift-lm/pull/143) | `feat(cache): spec 020 — StateReplayCache protocol + native GDN state-replay kernels + iterator wiring` | Open |

Originally specced as a single-repo Swift-JIT change; **revised mid-implementation to native-C kernels** following the established pattern of every other Metal kernel in this codebase. The mlx C++ `Primitive` classes (`GatedDeltaStepRecord`, `StateReplay`) live in PR #1, the C ABI in PR #2, the Swift wrappers in PR #3, and the cache + dispatch glue + iterator wiring in PR #4. Kernels are compiled into `mlx.metallib` via `make metal`.

### What ships in PR #143 (mlx-swift-lm)

**Protocol + helpers** (`Libraries/MLXLMCommon/StateReplayCache.swift`):
- `StateReplayCache` protocol with `beginRecord` / `recordStep(_ tensors: [MLXArray])` / `commitFull` / `rollback(acceptedPrefix:)` / `cancel`. Note: `recordStep` takes `[MLXArray]` from day 1 (not the spec's originally-narrow `_ delta: MLXArray`) so the SSM `[delta_t, k_t, g_t]` triple and future variants round-trip cleanly.
- `StateReplayCost` enum (`.o1` / `.ok` / `.reforward`).
- `StateReplayCacheError` enum (`alreadyRecording` / `notRecording` / `arityMismatch` / `outOfRange`) per upstream's `4bc72c8` fail-fast philosophy. The canonical `SSMStateCache` conformance uses Swift `precondition`s (programmer-error contract); the error enum is available for trim-fallback adapters.
- Dispatch helpers: `canRollbackPromptCache`, `beginCacheRecord`, `commitCacheRecord`, `rollbackPromptCache(_:acceptedPrefix:numDraft:)`, `cancelCacheRecord`.
- **Iterator → protocol semantic translation** documented inline on `rollbackPromptCache(...)`. The verify forward records `T = numDraft + 1` entries per state-replay cache (y baseline + `numDraft` drafts), so the helper passes `acceptedPrefix + 1` to the protocol — encapsulating the +1 so the protocol stays a low-level "fold first k entries" primitive and the iterator stays unaware of the entry count.

**SSMStateCache conformance** (`Libraries/MLXLMCommon/KVCache.swift::SSMStateCache`):
- `: StateReplayCache` extension with a delta-log buffer (`[[MLXArray]]`), pre-record snapshot, `beginRecord` / `recordStep` / `commitFull` / `rollback(acceptedPrefix:)` / `cancel`.
- `canStateReplay` is a **per-instance stored property** (default `true`) so Mamba-using factories can opt out without subclassing.
- `isRecording: Bool` is `public` so layer code can gate the kernel-routing branch.
- `rollback(acceptedPrefix:)` reads `state[1]` (recurrent state, 4D) — **not** `state[0]` (conv state, 3D); the wrong-slot bug was a `cd63ac0` fix.
- State is kept in fp32 throughout the GDN recurrence (matching the existing forward kernel at `GatedDelta.swift:387`). The replay kernel inherits this — no extra fp32 promotion needed at rollback time. This is what mitigates "fp32-accumulator drift" (Risk 1 below).

**Native Metal kernels** (mlx repo + mlx-swift sibling mirror):
- `gated_delta_step_record` — extends the forward `gated_delta_step` to write per-step `delta_t` into a tape output buffer alongside `(y, state_out)`.
- `state_replay` — re-folds the accepted prefix from a `[B, T_log, Hv, *]` delta log onto a state snapshot.
- Both adopt upstream's `3217e15` masked-timestep fix (save `old_state` before each step; `metal::select(old_state, new_val, do_step)` on masked positions) and `c9f992e` branchless pattern from day 1. Verified via direct read of `mlx/mlx/backend/metal/kernels/gated_delta_replay.metal:200-216`.
- Function-constant variants for masked vs. non-masked. Kernel cells instantiated for the GQA-asymmetric Qwen 3.5 shapes (Hk=16, Hv=32) AND the GQA-symmetric Hk=Hv shapes (32×32, 48×48 in fp16/bf16/fp32) since the replay kernel sees GQA-expanded k.

**Forward-with-record dispatcher** (`Libraries/MLXLMCommon/StateReplayKernels.swift::stateReplayUpdate` + `gatedDeltaKernelRecord`):
- Routes through `MLXFast.stateReplay` / `MLXFast.gatedDeltaStepRecord`.
- Auto-detects record format: T-axis (per-round, preferred — one `recordStep` per verify forward) vs. per-step (legacy — one `recordStep` per token within a round; still used by some tests).

**GDN layer integration** (`Libraries/MLXLLM/Models/GatedDelta.swift::gatedDeltaUpdateRecord`):
- When the cache is recording (`cache.isRecording == true`), dispatches the with-tape kernel and emits **one per-round `recordStep([delta, k, g])`** with the T-axis tensors intact (Option A — performance commit `50ba77f` batched this from T calls to 1 call per layer per round, ~30× fewer Swift dispatches on a 30-GDN-layer model).
- Call sites updated: `Qwen3Next.swift::Qwen3NextGatedDeltaNet.update(...)` and `Qwen35.swift::Qwen35GatedDeltaNet.update(...)`. The S==1 fused-decode path (`fusedGatedDeltaUpdate`) is **not** modified — speculative verify is always S>1, the fused path never sees a recording session.

**Iterator wiring** (`Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` + `Libraries/MLXLMCommon/Evaluate.swift`):
- Three `canTrimPromptCache` → `canRollbackPromptCache` predicate flips: N-gram iterator init, asymmetric `SpeculativeTokenIterator` init, `generate(...)` auto-route gate.
- `speculateRound` body wraps the verify forward: `beginCacheRecord(mainCache)` immediately before; on partial accept `rollbackPromptCache(mainCache, acceptedPrefix:, numDraft:)`; full accept routes through `commitFull()` internally via the helper.

**Mamba opt-out** (`Libraries/MLXLLM/Models/NemotronH.swift:810`, `Libraries/MLXLLM/Models/Jamba.swift:489`):
- Cache factories set `canStateReplay = false` on their `.mamba` cache slots. `canRollbackPromptCache` then returns `false` for the stack; iterator declines and falls back to `TokenIterator`. No crash, identical to pre-spec-020 behaviour. Future-work options to lift this restriction are documented in §"Mamba / Mamba 2 follow-up (post-MVP)".

**Tests** (`Tests/MLXLMTests/StateReplayCacheTests.swift`, `Tests/MLXLMTests/GDNRecordPathTests.swift`):
- 24 tests across 3 suites:
  - `CanRollbackPromptCacheTests` (6 fake-cache predicate tests),
  - `RollbackPromptCacheTests` (10 dispatch / off-by-one / mixed-stack tests),
  - `SSMStateCacheStateReplayTests` (8 end-to-end tests with a real `SSMStateCache` + the native kernels: commit, cancel, rollback at k=0/k=mid/k=full, per-step equivalence, free-helper dispatch with off-by-one regressions).
- `GDNRecordPathTests` (2 tests) — direct kernel-level exercise of `gatedDeltaUpdateRecord` at Qwen 3.5 9B shapes.

### Bench validation

M1 Max 64GB, 4bit weights, no-quant KV, T=0, 1000-token decode (recipe-bulk prompt). Full numbers in PR #143's bench-summary comment; headline rows:

- Qwen3.5-9B-4bit: D=4 1.03× baseline (77% accept).
- Qwen3.5-35B-A3B-4bit: D=4 **1.37× baseline** (83% accept); D=12 adapt+strict **1.64× baseline** (92% accept).
- Qwen3.6-27B-4bit: correctness OK, perf 0.94×–0.52× (dense 27B + freeform thinking-output is a poor spec-decode target; not a regression).
- Nemotron Cascade 2: cleanly opts out, identical to pre-spec-020 baseline.

## dflash-mlx upstream updates since spec drafted (historical)

This spec was originally drafted against `bstnxbt/dflash-mlx@engine-v2`. `main` (HEAD `8d8545d`, 2026-05-04) added material updates that were treated as **hard requirements** for our kernel work — all adopted from day 1 in the shipped implementation:

- **Masked-timestep correctness fix** ([`3217e15`](https://github.com/bstnxbt/dflash-mlx/commit/3217e15), 2026-04-22) — `gated_delta_step_record` and `state_replay` save `old_state` before each step, gate the decay/accumulate block on a uniform `do_step` predicate, restore via `metal::select(new_val, old_state, do_step)` on masked positions. **Adopted** — verified in `mlx/mlx/backend/metal/kernels/gated_delta_replay.metal:200-216`.
- **Branchless Metal kernel pattern** ([`c9f992e`](https://github.com/bstnxbt/dflash-mlx/commit/c9f992e), 2026-04-22) — `metal::select` everywhere instead of `if(...)` guards. **Adopted** — same file.
- **Function-constant variants**: upstream's `_make_gated_delta_kernel_with_tape(has_mask, vectorized)` is shipped as masked vs. non-masked specializations via Metal function constants. Our equivalent: function-constant variants in `gated_delta_replay.metal`, instantiated for fp16 / bf16 / fp32 × the dim cells Qwen 3.5 / 3.6 actually exercise.

## dflash-mlx post-`8d8545d` deltas (2026-05-08 sync)

Three upstream commits land between `8d8545d` and current `main` HEAD. The kernel surface is **unchanged** (our notes on `3217e15` masked-timestep + `c9f992e` branchless patterns stay authoritative), but two adjacent changes inform phase 2 / phase 3:

- **`4bc72c8` (2026-05-10) — "fix: harden runtime and cache contract failures"** — *adopted.*
  - Established the **fail-fast contract philosophy** for cache lifecycle. `StateReplayCacheError` (in `StateReplayCache.swift`) carries the four cases (`alreadyRecording`, `notRecording`, `arityMismatch`, `outOfRange`). The canonical `SSMStateCache` conformance uses Swift `precondition`s (the iterator is the only caller — programmer-error contract). The error enum is available for trim-fallback adapters in future cache types that want a throws-based discipline.

- **`05cc456` / `2274b67` (2026-05-06 / 2026-05-10) — Gemma4 DFlash backend + cache policy / quant provenance fix**
  - Gemma4 is pure-attention (no GDN layers), so the state-replay path is a no-op for it — `canRollbackPromptCache` evaluates `true` because every layer is trimmable. The asymmetric speculative iterator (`Evaluate.swift`) already runs Gemma4 through the same code path; no extra smoke test needed.
  - The "cache policy and quant provenance" fix doesn't interact directly with state replay. If our Qwen 3.5 35B-A3B path eventually adds TurboQuant on its attention layers, the state-replay path's mixed-stack handling (state-replay on GDN, trim on attention via `rollbackPromptCache`) is already validated by the `Mixed trimmable + state-replay stack passes` test.

- **Survival-gate methodology (commit `8c29e3e`, 2026-05-05)** — defined in spec 017's post-`8d8545d` section. End-to-end rollback round-trip equivalence is covered by the `SSMStateCacheStateReplayTests` suite — partial-accept rollback matches a hand-folded reference recurrence within bf16 tolerance.

**Reference commits:** `4bc72c8`, `05cc456`, `2274b67`, `8c29e3e`. Cross-relevant but not state-replay-shaping: `463d722` (prefix-cache format-version bump — touches spec 017), runtime-refactor cluster `ce36f62` / `0972afb` / `e2be8a4`.

## Expected impact

For PLD specifically: the entire **Qwen 3.5 / 3.6 family** becomes accessible. Today the auto-routing falls back to TokenIterator at parity; with state-replay rollback the iterator engages the n-gram path. Workload-dependent — but on input-grounded prompts where Gemma 4 26B A4B sees +25%, Qwen 3.5 9B should see comparable or better (it's already a smaller model with faster verify).

For DFlash: spec 015's reported 2.2-4.4× becomes architecturally cleaner — the state-replay primitive is shared, no separate code path.

For PrefixKVCache: hybrid models become cacheable. Multi-turn TTFT on Qwen 3.5 9B drops from ~3s on turn N+1 to ~0.5s (mostly suffix prefill).

## Risks

1. **Numerical equivalence under state replay** — *mitigated.* dflash-mlx upstream stabilises bf16-sensitive paths via fp32 accumulators. Our equivalent: `gatedDeltaUpdate` / `gatedDeltaUpdateRecord` keep the recurrent state in `.float32` throughout the GDN loop (`GatedDelta.swift:387, 438`), so the replay kernel receives an fp32 snapshot and folds entries through an fp32 recurrence — no precision loss across rollback boundaries. Equivalence is locked by:
   - `SSMStateCacheStateReplayTests::rollback(acceptedPrefix:) re-folds first k delta log entries via the GDN recurrence` — partial-accept matches a hand-computed reference via `referenceStep` (the same recurrence the kernel implements).
   - `Per-step equivalence: rolling 1+1 == rolling 0+2 within a round` — proves the kernel's fold order is consistent regardless of where the round boundary sits.
   - `rollback(acceptedPrefix: numDraft) matches a full-accept replay` — full-prefix equivalence.

2. **Memory pressure at long context + large verify windows** — *no longer a meaningful concern.* The original spec called for a hard cap (`MLX_STATE_REPLAY_MAX_STEPS=32`) to bound the delta log. The performance commit `50ba77f` (Option A) reshapes recording to **one entry per verify round per layer with the T-axis tensors intact** rather than T entries per round — so for a `numDraft=12` verify window on ~30 GDN layers, the per-round delta log is 30 entries (not 360). The MLXArray storage is references to slices of the kernel's existing output tape, not fresh allocations. The cap is unnecessary and was not implemented; revisit only if a future change reverts to per-step recording.

3. **Mamba / Mamba 2 variants** — *opted out, future work planned.* GDN coverage is in (Qwen 3.5 / 3.6). Mamba (Nemotron-H) and Mamba 2 (Jamba) use a different recurrence (`s_{t+1} = exp(dt·A)·s_t + dt·B·x` — 3D state, selective) and their S>1 forward uses a chunked parallel scan that doesn't materialise per-step `(dA, dB·x)`. **Mitigation in this PR:** `SSMStateCache.canStateReplay` is per-instance configurable (default `true`); `NemotronH.swift` and `Jamba.swift` cache factories set `false` on their Mamba slots. `canRollbackPromptCache` returns `false`, iterator declines, fallback to `TokenIterator` is identical to pre-spec-020 behaviour — no crash, no surprise. **Future work:** three options (native Mamba kernels / sequential `ssm_step` fallback / augmented `ssmAttn`) ranked in §"Mamba / Mamba 2 follow-up (post-MVP)" below; preferred option is native kernels with the same 4-repo PR chain shape as this PR.

4. **Built against the post-spec-006 KV-cache hierarchy** — *no longer a risk; integration complete.* Spec 006 (KVCache type consolidation, issue #73) merged in PRs #163–#166 before this work started. `SSMStateCache` (subclass of `ArraysCache` with `[MLXArray?]` slot storage) was the canvas; the `StateReplayCache` conformance is a clean addition. `SSMStateCache` reports `KVStorageKind.ssm` and stays outside the K/V storage/eviction axes spec 006 reorganised — verified during implementation, no friction.

5. **Masked-timestep correctness** — *mitigated.* Verified in `mlx/mlx/backend/metal/kernels/gated_delta_replay.metal:200-216`: the kernel saves `old_state[i] = state[i]` before each step and restores via `metal::select(old_state[i], new_val, do_step)` on masked / out-of-range positions, matching upstream's `3217e15` pattern. The `do_step` predicate composes both the `t < accepted` range check AND the mask, so both rollback truncation and intra-round masking go through the same correctness path.
   - **Gap acknowledged:** an explicit `testMaskedTimesteps` unit test (as called for in the original spec) was **not** added because every current call site in the codebase passes `mask = nil` — masked replay is not yet exercised end-to-end. The kernel correctness is locked at the source level; adding a `mask`-bearing test is a follow-up if/when a user surfaces masked replay (e.g. DFlash phase 4, batched mixed-attention).

6. **Branchless kernel discipline** — *mitigated.* Same `metal::select`-based control flow (no divergent `if`) used throughout `gated_delta_replay.metal`. SIMD utilisation matches upstream's `c9f992e` pattern; verified by source inspection.

7. **NEW: Off-by-one in iterator → protocol translation** — *mitigated, regression-tested.* The verify forward records `T = numDraft + 1` entries per round (y baseline + `numDraft` drafts), but the iterator's `acceptedPrefix` counts accepted draft tokens (0..numDraft). The state-replay path was initially translating `acceptedPrefix` directly to `rollback(acceptedPrefix:)`, folding one fewer entry than the trim path kept — causing SSM-state drift on every partial accept. Worse: `acceptedPrefix == 0` routed through `cancel()` which lost y entirely. Fix (`be3ecff`): helper `rollbackPromptCache(...)` translates to `rollback(acceptedPrefix + 1)` and the zero-accept case folds `rollback(1)` (keep y baseline). Two regression tests in `StateReplayCacheTests::RollbackPromptCacheTests` lock the helper's translation; one end-to-end test in `SSMStateCacheStateReplayTests` locks it against a real `SSMStateCache` + real referenceStep recurrence. The bug had a dramatic perf footprint — Qwen3.5-35B-A3B D=12 acceptance jumped 34% → 92% post-fix — so regression coverage on this is high-value.

## Files touched (final, post-PR-#143)

**mlx-swift-lm** (PR [#143](https://github.com/ekryski/mlx-swift-lm/pull/143)):

| File | What |
|---|---|
| `Libraries/MLXLMCommon/StateReplayCache.swift` (new) | `StateReplayCache` protocol with `recordStep(_ tensors: [MLXArray])` (general from day 1; takes the `[delta, k, g]` triple for GDN, extensible for future variants). `StateReplayCost`, `StateReplayCacheError`. Free helpers: `canRollbackPromptCache`, `beginCacheRecord`, `commitCacheRecord`, `rollbackPromptCache(_:acceptedPrefix:numDraft:)`, `cancelCacheRecord`. The `rollbackPromptCache` helper encapsulates the `acceptedPrefix + 1` translation so the protocol stays a low-level "fold first k entries" primitive. |
| `Libraries/MLXLMCommon/KVCache.swift` (`SSMStateCache` at line 1335) | `SSMStateCache: StateReplayCache` extension — delta log (`[[MLXArray]]`), pre-record snapshot, `canStateReplay` stored property (default `true`, per-instance configurable), `isRecording: Bool` public, `beginRecord` / `recordStep` / `commitFull` / `rollback(acceptedPrefix:)` / `cancel`. Rollback reads `state[1]` (recurrent), not `state[0]` (conv). |
| `Libraries/MLXLMCommon/StateReplayKernels.swift` (new) | Swift dispatch wrappers around the native `MLXFast.stateReplay` / `MLXFast.gatedDeltaStepRecord` primitives. Auto-detects T-axis vs. per-step record format. |
| `Libraries/MLXLLM/Models/GatedDelta.swift` | `gatedDeltaUpdateRecord(...)` — record-aware wrapper around `gatedDeltaUpdate`. When `cache.isRecording`, dispatches the with-tape kernel and emits one per-round `recordStep([delta, k, g])` with T-axis tensors intact (Option A, ~30× fewer Swift dispatches vs. per-step). |
| `Libraries/MLXLLM/Models/Qwen3Next.swift::Qwen3NextGatedDeltaNet.update(...)` | `gatedDeltaUpdate` → `gatedDeltaUpdateRecord` with cache argument. |
| `Libraries/MLXLLM/Models/Qwen35.swift::Qwen35GatedDeltaNet.update(...)` | S>1 prefill/verify path: `gatedDeltaUpdate` → `gatedDeltaUpdateRecord`. S==1 fused-decode path stays on `fusedGatedDeltaUpdate` (never sees a recording session). |
| `Libraries/MLXLLM/Models/NemotronH.swift::newCache(...)` | Mamba opt-out: `cache.canStateReplay = false` on `.mamba` cache slots. |
| `Libraries/MLXLLM/Models/Jamba.swift::newCache(...)` | Mamba opt-out: `cache.canStateReplay = false` on the non-attention cache slots. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Two predicate flips: `canTrimPromptCache` → `canRollbackPromptCache` at the asymmetric `SpeculativeTokenIterator` init guard and at the auto-route gate in `generate(...)`. |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | Predicate flip at iterator init + `speculateRound` body wrap: `beginCacheRecord(mainCache)` before the verify forward, `rollbackPromptCache(...)` on partial accept (helper internally routes full-accept to `commitFull`). |
| `Tests/MLXLMTests/StateReplayCacheTests.swift` (new) | 24 tests across 3 suites: predicate (6), dispatch helpers (10), end-to-end `SSMStateCache` (8). |
| `Tests/MLXLMTests/GDNRecordPathTests.swift` (new) | 2 direct kernel-level tests on Qwen 3.5 9B shapes. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | `MLX_BENCH_DEBUG` env-gated crash-stack capture surface (`BenchEnv.debugCrashCapture`, writes to `/tmp/mlx-bench-crash.log` before `fatalError`). |
| `scripts/benchmark.sh` | `--debug` flag wires `MLX_BENCH_DEBUG=1`. |
| `Libraries/MLXLMCommon/DFlashSpeculativeDecoding.swift` | **DEFERRED.** Predicate flip lands when DFlash phase 1 (PR #141) merges. |
| `Libraries/MLXLMCommon/MirrorSpeculativeDecoding.swift` | **DEFERRED.** Predicate flip lands when Mirror SD phase 1 (PR #142) merges. |

**Sibling repos** (kernels + plumbing):

| Repo / file | What |
|---|---|
| `mlx` PR #26 — `mlx/mlx/backend/metal/kernels/gated_delta_replay.metal` | Native Metal kernels: `gated_delta_step_record` (forward + delta-log capture) and `state_replay` (rollback). Adopts `3217e15` masked-timestep fix + `c9f992e` branchless pattern. Function-constant variants for masked vs. non-masked. Cells for fp16/bf16/fp32 × Hk×Hv shapes (asymmetric for Qwen 3.5 GQA + symmetric Hk=Hv for the GQA-expanded replay path). |
| `mlx` PR #26 — `mlx/mlx/backend/metal/gated_delta_replay.cpp` + headers | C++ `Primitive` classes: `GatedDeltaStepRecord`, `StateReplay`. |
| `mlx-c` PR #14 | C ABI bridges (`mlx_fast_gated_delta_step_record`, `mlx_fast_state_replay`). |
| `mlx-swift` PR #25 | `MLXFast.gatedDeltaStepRecord(...)` / `MLXFast.stateReplay(...)` Swift wrappers + canonical `.metal` mirror at `Source/Cmlx/mlx-generated/metal/gated_delta_replay.metal`. |

## Mamba / Mamba 2 follow-up (post-MVP)

Phase 2 (this PR) covers the **GatedDeltaNet** recurrence — Qwen 3.5 / 3.6 and any future GDN model. **Mamba** (Nemotron Cascade 2) and **Mamba 2** (Jamba, Granite-MoE-Hybrid, FalconH1) share the `SSMStateCache` storage class but use a *different* recurrence:

```
GDN:    s_{t+1} = g_t · s_t + k_t · δ_t                        (4D state, GQA-expanded k)
Mamba:  s_{t+1} = exp(dt_t · A) · s_t + dt_t · B_t · x_t       (3D state, selective)
```

The math is structurally analogous (decay × state + innovation), but the production S>1 forward path differs in a way that blocks dropping in the GDN kernels:

- **GDN's `gated_delta_step` (in `gated_delta.metal`) computes the recurrence sequentially** in a per-step Metal loop. Adding per-step `delta_t` capture is local — a single buffer write inside the loop. That's what `gated_delta_step_record` does in this PR.
- **Mamba's S>1 forward uses a chunked parallel scan** (`ssmAttn` in `Libraries/MLXLLM/Models/SSM.swift:145`) that computes the entire T-token block via a surrogate-attention matmul. Per-step `(dA_t, dB_t · x_t)` values are *not* materialised — they're rolled into the parallel-scan reduction.

Three implementation options for Mamba follow-up, ranked by quality:

1. **Native `ssm_step_record` + `ssm_replay` kernel pair** (preferred). New Metal kernels + C++ Primitive + mlx-c bridge + mlx-swift Swift wrapper, parallel to `gated_delta_step_record` / `state_replay`. ssm_step_record emits per-step `(dA, dB·x)` alongside the y/state_out outputs; ssm_replay re-folds them via `s = dA·s + dB·x` on rollback. Scope is similar to this PR's 4-repo chain (~1500 LOC). Ships independently of GDN.

2. **Sequential ssm_step fallback during recording.** When `cache.isRecording`, the layer switches from `ssmAttn` (parallel scan, fast for T>1) to the existing `ssmUpdateKernel` (sequential T-loop) and captures per-step values from each iteration. Simpler — no new kernels — but verify-forward throughput drops ~5–10× because chunked parallel scan is much faster than T sequential calls. Probably acceptable for verify windows ≤ 16 tokens, but eats most of the n-gram speedup margin.

3. **Augmented `ssmAttn` that also emits per-step deltas.** Computes the per-step `(dA, dB·x)` *in addition to* the chunked scan output. Doubles the layer's compute but keeps the parallel-scan throughput. Cleanest behaviour, but kernel surgery is invasive.

### Until then

`SSMStateCache.canStateReplay` is **per-cache configurable** (stored property on the class, default `true`). Mamba-using model factories opt out by setting `canStateReplay = false` on the SSM caches they emit:

- `Libraries/MLXLLM/Models/NemotronH.swift::newCache(...)` — sets `false` on the `.mamba` layer caches.
- `Libraries/MLXLLM/Models/Jamba.swift::newCache(...)` — sets `false` on the non-attention layer caches.

When `canStateReplay == false` on any layer, `canRollbackPromptCache` returns `false` for the stack, and `MLXLMCommon.generate(...)` auto-routing gracefully declines the n-gram speculative path → falls back to vanilla `TokenIterator`. No crash, no surprise; users get baseline AR decode on Mamba models with a `TokenIterator` traceback that's identical to the pre-spec-020 behaviour.

When the Mamba kernels land, the model factories flip the flag back to `true` (or just drop the setter — `true` is the default).

## Out of scope

- Tree-attention rollback. Tree shape is independent of recurrence type. When spec 014 lands tree attention on pure-attention models, this spec extends the same kernel to handle multi-path state replay (rollback to a non-linear position in a draft tree). Different spec.
- Variable per-layer recording. Today the iterator either records all layers or none. A finer-grained "only Mamba layers record" optimisation is possible but not necessary at our scale.

## References

- **Primary upstream (post-correctness-fix):** [`bstnxbt/dflash-mlx@main`](https://github.com/bstnxbt/dflash-mlx) HEAD `8d8545d` (2026-05-04).
  - [`dflash_mlx/kernels.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/kernels.py) — `_make_gated_delta_kernel_with_tape(...)` + `_make_tape_replay_kernel(...)` + runtime entrypoint `state_replay_kernel(tape, k, g, state, mask)`. **The canonical kernel reference for phase 2.**
  - [`dflash_mlx/recurrent_rollback_cache.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/recurrent_rollback_cache.py) — `RecurrentRollbackCache` class with the delta log lifecycle (`prepare` / `finalize` / `extend` / `extract` / `advance` / `make_mask`).
  - Commit [`3217e15`](https://github.com/bstnxbt/dflash-mlx/commit/3217e15) — masked-timestep correctness fix (hard requirement).
  - Commit [`c9f992e`](https://github.com/bstnxbt/dflash-mlx/commit/c9f992e) — branchless metal kernels (adopt from day 1).
- [dflash-mlx engine-v2 `recurrent_rollback_cache.py`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/recurrent_rollback_cache.py) — original Python reference (superseded by main; kept for historical comparison).
- [dflash-mlx engine-v2 `engine/rollback.py`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/engine/rollback.py) — full-accept vs. partial-accept paths (still the same conceptual flow on main).
- [Mamba: Linear-time sequence modeling with selective state spaces (Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752) — the recurrence we're rolling back.
- [GatedDeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2412.06464) — Qwen 3.5's specific variant.
- [Issue #73 / spec 006](https://github.com/ekryski/mlx-swift-lm/issues/73) — KVCache refactor that lands before this phase 2; provides the cleaner cache surface this spec's cross-cutting `StateReplayCache` extension targets.
