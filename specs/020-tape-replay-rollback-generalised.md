# 020 — State-replay rollback for non-DFlash speculative decoders

**Status:** spec, ready to issue (high-leverage, technically the deepest piece)
**Branch:** new branch off post-013 / post-015 phase 2
**Depends on:** spec 015 (DFlash) phase 3 lands the per-layer Mamba kernel; this spec generalises it.

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
if accepted == numDraft {
    commitCacheRecord(mainCache)   // full accept; delta log discarded, state already advanced
} else {
    rollbackPromptCache(mainCache, acceptedPrefix: accepted, numDraft: numDraft)
}
```

For pure-attention layers, `beginCacheRecord/commitCacheRecord` are no-ops; `rollbackPromptCache` translates to the existing `trim(numTokens: rejected)`.

For Mamba layers, the calls go through the `state_replay` Metal kernel; full-accept skips the kernel entirely.

### 5. Memory budget

The delta log stores `(numDraft + 1)` per-step records per Mamba layer per round. For Qwen 3.5 9B with ~30 GDN layers and a 16-token verify window, that's ~30 × 16 × `state_dim` MLXArrays = a few MB per round. Cleared on commit/rollback.

For verify windows above ~32 tokens this becomes large; cap the log size at `MLX_STATE_REPLAY_MAX_STEPS=32` and force a fall-back to AR if a round wants to draft more than that.

## Implementation phases

1. **Phase 1 — Protocol + dispatch helpers (no concrete conformance).** Land the protocol; concrete trimmable `KVCache`s (`StandardKVCache` / `AffineQuantizedKVCache`) get the `rollbackPromptCache(...)` free-function dispatch which routes to existing `trim(...)` semantics on `isTrimmable` caches. Iterators unchanged. Validates the API contract on workloads that already work.

2. **Phase 2 — SSMStateCache + Metal kernel.** This is the meat. Port dflash-mlx's recurrent-rollback Metal kernel as `state_replay`. Add a companion forward-with-record kernel `gated_delta_step_record` that captures per-step `delta_t` into a delta log buffer. Test against per-token equivalence with a sequential (no rollback) reference run.

3. **Phase 3 — Wire into NGramSpeculativeTokenIterator.** Replace the `canTrimPromptCache` gate with `canRollbackPromptCache`. Run the full benchmark suite on Qwen 3.5 / 3.6 — this is where PLD's coverage doubles overnight.

4. **Phase 4 — Wire into DFlashSpeculativeTokenIterator.** Spec 015's phase 3 already uses state replay; refactor it to use the protocol from this spec rather than its own private path. One implementation, two consumers.

5. **Phase 5 — Wire into PrefixKVCache.** With state replay, `SSMStateCache` becomes serialisable for prefix snapshots: snapshot the state, plus enough history to rebuild it. Spec 017 phase 3 depends on this.

## Phase 1 status (PR #143, ready for review)

Phase 1 lives in [PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143), rebased onto current `alpha` (post spec-006 / WS-A–D / consolidation sprint):

- `Libraries/MLXLMCommon/StateReplayCache.swift` — `StateReplayCache` protocol + dispatch helpers (`canRollbackPromptCache`, `beginCacheRecord`, `commitCacheRecord`, `rollbackPromptCache`, `cancelCacheRecord`), `StateReplayCost` enum (`.o1` / `.ok` / `.reforward`).
- `Tests/MLXLMTests/StateReplayCacheTests.swift` — 16 fake-cache tests across three suites (`CanRollbackPromptCacheTests`, `RollbackPromptCacheTests`, `CacheRecordHelpersTests`). Pure-Swift, no MLX evaluation, no metallib needed.

**Open work for phase 2 (kernel + `SSMStateCache` conformance):**

- `SSMStateCache` (subclass of `ArraysCache` in `Libraries/MLXLMCommon/KVCache.swift`) currently has no state-replay state. Today it stores `[MLXArray?]` slots through its `ArraysCache` parent — clean canvas for adding the delta log storage. Phase 2 adds the delta log + `recordStep` / `commitFull` / `rollback(acceptedPrefix:)` / `cancel` implementations.
- The protocol's `recordStep(_ delta: MLXArray)` signature is **too narrow** for the GDN recurrence — upstream's [`state_replay_kernel`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/kernels.py) takes three log arrays per step (`delta_log, k_log, g_log`) plus the current `state`, not a single delta. Phase 2 generalises `recordStep` to take `[MLXArray]` (or a typed `GDNInnovation` struct) so the protocol round-trips the upstream kernel surface. Phase 1's narrower form ships first to land the dispatch contract; phase 2 widens it as part of the conformance work.

**Open work for phase 3 (iterator wiring):**

- `NGramSpeculativeTokenIterator.speculateRound` (`Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift`) gates on `canTrimPromptCache(self.mainCache)` and throws on hybrid caches. Same gate exists in `Evaluate.swift` for auto-routing (in `generate(input:cache:parameters:context:draftModel:...)` and the parallel SpeculativeTokenIterator init) and in the DFlash / Mirror SD scaffolds (in their phase-1 PRs #141 / #142). Phase 3 swaps each to `canRollbackPromptCache`.
- The `speculateRound` body needs `beginCacheRecord(mainCache)` → forward → `commitCacheRecord` / `rollbackPromptCache(...)` / `cancelCacheRecord` based on accept count.

Specific line numbers are not pinned — they drift across rebases. The symbol names `canTrimPromptCache`, `canRollbackPromptCache`, `speculateRound`, `beginCacheRecord` etc. are the contract.

**Multi-repo PR chain (corrected from initial spec):**

The kernel ships in the **`mlx-swift` sibling repo** at `Source/Cmlx/mlx-generated/metal/`, **not** in mlx-swift-lm. The Eric-authored `gated_delta.metal` is already there as the structural template. Choose: extend `gated_delta.metal` with `gated_delta_step_record` + `state_replay` kernel variants, **or** add a sibling `gated_delta_replay.metal` (recommended: cleaner diff review). Built into `mlx.metallib` via `make metal` (driven by `scripts/build-metallib.sh`); no JIT. Likely **no `mlx-c` / `mlx` PRs needed** since the kernel only references already-present headers (`bf16.h`, `metal_simdgroup`, `utils.h`).

## dflash-mlx upstream updates since spec drafted

This spec was drafted against `bstnxbt/dflash-mlx@engine-v2`. `main` (HEAD `8d8545d`, 2026-05-04) contains material updates that are **hard requirements** for our phase 2 kernel:

- **Masked-timestep correctness fix** (commit [`3217e15`](https://github.com/bstnxbt/dflash-mlx/commit/3217e15), 2026-04-22): the `gated_delta_step_record` and `state_replay` kernels now save `old_state` before each step, gate the entire decay/accumulate/quantize block on a uniform `do_step` predicate, and restore via `metal::select(new_state, old_state, do_step)` on masked positions. **Without this, masked positions silently corrupt state.** Phase 2 must implement this from day 1 — see Risks #5 below.
- **Branchless Metal kernel pattern** (commit [`c9f992e`](https://github.com/bstnxbt/dflash-mlx/commit/c9f992e), 2026-04-22): state replay uses `mask_gate = float(mask)` multiply instead of `if(mask)` guards; SDPA partials use `metal::select(-inf, score, use_key)`; SDPA reduce uses `inv_sum = 1/max(sum, 1e-9)`. Better SIMD utilisation, no divergent execution. **Adopt from day 1** — see Risks #6 below.
- **4-variant kernel surface**: upstream ships `_make_gated_delta_kernel_with_tape(has_mask, vectorized)` and `_make_tape_replay_kernel(has_mask, vectorized)` — 4 templated variants each (vec × non-vec, masked × non-masked). Our LoC estimate is ~400-500 (was ~150-300).

## dflash-mlx post-`8d8545d` deltas (2026-05-08 sync)

Three upstream commits land between `8d8545d` and current `main` HEAD. The kernel surface is **unchanged** (our notes on `3217e15` masked-timestep + `c9f992e` branchless patterns stay authoritative), but two adjacent changes inform phase 2 / phase 3:

- **`4bc72c8` (2026-05-10) — "fix: harden runtime and cache contract failures"**
  - Establishes the **fail-fast contract philosophy** for cache lifecycle. For `StateReplayCache`, adopt from day 1:
    - `beginRecord()` while a delta log is already active → throws `StateReplayCacheError.alreadyRecording`.
    - `recordStep([...])` outside a recording session → throws `StateReplayCacheError.notRecording`.
    - `recordStep` with wrong arity (SSMStateCache expects exactly 3: `[delta, k, g]`) → throws `StateReplayCacheError.arityMismatch(expected: Int, got: Int)`.
    - `rollback(acceptedPrefix: k)` with `k < 0 || k > log.count` → throws `StateReplayCacheError.outOfRange(k: Int, logLength: Int)`.
    - `commitFull()` / `cancel()` outside a recording session → throws `.notRecording`.
  - Mirrors upstream's `RuntimeCacheManagerClosed` / `ValueError` discipline. No silent fallbacks across the delta log lifecycle — the iterator must `try` every terminator. Phase 2 ships `StateReplayCacheError` alongside the `SSMStateCache` conformance.

- **`05cc456` / `2274b67` (2026-05-06 / 2026-05-10) — Gemma4 DFlash backend + cache policy / quant provenance fix**
  - Upstream now supports Gemma4 (31B + 26B-A4B) end-to-end through the dflash engine, including state replay. Gemma4 is pure-attention (no GDN layers), so the **state-replay path is a no-op** for it — `canRollbackPromptCache` evaluates `true` because every layer is trimmable. Phase 3 smoke-tests this explicitly: with Gemma4 26B-A4B + n-gram speculative, the iterator must take the state-replay path (matching `canRollbackPromptCache`) and not regress on existing Gemma4 acceptance rates.
  - The "cache policy and quant provenance" fix is a TurboQuant-adjacent contract tightening; not directly state-replay, but if our Qwen 3.5 35B-A3B path uses TurboQuant on its attention layers, phase 3 must verify the state-replay path composes correctly with `TurboQuantizedKVCache.trim(...)` on full-accept rounds.

- **Survival-gate methodology (commit `8c29e3e`, 2026-05-05)** — defined in spec 017's post-`8d8545d` section. For spec 020 specifically: phase 3 adds an n-gram-on-Qwen-3.5 smoke that **measures rollback round-trip equivalence** end-to-end: cold-turn KV state after N tokens via fresh prefill, vs. warm-turn KV state after N tokens via state-replay rollback from a checkpoint at N-M. The two states must match within the same bf16-stability bounds as the per-step equivalence test.

**Reference commits:** `4bc72c8`, `05cc456`, `2274b67`, `8c29e3e`. Cross-relevant but not state-replay-shaping: `463d722` (prefix-cache format-version bump — touches spec 017), runtime-refactor cluster `ce36f62` / `0972afb` / `e2be8a4`.

## Expected impact

For PLD specifically: the entire **Qwen 3.5 / 3.6 family** becomes accessible. Today the auto-routing falls back to TokenIterator at parity; with state-replay rollback the iterator engages the n-gram path. Workload-dependent — but on input-grounded prompts where Gemma 4 26B A4B sees +25%, Qwen 3.5 9B should see comparable or better (it's already a smaller model with faster verify).

For DFlash: spec 015's reported 2.2-4.4× becomes architecturally cleaner — the state-replay primitive is shared, no separate code path.

For PrefixKVCache: hybrid models become cacheable. Multi-turn TTFT on Qwen 3.5 9B drops from ~3s on turn N+1 to ~0.5s (mostly suffix prefill).

## Risks

1. **Numerical equivalence under state replay.** dflash-mlx claims "bf16-sensitive paths are stabilised" — there's a fp32-accumulator step on the rollback kernel to avoid drift. We need the same. Testing against a sequential reference is non-negotiable.

2. **Memory pressure at long context + large verify windows.** Delta log can blow up if the iterator gets ambitious. Cap and fall back as described.

3. **Mamba variants we haven't handled.** Nemotron-H's "cascade" variant and Jamba's partial-Mamba have slightly different state shapes than Qwen 3.5's GatedDeltaNet. Each needs a kernel verification pass. Probably 2-3 kernel variants total across the supported model zoo. **Nemotron Cascade 2 30B-A3B (4-bit + 8-bit MLX-community builds) is cached locally** — phase 2 ships GDN-only kernel coverage but smoke-tests Cascade 2 to validate the second-Mamba-variant path.

4. **Built against the post-spec-006 KV-cache hierarchy.** Spec 006 (KVCache type consolidation, issue #73) merged in PRs #163–#166. `MambaCache` was renamed to `SSMStateCache` and is now a subclass of `ArraysCache`, with `[MLXArray?]` slot storage (see `Libraries/MLXLMCommon/KVCache.swift:1335`). The `StateReplayCache` conformance is a clean addition on top — `SSMStateCache` reports `KVStorageKind.ssm` and stays outside the K/V storage/eviction axes spec 006 reorganised.

5. **Masked-timestep correctness** (NEW from upstream `3217e15`). Both `gated_delta_step_record` and `state_replay` kernels must save `old_state` before each step, gate decay/accumulate/quantize on a uniform `do_step` predicate, and restore via `metal::select` on masked positions. Without this, masked positions silently corrupt state. Mitigation: implement the `do_step` + `metal::select` pattern from day 1; add a `testMambaTapeReplayMaskedTimesteps` test that locks the contract against a masked-token reference run.

6. **Branchless kernel discipline** (NEW from upstream `c9f992e`). Divergent execution within a SIMD group hurts throughput. Mitigation: use `mask_gate = float(mask)` multiply instead of `if(...)` guards; replace `if(use_key) { score = ...; }` with unconditional compute + `metal::select(-inf, score, use_key)`; replace `sum==0 ? 0 : x/sum` with `x * (1/max(sum, 1e-9))`.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/StateReplayCache.swift` | (Phase 1 in PR #143) Phase 2 generalises `recordStep(_ delta: MLXArray)` → `recordStep(_ innovations: [MLXArray])` so per-step state can carry the (k, g, qkv) trio that GDN's recurrent kernel needs. |
| `Libraries/MLXLMCommon/KVCache.swift` (around `SSMStateCache` at line 1335) | `SSMStateCache: StateReplayCache` extension — delta log (`[[MLXArray]]`), pre-record snapshot, `beginRecord` / `recordStep` / `commitFull` / `rollback(acceptedPrefix:)` / `cancel`. |
| `mlx-swift/Source/Cmlx/mlx-generated/metal/gated_delta_replay.metal` (new, ~400–500 lines) | **In sibling `mlx-swift` repo.** Precompiled metal kernels: `gated_delta_step_record` (forward + delta-log capture) and `state_replay` (rollback by re-folding accepted prefix). Ship 4 templated variants each (vec × non-vec, masked × non-masked). Built into `mlx.metallib` via `make metal`. |
| `Libraries/MLXLMCommon/StateReplayKernels.swift` (new, ~80 lines) | Swift wrappers around the precompiled metallib symbols — no JIT. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Predicate flip from `canTrimPromptCache` → `canRollbackPromptCache` at the n-gram + speculative-iterator init guards (currently around lines 1429 + 2065 in alpha; symbol-level edit). |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | Predicate flip at the iterator init (currently line 633) + body wiring (`beginCacheRecord` / `commitCacheRecord` / `rollbackPromptCache` / `cancelCacheRecord`) around `speculateRound`. |
| `Libraries/MLXLMCommon/DFlashSpeculativeDecoding.swift` | Predicate flip — file lands with phase-1 PR #141; touch when that's merged. |
| `Libraries/MLXLMCommon/MirrorSpeculativeDecoding.swift` | Predicate flip — file lands with phase-1 PR #142; touch when that's merged. |
| GDN layer call site (likely `Libraries/MLXLLM/Models/Qwen3Next.swift` / `Qwen35.swift` `Qwen3NextGatedDeltaNet.update(...)` and `Libraries/MLXLLM/Models/NemotronH.swift` for the Cascade 2 variant) | When cache is recording, call `cache.recordStep([k_t, g_t, qkv_t])`. Gate via `if ssmCache.isRecording { ... }` early-exit. |
| `Tests/MLXLMTests/StateReplaySSMStateCacheTests.swift` (new) | Per-step / partial-accept / cancel / per-token-equivalence / bf16-stability / masked-timesteps coverage on Qwen 3.5 GDN + Nemotron Cascade 2 + iterator smoke tests. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | Drop the "Qwen 3.5 omitted due to SSMStateCache" guard. Add Nemotron Cascade 2 ngram-spot row. |

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
