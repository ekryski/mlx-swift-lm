# 040 — Mamba / Mamba 2 state-replay rollback

**Status:** Option 1 kernel pair + conv-state capture shipped 2026-05-13. Coverage spans **NemotronH**, **GraniteMoeHybrid**, and **FalconH1** (all three use the shared `ssmUpdate` Mamba 2 selective-state formulation). Jamba is **not yet integrated** — its `ssmStep` uses a 2D `A_log` shape that doesn't match the kernel's `[H]` ALog signature; kernel-side extension (or Swift-side reformulation) is the follow-up.

The 2026-05-13 conv-state fix is the load-bearing addition: `SSMStateCache.recordMambaConvPadded(_:)` captures the per-record padded conv input from each Mamba mixer's `applyConv`, and `rollback(acceptedPrefix: k)` slices `padded[k : k + K - 1]` back into `cache[0]`. Without this, partial-accept rollback restored the conv state to its pre-record value while the recurrent state advanced to step k — produced mild output drift on Nemotron-30B-A3B + n-gram speculative (`"Hello! What is your and ( ( this this..."`-style repetition). After the fix the same prompt produces fully coherent output (`"We need to respond with name and capabilities, concise..."`) at 21.4% draft-acceptance rate.

Round-trip unit tests (`testMambaStateReplayRoundTrip`, `testSSMStateCacheMambaRollback`) cover the kernel pair; the Nemotron + `--ngram 3` smoke covers the conv-state slice end-to-end.

The 2026-05-13 session surfaced one design tension worth recording for the next attempt:

1. **Option 2 design tension on `SSMStateCache` layout.** The class currently owns the GDN `deltaLog` (a `[[MLXArray]]?` of `(δ_t, k_t, g_t)` triples per layer). The Mamba flavour would either (a) parallel-store an `[MLXArray]?` of per-step recurrent-state snapshots — cheap to record, O(1) replay, ~25 KB per step per layer (so ~320 MB transient across 50 SSM layers on Nemotron-30B-A3B with a 256-token verify window, freed at `commitFull` / `rollback`); or (b) per-step `(dA_t, dB_t·x_t)` tuples matching the GDN shape — half the storage, O(k) replay. Both interact with the conv-state cache in `applyConv` (`cache[0]`), which would also need per-step capture to support mid-window rollback — that's the part that pushes Option 2 past the trivial-fallback estimate.

**Branch:** TBD (`ek/040-mamba-state-replay-phase1` once implementation begins)
**Depends on:** [spec 020](020-tape-replay-rollback-generalised.md) phases 1–3 (shipped 2026-05-11 via [PR #143](https://github.com/ekryski/mlx-swift-lm/pull/143) + cross-repo chain). Reuses the `StateReplayCache` protocol, the `SSMStateCache` storage class, the `beginRecord` / `commitFull` / `rollback` lifecycle, the n-gram iterator's record-then-rollback dispatch, and the 4-repo kernel-chain pattern from spec 020 verbatim — this spec is the **Mamba-recurrence kernel pair**, nothing else.

## Problem

Spec 020 phases 1–3 shipped state-replay rollback for the **GatedDeltaNet** recurrence — Qwen 3.5 / 3.6 and any future GDN model. **Mamba** (Nemotron Cascade 2) and **Mamba 2** (Jamba, Granite-MoE-Hybrid, FalconH1) share the `SSMStateCache` storage class but use a *different* recurrence:

```
GDN:    s_{t+1} = g_t · s_t + k_t · δ_t                        (4D state, GQA-expanded k)
Mamba:  s_{t+1} = exp(dt_t · A) · s_t + dt_t · B_t · x_t       (3D state, selective)
```

The math is structurally analogous (decay × state + innovation), but the production `S > 1` forward path differs in a way that blocks dropping in the GDN kernels:

- **GDN's `gated_delta_step` (in `gated_delta.metal`) computes the recurrence sequentially** in a per-step Metal loop. Adding per-step `delta_t` capture is local — a single buffer write inside the loop. That's what `gated_delta_step_record` (spec 020) does.
- **Mamba's `S > 1` forward uses a chunked parallel scan** (`ssmAttn` in [`Libraries/MLXLLM/Models/SSM.swift:145`](../Libraries/MLXLLM/Models/SSM.swift)) that computes the entire T-token block via a surrogate-attention matmul. Per-step `(dA_t, dB_t · x_t)` values are *not* materialised — they're rolled into the parallel-scan reduction.

Until this spec lands, all Mamba-family models opt out of state replay (`SSMStateCache.canStateReplay = false`), which means **no** of the following work on Mamba families:

- N-gram speculative decoding (`canRollbackPromptCache` returns false → iterator declines)
- Prefix KV cache (spec 017 — `serialiseSSM` throws `snapshotInvariantViolation` for non-replay caches)
- DFlash speculative decoding (spec 015 phase 3, future)
- Compressed-domain prefix cache (spec 039, future)

All four downstream specs flip on for Mamba families the day this kernel pair lands — no per-spec adapter work required.

## Design

Three implementation options, ranked by quality:

### Option 1 — Native `ssm_step_record` + `ssm_replay` kernel pair (preferred)

New Metal kernels + C++ Primitive + `mlx-c` bridge + `mlx-swift` Swift wrapper, parallel to spec 020's `gated_delta_step_record` / `state_replay`:

- `ssm_step_record(x, dt, A, B, C, D, state_in, mask) → (y, state_out, dA_log, dBx_log)`
  - Sequential per-step Metal loop (like `gated_delta_step` but for the Mamba recurrence).
  - Emits per-step `dA = exp(dt_t · A)` and `dBx = dt_t · B_t · x_t` into the delta log buffers, alongside the regular `y_t` and `state_out` outputs.
  - Per-step `(A, B, C, D)` selective-state-space coefficients honour the masked-timestep correctness fix (cf. spec 020's adoption of [dflash-mlx `3217e15`](https://github.com/bstnxbt/dflash-mlx/commit/3217e15)) — `dA = 1, dBx = 0` at masked positions so rollback past a masked timestep is identity-preserving.
  - Branchless mask handling per dflash-mlx `c9f992e`.

- `ssm_replay(dA_log, dBx_log, state_snapshot, k, mask) → state_after_k`
  - Re-folds the first `k` entries of the delta log onto the start-of-record snapshot: `s ← dA_i · s + dBx_i` for `i ∈ [0, k)`, masked timesteps skipped.
  - Single Metal dispatch per layer per rollback. O(k · state_dim) work — same scaling as spec 020's `state_replay`.

- Function-constant variants for fp16 / bf16 / fp32 × Mamba state-dim shapes (the Mamba families ship at different `d_state` / `d_conv` values). Same cell pattern as `gated_delta_replay.metal`.

Scope: ~1500 LOC across the 4-repo PR chain. Ships independently of GDN — no overlap with spec 020's kernels.

**This is the preferred option** because it preserves the parallel-scan throughput of the unmasked forward path AND adds zero overhead when not recording (the record buffers are only written when `cache.isRecording`).

### Option 2 — Sequential `ssm_step` fallback during recording (rejected)

When `cache.isRecording`, swap the layer's forward path from `ssmAttn` (parallel scan, fast for T>1) to the existing `ssmUpdateKernel` (sequential T-loop) and capture per-step values from each iteration.

**Rejected because:** verify-forward throughput drops ~5–10× compared to the parallel scan. Verify windows of 8–16 tokens (typical for n-gram speculative) would eat most of the speculative-decode margin. Probably acceptable for `D ≤ 4` but not for the longer windows where n-gram on Qwen 3.5-35B-A3B sees 1.6× speedup. Net effect: spec 040 would land but n-gram-on-Mamba would underperform n-gram-on-GDN by a noticeable margin.

Kept here as the cheap fallback (~200 LOC, no kernel work) if Option 1's kernel chain hits an unforeseen Metal blocker.

### Option 3 — Augmented `ssmAttn` that also emits per-step deltas (rejected)

Compute the per-step `(dA, dB·x)` *in addition to* the chunked scan output. Doubles the layer's compute but keeps the parallel-scan throughput.

**Rejected because:** kernel surgery is invasive — the chunked scan's surrogate-attention matmul doesn't naturally factor through per-step values, so this means rewriting the scan to also keep the per-step intermediates. Risk surface bigger than Option 1's clean parallel kernel pair, with the same end behaviour.

## Cross-repo PR chain (Option 1)

Identical shape to spec 020's chain. Estimated ~1500 LOC total:

| Repo | What | Notes |
|---|---|---|
| `mlx` | `mlx/mlx/backend/metal/kernels/ssm_replay.metal` — `ssm_step_record` (forward + delta-log capture) + `ssm_replay` (rollback). Masked-timestep correctness + branchless mask. Function-constant variants for fp16 / bf16 / fp32 × Mamba state shapes. | Parallel to spec 020's `gated_delta_replay.metal` |
| `mlx` | `mlx/mlx/backend/metal/ssm_replay.cpp` + headers — C++ `Primitive` classes (`SSMStepRecord`, `SSMReplay`). | Parallel to `GatedDeltaStepRecord`, `StateReplay`. |
| `mlx-c` | C ABI bridges (`mlx_fast_ssm_step_record`, `mlx_fast_ssm_replay`). | Parallel to `mlx_fast_gated_delta_step_record`. |
| `mlx-swift` | `MLXFast.ssmStepRecord(...)` / `MLXFast.ssmReplay(...)` Swift wrappers + canonical `.metal` mirror at `Source/Cmlx/mlx-generated/metal/ssm_replay.metal`. | Parallel to `MLXFast.gatedDeltaStepRecord`. |
| `mlx-swift-lm` | `Libraries/MLXLMCommon/StateReplayKernels.swift` — extend `stateReplayUpdate(...)` dispatcher to recognise Mamba-style cache state shape (3D selective state vs GDN's 4D state) and dispatch to the new kernel pair. | Reuses existing dispatcher; pure addition. |
| `mlx-swift-lm` | `Libraries/MLXLLM/Models/NemotronH.swift` + `Libraries/MLXLLM/Models/Jamba.swift` — drop the `cache.canStateReplay = false` setters; default `true` from `SSMStateCache` then takes effect. | Mechanical — just delete the opt-out lines. |
| `mlx-swift-lm` | `Tests/MLXLMTests/StateReplayCacheTests.swift` — add Mamba + Mamba 2 round-trip test parallel to the existing GDN test. | Reuses test scaffolding. |

## Until then

`SSMStateCache.canStateReplay` is **per-cache configurable** (stored property on the class, default `true`). Mamba-using model factories opt out by setting `canStateReplay = false` on the SSM caches they emit:

- [`Libraries/MLXLLM/Models/NemotronH.swift`](../Libraries/MLXLLM/Models/NemotronH.swift) `::newCache(...)` — sets `false` on the `.mamba` layer caches.
- [`Libraries/MLXLLM/Models/Jamba.swift`](../Libraries/MLXLLM/Models/Jamba.swift) `::newCache(...)` — sets `false` on the non-attention layer caches.

When `canStateReplay == false` on any layer, `canRollbackPromptCache` returns `false` for the stack, and `MLXLMCommon.generate(...)` auto-routing gracefully declines the n-gram speculative path → falls back to vanilla `TokenIterator`. No crash, no surprise; users get baseline AR decode on Mamba models with a `TokenIterator` traceback that's identical to the pre-spec-020 behaviour.

When the Mamba kernels land, the model factories flip the flag back to `true` (or just drop the setter — `true` is the default).

## Expected impact

Once shipped, Mamba families inherit the full Tier-1 speculative-decode + prefix-cache toolkit at once:

| Model | What lights up | Expected lift |
|---|---|---|
| Nemotron Cascade 2-30B-A3B 4bit | n-gram speculative (D=12 adapt+strict) | **1.4–1.7× baseline** (similar accept-rate profile to Qwen3.5-35B-A3B which sees 1.64× at the same D / accept-rate operating point in spec 020 phase 2's measurements) |
| Nemotron Cascade 2-30B-A3B 4bit | prefix KV cache (spec 017) multi-turn TTFT | ~2–4× warm TTFT (parallel to Qwen3.5-35B-A3B's measurements) |
| Jamba / Granite-MoE-Hybrid / FalconH1 | n-gram + prefix cache | Per-family bench needed; expected in the same envelope |
| All of the above | DFlash phase 3 (spec 015), spec 039 compressed prefix cache | Trivial — drops in with no per-family adapter work |

## Test plan

- **Cross-repo PR chain** parallel to spec 020's: kernel correctness test in `mlx`, C ABI smoke test in `mlx-c`, Swift wrapper unit test in `mlx-swift`, integration test in `mlx-swift-lm`.
- **`mlx-swift-lm` integration:**
  - `Tests/MLXLMTests/StateReplayCacheTests.swift` — extend with `testMambaStateReplayRoundTrip` and `testMamba2StateReplayRoundTrip`. Same record-then-rollback pattern as the existing GDN test; assert state matches the unmasked-forward reference within tolerance.
  - `Tests/MLXLMTests/SpeculativeDecodingTests.swift` — add n-gram-on-Nemotron-Cascade-2 smoke test.
- **Bench validation:** rerun the spec 020 phase 2 measurement matrix on `nemotron-cascade-2-30b-a3b` (4bit, `--method ngram-spot`, D∈{4,8,12}). Decision gate: ≥1.3× baseline at D=12 adapt+strict matches the GDN baseline envelope.

## Known limitations / open follow-ups

1. **Jamba integration** — its `ssmStep` uses a 2D `A_log` shape that doesn't match the kernel's `[H]` ALog signature; needs either kernel-side shape generalisation or Swift-side reformulation in `Jamba.swift`. Tracked as a separate follow-up; not blocking near-term bench targets.

The **conv-state-on-partial-accept** drift originally observed on #213's Nemotron-30B-A3B `--ngram 3` bench was closed by the conv-state slice in `SSMStateCache.rollback` (this PR's [`fad50f1`](Libraries/MLXLMCommon/KVCache.swift), lines 1654-1677). Both the recurrent state (via `MLXFast.ssmReplay`) and the depthwise-conv buffer `cache[0]` (via `padded[k : k + K - 1]` slice from the per-step `mambaConvPadded` capture) now replay in lock-step on partial accept.

## Out of scope

- **GDN kernels** — shipped in spec 020 phases 1–3.
- **Mamba's S=1 forward** — already uses `ssmUpdateKernel` (sequential, suitable for recording) in decode. Spec 040's `ssm_step_record` is the recording variant of that kernel.
- **GroupNorm / `(C, D)` projections** — Mamba's selective-state-space `C` and skip-connection `D` are linear in state at output; rollback only touches `state` evolution, the output projections are computed fresh on the post-rollback state.

## References

- [Spec 020](020-tape-replay-rollback-generalised.md) — state-replay rollback for GDN (shipped). This spec is its Mamba-recurrence parallel.
- [Mamba: Linear-time sequence modeling with selective state spaces (Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752) — the recurrence we're rolling back.
- [Mamba 2: State Space Duality (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) — Jamba / Granite-MoE-Hybrid / FalconH1.
- **dflash-mlx kernel reference (precedent):** [`bstnxbt/dflash-mlx@main`](https://github.com/bstnxbt/dflash-mlx) HEAD `8d8545d` — same correctness-fix patterns (`3217e15` masked timestep, `c9f992e` branchless) apply to the Mamba kernel variant.
- [`Libraries/MLXLLM/Models/SSM.swift`](../Libraries/MLXLLM/Models/SSM.swift) — `ssmAttn` (parallel scan, S>1 forward) and `ssmUpdateKernel` (sequential, S=1 forward). The recording forward path extends from `ssmUpdateKernel`.
- [Spec 015](015-dflash-mlx-port.md) phase 3 — DFlash on hybrid models; refactors onto `StateReplayCache` once Mamba kernels land.
- [Spec 017](017-prefix-kv-cache.md) — prefix KV cache; `serialiseSSM` opens up to Mamba families when this spec ships.
- [Spec 039](039-compressed-prefix-kv-cache.md) — compressed-domain prefix cache; inherits Mamba coverage from spec 017.
