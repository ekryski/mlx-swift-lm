# 020 — Tape-replay rollback for non-DFlash speculative decoders

**Status:** spec, ready to issue (high-leverage, technically the deepest piece)
**Branch:** new branch off post-013 / post-015 phase 2
**Depends on:** spec 015 (DFlash) phase 3 lands the per-layer Mamba kernel; this spec generalises it.

## The insight

Today's `NGramSpeculativeTokenIterator` refuses to run on hybrid models (Qwen 3.5 / 3.6 GatedDeltaNet / Nemotron-H / Jamba / Granite-MoE-Hybrid / FalconH1) because their Mamba / SSM layers have non-trimmable cache state — there's no positional rollback. dflash-mlx solves this for DFlash with **innovation-tape rollback**: during the verify forward, record the per-step recurrent updates; on partial acceptance, replay only the accepted prefix's updates onto a snapshot taken at round entry.

**This trick is decoder-agnostic.** Once we have a tape-replay rollback primitive in the cache layer, *any* speculative decoder using that cache can do partial-accept rollback, including the n-gram iterator. The mamba constraint that has been blocking PLD on the entire Qwen 3.5 / 3.6 family becomes a non-issue.

This is the highest-leverage technical change in the speculative-decoding stack for this codebase.

## What "tape-replay" means

A linear recurrent layer evolves its hidden state via `s_{t+1} = f(s_t, x_t, params)`. A typical SSM state update looks like:

```
s_{t+1} = A_t * s_t + B_t * x_t      (linear; can be batched / parallel-scanned)
y_t     = C_t * s_t + D_t * x_t
```

Standard cache: store `s_t` after every token. Standard rollback: snapshot `s_{t_0}` at speculation start; on partial accept of `k` tokens, snapshot becomes invalid, restore from `s_{t_0}` and redo `k` updates.

dflash-mlx's tape-replay: don't snapshot the **full state** at every speculation; record the **deltas** (`B_t * x_t`, the "innovation") per step during verify, plus the **start state** `s_{t_0}` once. On accept of `k` tokens, the post-rollback state is `s_{t_0 + k} = A^{(k)} * s_{t_0} + ∑_{i=0..k-1} A^{(k-1-i)} * (B_{t_0+i} * x_{t_0+i})`. The deltas were already computed during verify; we just re-fold them through the linear recurrence.

For non-linear recurrences (Mamba's gated delta is partially nonlinear), the same idea applies if the nonlinearity factorises through the per-step inputs — which is the case for Mamba-style layers per the dflash-mlx implementation.

## Design

### 1. New cache protocol

```swift
public protocol TapeReplayCache: KVCache {
    /// Begin recording an innovation tape. Caller is committing to one of:
    ///   - `commitFull()` — accept all tape entries; cache state advances to end-of-tape.
    ///   - `rollback(acceptedPrefix:)` — accept first `k` entries; tape entries `k..<count` discarded.
    ///   - `cancel()` — reject all entries; cache state restored to pre-record snapshot.
    func beginRecord()

    /// Append the next innovation. Called from inside `update(...)` during a recording session.
    func appendInnovation(_ delta: MLXArray)

    func commitFull()
    func rollback(acceptedPrefix k: Int)
    func cancel()

    /// Cost: replay-time complexity. Used by the iterator to decide whether
    /// tape-replay is cheaper than re-running the layer from snapshot.
    var replayCost: ReplayCost { get }  // .o1 (linear SSM), .ok (Mamba gated delta), .reforward (worst case)
}
```

`update(...)` becomes mode-aware: outside a recording session it behaves as today; inside one, it also calls `appendInnovation(...)` with the delta from the current input.

### 2. MambaCache conformance

This is the load-bearing change. Today `MambaCache.isTrimmable` is `false`. After this spec lands:

- `MambaCache: TapeReplayCache` — supports beginRecord/append/rollback.
- `isTrimmable` stays false (semantic: positional trim is not supported), but `canTapeReplay` is true.

The Metal kernel for `rollback(acceptedPrefix:)` lives in `Sources/Cmlx/mlx-generated/metal/mamba_replay.metal` and matches dflash-mlx's `tape-replay rollback` Metal kernel byte-for-byte where possible.

### 3. New cache trimmability predicate

```swift
public func canRollbackPromptCache(_ cache: [KVCache]) -> Bool {
    cache.allSatisfy { $0.isTrimmable || ($0 as? TapeReplayCache) != nil }
}
```

This replaces `canTrimPromptCache` in the `MLXLMCommon.generate(...)` auto-routing decision. Pure-attention models still take the trim path (cheap, no tape recording). Hybrid models take the tape-replay path.

### 4. Iterator changes

`NGramSpeculativeTokenIterator.speculateRound` becomes:

```swift
beginCacheRecord(mainCache)        // tape mode: only fires on tape-replay caches
let mainResult = mainModel(verifyInput, cache: mainCache, state: mainState)
// ... compute accepted ...
if accepted == numDraft {
    commitCacheRecord(mainCache)   // full accept; tape discarded, full delta committed
} else {
    rollbackCacheRecord(mainCache, acceptedPrefix: 1 + accepted)  // y + accepted drafts kept
}
```

For pure-attention layers, `beginCacheRecord/commitCacheRecord` are no-ops; `rollbackCacheRecord` translates to the existing `trim(numTokens: rejected)`.

For Mamba layers, the calls go through the tape-replay Metal kernel; full-accept skips the kernel entirely.

### 5. Memory budget

The tape stores `(numDraft + 1)` deltas per Mamba layer per round. For Qwen 3.5 9B with ~30 GDN layers and a 16-token verify window, that's ~30 × 16 × `state_dim` MLXArrays = a few MB per round. Cleared on commit/rollback.

For verify windows above ~32 tokens this becomes large; cap the tape size at `MLX_TAPE_MAX_DELTAS=32` and force a fall-back to AR if a round wants to draft more than that.

## Implementation phases

1. **Phase 1 — Protocol + KVCacheSimple no-op conformance.** Land the protocol; concrete trimmable caches conform with `appendInnovation = noop`, `rollback = trim`, `commitFull = noop`. Iterator unchanged. Validates the API on a workload that already works.

2. **Phase 2 — MambaCache + Metal kernel.** This is the meat. Port dflash-mlx's recurrent-rollback Metal kernel. Test against per-token equivalence with a sequential (no rollback) reference run.

3. **Phase 3 — Wire into NGramSpeculativeTokenIterator.** Replace the `canTrimPromptCache` gate with `canRollbackPromptCache`. Run the full benchmark suite on Qwen 3.5 / 3.6 — this is where PLD's coverage doubles overnight.

4. **Phase 4 — Wire into DFlashSpeculativeTokenIterator.** Spec 015's phase 3 already uses tape-replay; refactor it to use the protocol from this spec rather than its own private path. One implementation, two consumers.

5. **Phase 5 — Wire into PrefixKVCache.** With tape replay, `MambaCache` becomes serialisable for prefix snapshots: snapshot the state, plus enough history to rebuild it. Spec 017 phase 3 depends on this.

## Expected impact

For PLD specifically: the entire **Qwen 3.5 / 3.6 family** becomes accessible. Today the auto-routing falls back to TokenIterator at parity; with tape-replay it engages the n-gram path. Workload-dependent — but on input-grounded prompts where Gemma 4 26B A4B sees +25%, Qwen 3.5 9B should see comparable or better (it's already a smaller model with faster verify).

For DFlash: spec 015's reported 2.2-4.4× becomes architecturally cleaner — the tape-replay primitive is shared, no separate code path.

For PrefixKVCache: hybrid models become cacheable. Multi-turn TTFT on Qwen 3.5 9B drops from ~3s on turn N+1 to ~0.5s (mostly suffix prefill).

## Risks

1. **Numerical equivalence under tape replay.** dflash-mlx claims "bf16-sensitive paths are stabilised" — there's a fp32-accumulator step on the rollback kernel to avoid drift. We need the same. Testing against a sequential reference is non-negotiable.

2. **Memory pressure at long context + large verify windows.** Tape can blow up if the iterator gets ambitious. Cap and fall back as described.

3. **Mamba variants we haven't handled.** Nemotron-H's "cascade" variant and Jamba's partial-Mamba have slightly different state shapes than Qwen 3.5's GatedDeltaNet. Each needs a kernel verification pass. Probably 2-3 kernel variants total across the supported model zoo.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/KVCache.swift` | New `TapeReplayCache` protocol + default no-op conformances. |
| `Libraries/MLXLMCommon/MambaCache.swift` | Tape-replay conformance + delta-tape buffer. |
| `Sources/Cmlx/mlx-generated/metal/mamba_replay.metal` (new) | Replay kernel; ~150-300 lines. |
| `Libraries/MLXLMCommon/Evaluate.swift` | New `canRollbackPromptCache` predicate; auto-routing uses it. |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | `beginRecord` / `commit` / `rollback` calls in `speculateRound`. |
| `Tests/MLXLMTests/TapeReplayTests.swift` (new) | Per-token equivalence tests against sequential reference for each Mamba variant. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | Drop the "Qwen 3.5 omitted due to MambaCache" guard. |

## Out of scope

- Tree-attention rollback. Tree shape is independent of recurrence type. When spec 014 lands tree attention on pure-attention models, this spec extends the same kernel to handle multi-path tape replay (rollback to a non-linear position in a draft tree). Different spec.
- Variable per-layer recording. Today the iterator either records all layers or none. A finer-grained "only Mamba layers record" optimisation is possible but not necessary at our scale.

## References

- [dflash-mlx engine-v2 `recurrent_rollback_cache.py`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/recurrent_rollback_cache.py) — Python reference, the tape-replay design we're porting.
- [dflash-mlx engine-v2 `engine/rollback.py`](https://github.com/bstnxbt/dflash-mlx/blob/engine-v2/dflash_mlx/engine/rollback.py) — full-accept vs. partial-accept paths.
- [Mamba: Linear-time sequence modeling with selective state spaces (Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752) — the recurrence we're rolling back.
- [GatedDeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2412.06464) — Qwen 3.5's specific variant.
