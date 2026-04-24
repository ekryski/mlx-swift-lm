# Spec 008 — Prefill eval every N layers (small-dense Qwen3.5 / 3.6 hybrids)

**Status:** landing as gated optimization
**Target:** `Qwen35TextModelInner.callAsFunction` ([Libraries/MLXLLM/Models/Qwen35.swift](../Libraries/MLXLLM/Models/Qwen35.swift)) — small dense Qwen3.5 / 3.6 hybrids only (gated by `numExperts == 0 && hiddenSize <= 4096`).
**Measured gain:** Modest prefill (+2–8%) and scaling decode gains (+3–16%, strongest at long ctx) with 15–25% peak-memory reduction on Qwen3.5-0.8B / 2B / 4B / 9B. No change on larger / MoE variants (gated out).
**Benchmarks:** [`benchmarks/notes/prefill-eval-every-n-layers-2026-04-24.md`](../benchmarks/notes/prefill-eval-every-n-layers-2026-04-24.md)

## Motivation

Alpha runs an `asyncEval` **per layer** during prefill on hybrid models
(Qwen3.5 / 3.6 / NemotronH). `asyncEval` is strictly better than
synchronous `eval` — the CPU keeps building layer i+1's graph while
the GPU executes layer i — but **it still splits the lazy graph at
every layer boundary.** Each split:

1. Forces MLX to commit whatever command buffer is open.
2. Adds a scheduling round-trip.
3. Shortens the window MLX has to fuse adjacent kernels across layers.

Batching `asyncEval` every N layers reduces the sync count and lets
MLX fuse across more layers per command buffer.

## Non-goals

- **Not touching decode.** `if isPrefill` (T > 1) gates the code path. At T=1 the lazy graph is tiny and any eval adds latency.
- **Not changing `GatedDelta.swift`'s intra-layer `GDN_EVAL_INTERVAL=128`.** Orthogonal.
- **Not changing Gemma4 / GPT-OSS prefill.** Separate code paths that weren't shown to have the same problem.
- **Not changing `NemotronH`.** A/B showed a consistent +3–5% peak-memory regression across all contexts with no throughput gain, so NemotronH stays on per-layer `asyncEval`.

## Design

### `PrefillEvalInterval` helper

New [`Libraries/MLXLMCommon/PrefillEvalInterval.swift`](../Libraries/MLXLMCommon/PrefillEvalInterval.swift):

```swift
public enum PrefillEvalInterval {
    public static let value: Int = {
        let raw = ProcessInfo.processInfo.environment["MLX_PREFILL_EVAL_INTERVAL"] ?? ""
        return max(1, Int(raw) ?? 8)
    }()
}
```

Default N = 8 picked from the A/B matrix on M1 Max. `N = 1`
force-disables the optimization globally even on eligible models
(rollback lever). Non-positive / unparseable clamps to 1.

### Gating: `batchedPrefillEvalEligible`

Stored on `Qwen35TextModelInner` at init time:

```swift
self.batchedPrefillEvalEligible = (args.numExperts == 0) && (args.hiddenSize <= 4096)
```

| Model | `numExperts` | `hiddenSize` | Eligible? |
|---|---:|---:|:---:|
| Qwen3.5 0.8B    | 0   | 1024 | ✓ |
| Qwen3.5 2B      | 0   | 2048 | ✓ |
| Qwen3.5 4B      | 0   | 2560 | ✓ |
| Qwen3.5 9B      | 0   | 4096 | ✓ |
| Qwen3.5 27B     | 0   | 5120 | ✗ |
| Qwen3.6 27B     | 0   | 5120 | ✗ |
| Qwen3.5 35B A3B | 256 | 2048 | ✗ (MoE) |

### Call-site logic

```swift
let batchEval = batchedPrefillEvalEligible && PrefillEvalInterval.value > 1
let evalEvery = PrefillEvalInterval.value
var pendingEval: [MLXArray] = []
for (i, layer) in layers.enumerated() {
    // ... layer forward pass, dtype cast ...
    if isPrefill, let c = cacheArray?[i] {
        if batchEval {
            pendingEval.append(contentsOf: c.innerState())
            let atBoundary = (i + 1) % evalEvery == 0
            let atEnd = (i + 1) == layers.count
            if atBoundary || atEnd {
                pendingEval.append(hiddenStates)
                asyncEval(pendingEval)
                pendingEval.removeAll(keepingCapacity: true)
            }
        } else {
            var toEval: [MLXArray] = [hiddenStates]
            toEval.append(contentsOf: c.innerState())
            asyncEval(toEval)
        }
    }
}
```

**Why intermediate `hiddenStates` are not accumulated:** retaining
references to every intermediate hidden state across the window keeps
dead activations alive and regressed prefill 10% on the first
iteration of this spec. Only cache inner states accumulate; the
*current* hidden state is appended only at flush time.

## Measured results

See the benchmark note for the full 10-model × 6-context sweep. Headline:

- **Qwen3.5 0.8B**: +4% prefill, +4% decode at ctx=1024, scaling to +7.8% prefill and +15.5% decode at ctx=32768. Peak memory −18–27%.
- **Qwen3.5 2B**: +4–11% prefill, +10–13% decode at ctx ≤ 4096. Peak memory −3–13%.
- **Qwen3.5 4B / 9B**: noise-level prefill, +2–5% decode at some contexts. Mild peak reduction on 4B.
- **Qwen3.5 27B / 3.6 27B / 35B A3B / NemotronH**: gated out or reverted — run at alpha speed.
- **GPT-OSS 20B, Gemma 4 E4B** (control): ±1% across the board. Confirms no leakage.

## Env flag

- `MLX_PREFILL_EVAL_INTERVAL=N`
  - Default (unset): **N=8** on eligible models, per-layer on ineligible.
  - `N=1`: force per-layer `asyncEval` globally (rollback).
  - `N` between 2 and 16: exposed for re-tuning on other hardware / future variants.

## Acceptance

All met against alpha on M1 Max:

- [x] Correctness: single-token and multi-token generation unchanged. Test suite passes.
- [x] N=1 parity: prefill matches alpha within noise on every model tested.
- [x] Gate verification: N=8 on Qwen3.5-27B (ineligible) matches N=1 within noise.
- [x] Controls: GPT-OSS-20B and Gemma 4 E4B show ±1% at N=8 vs N=1 — the env flag doesn't leak.
- [x] No regression on any non-eligible model.

## Risks / follow-ups

- **New Qwen3.5 / 3.6 variants** landing with hidden_size ≤ 4096 will auto-opt-in. If a future variant regresses, the gate needs a more specific check or `MLX_PREFILL_EVAL_INTERVAL=1` rollback.
- **NemotronH peak regression** was not explained. Worth a Phase-3 trace to understand why that model's graph structure responds differently — potentially informative for other MoE-heavy models.
- **Decode gains at long ctx on 0.8B** (+15.5% at 32k) are larger than the prefill gain. That wasn't predicted by the spec's motivation. Possible explanation: the batched path changes how the first forward pass leaves the graph in a state that decode's per-token forwards can reuse. Not investigated further; filed as a curiosity.
