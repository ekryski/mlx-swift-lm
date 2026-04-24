# Spec 001 — Dense MLP gate+up fusion

**Status:** draft
**Target:** `Qwen3NextMLP` (used by Qwen3.5 dense / non-MoE text models and GatedDeltaNet hybrids) and `Gemma4SharedMLP` (used by Gemma 4 E2B / E4B / 31B)
**Owner:** TBD
**Expected gain:** +5–10% decode tok/s at 1024 ctx on M1 Max. Smaller at 4096 ctx (KV-read-bound).

## Motivation

Both `Qwen3NextMLP` and `Gemma4SharedMLP` call their `gate_proj` and `up_proj` Linears as two separate matmul dispatches on the same input `x`, then combine the results:

```swift
// Qwen3NextMLP
func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
}
```

```swift
// Gemma4SharedMLP — compile()-cached variant
downProj(compiledGeglu(gateProj(x), upProj(x)))
```

During decode (T=1), each Linear call is a small vector-matrix multiply bottlenecked by **weight-read bandwidth + dispatch overhead**, not compute. Merging the two weights along the output axis into a single `[H, 2I]` matrix and doing one matmul followed by `split` and activation:

1. Saves **one Metal dispatch per MLP layer** (~50 µs on M1 Max applegpu_g13s).
2. Reads the same total weight bytes but in one contiguous stream — slightly better DRAM efficiency.
3. Mirrors the pattern `FusedGateUpSwitchGLU` already uses for MoE, which landed in PR #64 and delivered **+47% decode** on GPT-OSS-20B at ctx=128 no-quant.

For a 32-layer Qwen3.5-4B decode at ctx=1024: saving ~32 dispatches × 50 µs = ~1.6 ms/token. Current decode is ~13 ms/token (77 tok/s), so expected improvement is **~5–10%** — up to ~85 tok/s.

For a 48-layer Gemma 4 31B decode at ctx=1024 (~14 tok/s ≈ 71 ms/token): saving ~48 × 50 µs = 2.4 ms/token, ~3–4% decode, up to ~14.5 tok/s. Smaller relative gain because the model is already bandwidth-bound on weight reads, not dispatch-bound.

## Non-goals

- This spec is strictly about **weight layout + forward pass + sanitize**. It does not adopt `MLXFast.rmsNormQuantizedGEMV` (that's spec 004) or modify attention paths.
- It does not touch the MoE classes (`SwitchGLU`, `FusedGateUpSwitchGLU`) — those are already fused as of PR #64.
- It does not change checkpoint file formats. Fusion happens at `sanitize()` time, purely in-memory.

## Design

### New class: `FusedGateUpMLP`

New file `Libraries/MLXLMCommon/FusedGateUpMLP.swift` (reusable across models):

```swift
/// Dense MLP that fuses `gate_proj` + `up_proj` into a single `gate_up_proj`
/// Linear with output dim `2 * hiddenDims`. Decode-time saving: one Metal
/// dispatch per layer vs separate gate / up matmuls.
///
/// Weight layout: `gate_up_proj.weight` shape `[2*hiddenDims, inputDims]`
/// where rows `[0..hiddenDims)` are gate, rows `[hiddenDims..2*hiddenDims)`
/// are up. `sanitize()` callers must concatenate on axis 0 (output axis).
///
/// Two activation modes mirror `FusedGateUpSwitchGLU`:
/// - Single-argument (default): `activation(gate) * up` — silu / gelu
/// - Two-argument: `twoArgActivation(up, gate)` — for asymmetric activations
///   (clipped swiglu, etc.)
public final class FusedGateUpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let hiddenDims: Int
    let activation: (MLXArray) -> MLXArray
    let twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)?

    public init(
        dimensions: Int,
        hiddenDimensions: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)? = nil,
        bias: Bool = false
    ) {
        self.hiddenDims = hiddenDimensions
        self.activation = activation
        self.twoArgActivation = twoArgActivation
        self._gateUpProj.wrappedValue = Linear(
            dimensions, 2 * hiddenDimensions, bias: bias)
        self._downProj.wrappedValue = Linear(
            hiddenDimensions, dimensions, bias: bias)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated: MLXArray
        if let twoArgActivation {
            activated = twoArgActivation(parts[1], parts[0])
        } else {
            activated = activation(parts[0]) * parts[1]
        }
        return downProj(activated)
    }
}
```

### Weight fusion helper

New helper function in the same file (and possibly reusable for the MoE fuse in `GPTOSS.swift`, whose `fuseGateUpWeights` has the same shape — axis=1 there because of the leading `E` expert dim):

```swift
/// Concatenate `gate_proj.*` and `up_proj.*` tensors into a single
/// `gate_up_proj.*` entry along the output axis (axis=0 for dense Linear;
/// axis=1 for SwitchLinear). Call from a model's `sanitize()` once the
/// rest of the weight-map transformations are done.
///
/// - Parameters:
///   - weights: mutable weight map
///   - keyFilter: substring that identifies gate/up tensor paths
///     (e.g. `".mlp."` for dense, `".experts."` for MoE)
///   - outputAxis: axis to concatenate on (0 for Linear, 1 for SwitchLinear)
public func fuseGateUpWeights(
    _ weights: inout [String: MLXArray],
    keyFilter: String,
    outputAxis: Int
) {
    let gateKeys = weights.keys.filter {
        $0.contains(keyFilter) && $0.contains("gate_proj")
    }
    for gateKey in gateKeys {
        let upKey = gateKey.replacingOccurrences(
            of: "gate_proj", with: "up_proj")
        guard let gateVal = weights[gateKey], let upVal = weights[upKey]
        else { continue }
        let fusedKey = gateKey.replacingOccurrences(
            of: "gate_proj", with: "gate_up_proj")
        weights[fusedKey] = concatenated([gateVal, upVal], axis: outputAxis)
        weights.removeValue(forKey: gateKey)
        weights.removeValue(forKey: upKey)
    }
}
```

Covers weight, scales, and biases uniformly (the substring "gate_proj" matches all of `.gate_proj.weight`, `.gate_proj.scales`, `.gate_proj.biases`, `.gate_proj.bias`). Same pattern the current Gemma 4 26B A4B MoE sanitize uses.

### Model integration points

#### Qwen3.5 (primary target)

Files: `Libraries/MLXLLM/Models/Qwen3Next.swift`, `Libraries/MLXLLM/Models/Qwen35.swift`.

1. Replace `Qwen3NextMLP` contents with a thin wrapper over `FusedGateUpMLP` (single-arg silu). Keep class name for source compatibility:

```swift
final class Qwen3NextMLP: Module, UnaryLayer {
    private let fused: FusedGateUpMLP

    init(dimensions: Int, hiddenDimensions: Int) {
        self.fused = FusedGateUpMLP(
            dimensions: dimensions,
            hiddenDimensions: hiddenDimensions,
            activation: MLXNN.silu)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fused(x) }
}
```

Or — preferable — inline `FusedGateUpMLP` fields into `Qwen3NextMLP` so `@ModuleInfo(key: "gate_up_proj")` lands at the expected weight path without wrapper indirection.

2. Add weight fusion to `Qwen35TextModel.sanitize`:

```swift
public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var weights = ... // existing Qwen35 sanitize
    fuseGateUpWeights(&weights, keyFilter: ".mlp.", outputAxis: 0)
    fuseGateUpWeights(&weights, keyFilter: ".shared_expert.", outputAxis: 0)
    return weights
}
```

Note: `shared_expert` inside `Qwen35SparseMoeBlock` is *also* a `Qwen3NextMLP` — it gets the fusion too. The top-level `switch_mlp` (SwitchGLU) is a separate code path and is **not** touched.

3. Qwen3Next (sibling hybrid) also uses `Qwen3NextMLP` — update its sanitize with the same call.

#### Gemma 4 (secondary target — 31B dense, E2B/E4B)

File: `Libraries/MLXLLM/Models/Gemma4.swift`.

`Gemma4SharedMLP` has three Linears (`gate_proj`, `up_proj`, `down_proj`) and an optional `compile()` wrapper around the forward. Two integration paths:

**Path A (recommended):** Inline fused fields into `Gemma4SharedMLP`, keep the `compile()` wrapper.

```swift
class Gemma4SharedMLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    // ... compile setup unchanged ...

    init(dimensions: Int, hiddenDimensions: Int, isDoubleWide: Bool = false, isMoEContext: Bool = false) {
        let effectiveHidden = isDoubleWide ? hiddenDimensions * 2 : hiddenDimensions
        self._gateUpProj.wrappedValue = Linear(dimensions, 2 * effectiveHidden, bias: false)
        self._downProj.wrappedValue = Linear(effectiveHidden, dimensions, bias: false)
        // ...
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if compileEnabled { return compiledForward(x) }
        let gateUp = gateUpProj(x)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        return downProj(compiledGeglu(parts[0], parts[1]))
    }
}
```

Update `compiledForward` closure similarly.

**Path B:** Switch entirely to `FusedGateUpMLP` with `activation: gelu_approx` (Gemma uses geglu: `gelu(gate) * up`). Loses the compile-cache trace but gets the shared primitive. Measure both before choosing.

Add fusion to `Gemma4TextModel.sanitize`:

```swift
fuseGateUpWeights(&processedWeights, keyFilter: ".mlp.", outputAxis: 0)
```

Note: the existing `gate_proj` → `gate_up_proj` concat for `.experts.` in the current Gemma 4 sanitize (MoE path) is at `outputAxis: 1` (SwitchLinear's `[E, out, in]` layout). The new MLP path is `outputAxis: 0` (Linear's `[out, in]` layout). Same helper handles both via the `outputAxis` parameter.

### Compat / migration

- **Checkpoints already on disk stay unchanged.** Sanitize merges in memory at load time.
- **LoRA adapters** that target `gate_proj` or `up_proj` keys need a translation step (concat their deltas on the same axis). If LoRA is a live requirement for Qwen3.5 users, surface a loader warning with a migration note. **Open question for product:** do any shipped adapters target those keys?
- **Backward source compat** preserved by keeping the `Qwen3NextMLP` class name (only the weight key changes from `gate_proj.weight` / `up_proj.weight` to `gate_up_proj.weight`).
- Models loaded from **already-fused checkpoints** (same naming convention as Gemma 4 26B A4B MoE post-fuse) should short-circuit: `if weights.keys.contains(where: { $0.contains(".mlp.gate_up_proj.weight") }) { return weights }`.

## Acceptance criteria

1. `Qwen3.5-0.8B`, `Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B` all load and produce coherent summarization output at ctx=1024.
2. Decode tok/s on Qwen3.5-4B (M1 Max, 4bit, no-quant KV, ctx=1024): **≥ 82 tok/s** (baseline 77.8).
3. Prefill tok/s unchanged within ±3% at ctx=1024 (this is not a prefill optimization).
4. Gemma 4 31B decode ≥ 14.3 tok/s at ctx=1024 (baseline 13.9).
5. Gemma 4 E2B decode unchanged within ±2% (it's small enough that even dispatch savings are noise against vocab-head cost). If it regresses by >3%, revert Gemma 4 from this spec and ship Qwen3.5 only.
6. KL divergence vs bf16 baseline unchanged within measurement noise (the fusion is arithmetically identical).

## Measurement plan

Run on M1 Max 64GB, commit lock, N=3 per config. Record the **median**, not the worst; note the spread.

```bash
# Qwen3.5 non-MoE sweep
for m in qwen35-0.8b qwen35-2b qwen35-4b qwen35-9b; do
    scripts/benchmark.sh --model $m --method summarization \
        --quant 4bit --kv none --context 128,1024,4096
done

# Gemma 4 dense sweep
for m in gemma4-e2b gemma4-e4b gemma4-31b; do
    scripts/benchmark.sh --model $m --method summarization \
        --quant 4bit --kv none,turbo4v2 --context 128,1024,4096
done
```

Report headline numbers before and after for ctx=1024 decode. Call out anything outside the acceptance ranges.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `MLX.split` on a larger tensor is slower than returning two separate matmul outputs | Profile with Instruments Metal System Trace. If the split dispatch dominates, measure the compile-cached variant (Gemma 4 Path A) — `compile()` should fuse the split back in. |
| Weight-load OOM spike during `concatenated([gate, up], axis: 0)` in sanitize | The concat runs on GPU under lazy eval; peak temporarily doubles the one tensor's footprint. For 9B it's ≤ 400 MB — tolerable. If problematic, concat one layer at a time with explicit `eval()` + `MLX.Memory.clearCache()` between. |
| Existing LoRA adapters break | Loader translation step + regression test with a shipped adapter (if any). Document migration. |
| Gemma 4 31B regresses (compile loses traces because split is now a primitive on the compile boundary) | Path B fallback: drop `compile()` wrapper, use `FusedGateUpMLP` directly. Or keep Path A with compile-cached forward still fusing the split internally. |

## Out of scope / follow-ups

- **Inline activation Metal kernel** (`FusedGateActivationKernel` already exists in `SwitchLayers.swift` but is gated off): dense variant promoted to **spec 002**.
- **QKV fusion via `MLXFast.batchedQKVQuantizedGEMV`** on the attention projections: **spec 003**.
- **`MLXFast.rmsNormQuantizedGEMV`** on the pre-MLP norm + gate/up projection: **spec 004**. Stackable with this spec; an additional ~+5% decode.

## Open questions

1. Are there LoRA adapters in the wild targeting `gate_proj` / `up_proj` keys on Qwen3.5 or Gemma 4 that we need to migrate?
2. Should `FusedGateUpMLP` live in `MLXLMCommon` (reusable) or stay per-model? Leaning common since Qwen + Gemma both want it.
3. On Gemma 4 decision: Path A (inline + compile) or Path B (reuse `FusedGateUpMLP` primitive)? Decide after A/B-measuring both on 31B.
