# Spec 002 — Dense MLP inline activation Metal kernel

**Status:** draft
**Target:** `FusedGateUpMLP` (landed in spec 001) — used by `Qwen3NextMLP` (Qwen3.5 dense + GatedDeltaNet hybrids) and `Gemma4SharedMLP` (Gemma 4 E2B / E4B / 31B / 26B A4B shared MLP).
**Owner:** TBD
**Expected gain:** +3–6% decode tok/s on top of spec 001. Decode-only. Prefill unchanged (the kernel is a GEMV fast-path).
**Depends on:** spec 001 (the kernel consumes the fused `gate_up_proj` weight layout).

## Motivation

After spec 001, the dense MLP forward pass is:

```
gateUp = gate_up_proj(x)          # 1 quantized matmul  (Metal)
gate, up = split(gateUp, 2, -1)   # 2 Slice dispatches  (Metal)
activated = silu(gate) * up       # 2 element-wise      (Metal)
y = down_proj(activated)          # 1 quantized matmul  (Metal)
```

That's **6 Metal dispatches per MLP layer** at decode (T=1). On a 32-layer model at ~50 µs/dispatch, the 4 middle dispatches (`Slice ×2 + silu + multiply`) cost ~6.4 ms/token — on top of the two matmuls. A single inline kernel that reads the fused `gateUp` tensor, splits internally, applies the activation, and writes the `hidden = silu(gate) * up` result collapses that to **1 dispatch**, saving 3 per layer.

`FusedGateActivationKernel` already exists in [`Libraries/MLXLMCommon/SwitchLayers.swift`](Libraries/MLXLMCommon/SwitchLayers.swift) for the MoE path (gated off pending validation). This spec lifts the same pattern into a dense variant and wires it into `FusedGateUpMLP`.

Rough math for Qwen3.5-4B: 32 layers × 3 saved dispatches × 50 µs ≈ 4.8 ms/token. Post-001 decode is ~12.5 ms/token (80 tok/s), so expected gain is **~4–6% → 84–86 tok/s**. Smaller models get a bigger relative win (dispatch overhead dominates more).

## Non-goals

- No weight layout changes — this spec consumes `gate_up_proj.weight` as written by spec 001.
- No MoE-path changes. The existing gated-off `FusedGateActivationKernel` in `SwitchLayers.swift` stays gated off — its own enablement belongs in a separate spec after this dense variant validates correctness.
- No RMSNorm fusion (that's spec 004).
- No attention changes.

## Design

### New kernel: `FusedDenseGateActivation`

Add a new Metal kernel next to the existing `FusedGateActivationKernel` in `SwitchLayers.swift` (or a new file `FusedDenseActivation.swift` if the files get too long). The signature: one input (`gateUp: [..., 2 * hiddenDims]`), one output (`hidden: [..., hiddenDims]`). Activation selector is a kernel template parameter (silu vs gelu_approx).

Pseudocode (Metal):

```metal
kernel void fused_dense_gate_silu(
    const device bfloat *gateUp [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &hiddenDims [[buffer(2)]],
    constant uint &totalRows [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= hiddenDims || tid.y >= totalRows) return;
    uint row = tid.y;
    uint col = tid.x;
    bfloat g = gateUp[row * 2 * hiddenDims + col];
    bfloat u = gateUp[row * 2 * hiddenDims + hiddenDims + col];
    // SiLU(g) * u, templated on activation
    float gf = (float)g;
    float silu = gf / (1.0f + exp(-gf));
    out[row * hiddenDims + col] = (bfloat)(silu * (float)u);
}
```

### `FusedGateUpMLP` integration

Extend the class with an optional `inlineActivation` flag (default `false` until validated). When enabled, `callAsFunction` replaces the split + activation + multiply path with a single `MLXFast.metalKernel` dispatch over the `gateUp` tensor:

```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let gateUp = gateUpProj(x)
    let hidden: MLXArray
    if inlineActivation, let kernel = Self.kernel(for: activationKind) {
        hidden = kernel(gateUp, hiddenDims: hiddenDims)
    } else {
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        if let twoArgActivation {
            hidden = twoArgActivation(parts[1], parts[0])
        } else {
            hidden = activation(parts[0]) * parts[1]
        }
    }
    return downProj(hidden)
}
```

`activationKind` enum: `.silu`, `.geluApprox`, `.geluExact`. Covers both Qwen (silu) and Gemma 4 (gelu_approx) callers. Kernels are lazy-loaded singletons keyed on (kind, dtype).

### Fallback triggers

Follow the `rmsNormQuantizedGEMV` guard pattern — the inline kernel is decode-only and must fall back to the MLX-primitive path for any input that's outside its fast case:
- `x.dim(-2) != 1` (prefill → use MLX primitives; batching gets the GEMM path which is already dispatch-efficient)
- dtype not in `{ .bfloat16, .float16 }` (no fp32 kernel specialization; not worth the build cost)
- last-axis size not a multiple of the thread-group width (safe fallback)

### Gemma 4 `compile()` compatibility

Gemma 4's `Gemma4SharedMLP` has a `compile(shapeless: true)` wrapper around the split + geglu + down_proj path. The inline kernel replaces that entire path with a single `MLXFast.metalKernel` invocation. Options:
- **A (preferred):** skip the `compile()` wrapper when `inlineActivation` is on — the kernel itself is the fusion, compile buys nothing.
- **B:** keep the wrapper around the `downProj` call only. Probably not worth the complexity.

## Acceptance criteria

1. All dense Qwen3.5 and Gemma 4 models load and produce coherent summarization output at ctx=1024.
2. Decode tok/s on Qwen3.5-4B (M1 Max, 4bit, no-quant KV, ctx=1024): **≥ 84 tok/s** (baseline-post-001 ≈ 80).
3. Gemma 4 31B decode ≥ 15.0 tok/s at ctx=1024 (baseline-post-001 ≈ 14.7).
4. Gemma 4 E2B decode within ±2% of baseline-post-001 (small model; dispatch savings and kernel-launch overhead can net out near zero).
5. KL divergence vs bf16 baseline unchanged within measurement noise (the fusion is arithmetically equivalent up to fp32 intermediate rounding).
6. `inlineActivation` off by default; enabled per-model after KLD + output-coherence gates pass.

## Measurement plan

```bash
# Matrix matches spec 001; re-run to quantify the stacked gain.
for m in qwen35-0.8b qwen35-2b qwen35-4b qwen35-9b qwen35-27b qwen36-27b; do
    scripts/benchmark.sh --model $m --method summarization \
        --quant 4bit --kv none --context 1024,4096,8192
done

for m in gemma4-e2b gemma4-e4b gemma4-31b gemma4-26b-a4b; do
    scripts/benchmark.sh --model $m --method summarization \
        --quant 4bit --kv none --context 1024,4096,8192
done
```

Report post-001 vs post-002 deltas side-by-side.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| fp32-in-bf16-out activation rounds slightly differently from the MLX primitive chain, causing KLD regressions on long generations | Block ship on `--kld` equivalence test. If the gap is real, emit intermediate fp16 accumulator (matches MLX primitive order) at a small perf cost. |
| Kernel launch overhead exceeds the savings on tiny models (E2B, 0.8B) | Make `inlineActivation` per-model via an env flag (`MLX_INLINE_ACTIVATION=1/0`) before hard-enabling. Per-architecture defaults once measured. |
| Gemma 4 two-arg activation (clipped swiglu, if it ever lands) doesn't map to the single-selector kernel | Kernel gets a `twoArg` template variant. Not needed until a two-arg activation ships — out of scope. |
| Shared Metal kernel cache growth (many activation×dtype combos) | Lazy-instantiate; 2 activations × 2 dtypes = 4 kernels. Tolerable. |

## Out of scope / follow-ups

- **MoE path (`FusedGateActivationKernel` in SwitchLayers.swift)**: currently gated off for the MoE side. Separate spec — needs its own perf/KLD validation on GPT-OSS-20B and Gemma 4 26B A4B.
- **Fusing `down_proj` into the kernel** (full MLP-in-one-kernel): requires a matmul inside the kernel, which is a much larger project. Belongs in an mlx-upstream spec.
- **RMSNorm fusion on the input side** (pre-MLP norm + gate_up_proj): spec 004.

## Open questions

1. Does `MLXFast.metalKernel` support template parameters (activation kind) or do we need one compiled library per variant? Check against the existing `FusedGateActivationKernel` entry point.
2. Per-model default for `inlineActivation` — should we gate on a single env var, or auto-enable for models that benefit (> some threshold decode speedup)?
