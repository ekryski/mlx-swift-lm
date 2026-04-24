# Spec 004 — `rmsNormQuantizedGEMV` adoption for dense MLP

**Status:** draft
**Target:** Qwen3.5 dense + Gemma 4 31B (and stackable on top of spec 001's fused gate+up class)
**Owner:** TBD
**Expected gain:** Additional +3–8% decode tok/s on top of spec 001. Decode-only; prefill unchanged.
**Depends on:** spec 001 (recommended — much higher marginal gain once gate/up are already one projection)

## Motivation

`MLXFast.rmsNormQuantizedGEMV` already exists in mlx-swift and fuses **RMSNorm + 4-bit GEMV** into a single Metal kernel. Gemma 4 attention uses it for the Q/K/V projections via a private `fusedNormProj` helper ([Gemma4.swift:541](Libraries/MLXLLM/Models/Gemma4.swift:541)) — one dispatch instead of two (norm + matmul).

The same fusion is available for every pre-norm → projection site in a decoder. It is **decode-only** (T=1) because the underlying kernel is a GEMV, not a GEMM — the guards in Gemma 4's helper are `x.dim(-1) == x.size` (i.e. a flat vector) and `x.dtype == .float16 || x.dtype == .bfloat16`.

Per decode token on Qwen3.5-4B (32 layers, pre-norm architecture):
- 1 norm dispatch before attention → fuse with Q (or with the fused QKV if batched)
- 1 norm dispatch before MLP → fuse with `gate_up_proj` (post-spec-001)
- = **2 saved dispatches per layer × 32 = 64 fewer dispatches per token**

At ~50 µs/dispatch on M1 Max `g13s`, that's ~3.2 ms/token. On Qwen3.5-4B current decode (~11.8 ms/token after spec 001 lands at an estimated 85 tok/s), that's another +3–8% → target ~88–92 tok/s.

Gemma 4 31B similarly: 48 layers × 2 saved = 96 dispatches/token, ~4.8 ms. On a ~70 ms/token baseline, ~6–7% → target ~15 tok/s.

## Non-goals

- This spec does **not** pursue rmsNormQuantizedGEMV on the **down_proj** post-activation — there's no pre-norm in front of it.
- It does not pursue it on the MoE `gatherQuantizedMM` path — that needs a different kernel (`gather_rms_norm_qgemv`, not shipped) and is a separate investigation.
- It does not touch bf16 / fp32 code paths — MLX only ships the fp16/bf16 qgemv primitive.
- It does not address **non-quantized** deployments — falls back to two-dispatch (norm + matmul) for those.

## Design

### Shared helper in `MLXLMCommon`

Promote Gemma 4's private `fusedNormProj` to a public helper in a new file `Libraries/MLXLMCommon/FusedNormProjection.swift`. Gemma 4 switches to the helper; Qwen3.5 and any other pre-norm decoder adopts it as-is.

```swift
/// Apply RMSNorm and a (possibly quantized) Linear projection as a single
/// fused Metal kernel when all guards are met, otherwise fall back to the
/// two-dispatch path.
///
/// Guards (from `MLXFast.rmsNormQuantizedGEMV`):
/// 1. `proj` is a 4-bit `QuantizedLinear` with non-nil biases
/// 2. `x.dim(-1) == x.size` — T=1 decode (GEMV, not GEMM). Prefill and
///    seqLen > 1 decode steps (speculative, multi-turn) hit the fallback.
/// 3. `x.dtype ∈ {.float16, .bfloat16}` — the primitive is not instantiated
///    for fp32.
///
/// - Parameters:
///   - x: input [*, K] — any leading dims flatten into the GEMV
///   - normWeight: RMSNorm weight [K]
///   - eps: RMSNorm epsilon
///   - proj: the Linear (or QuantizedLinear) to apply after the norm
/// - Returns: normed-and-projected activation, shape `[*, proj.outputDims]`
public func applyNormLinear(
    _ x: MLXArray, normWeight: MLXArray, eps: Float, proj: Linear
) -> MLXArray {
    if let qProj = proj as? QuantizedLinear,
       qProj.bits == 4,
       let biases = qProj.biases,
       x.dim(-1) == x.size,
       x.dtype == .float16 || x.dtype == .bfloat16
    {
        return MLXFast.rmsNormQuantizedGEMV(
            x, normWeight: normWeight,
            w: qProj.weight, scales: qProj.scales, biases: biases,
            eps: eps, groupSize: qProj.groupSize)
    }
    let normed = MLXFast.rmsNorm(x, weight: normWeight, eps: eps)
    return proj(normed)
}
```

### Model integration points

#### Qwen3.5 attention pre-norm

File: `Libraries/MLXLLM/Models/Qwen35.swift`, `Qwen35DecoderLayer.callAsFunction`.

Current:
```swift
let r: MLXArray
if isLinear {
    r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
} else {
    r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
}
```

Problem: `inputLayerNorm(x)` is applied once and the result is consumed by `qProj`, `kProj`, `vProj` inside `selfAttn`. To fuse, we'd need the normed input to the projections to *be* the fused norm-plus-projection. This means pushing the norm **into** the attention block.

Change `Qwen35Attention.callAsFunction`:
- Drop the assumption that the caller has already normed `x`.
- Take the norm weight as an input or store it in the block.
- Call `applyNormLinear(x, normWeight:, eps:, proj: qProj)` (and k, v) instead of `qProj(normed_x)` + `kProj(normed_x)` + `vProj(normed_x)`.

This replaces **3 norm+GEMM pairs with 3 fused kernels** per attention layer (if not using `batchedQKVQuantizedGEMV`, which is a future optimization). At decode it's **net 1 fewer dispatch per attention layer** because the shared norm was previously done once.

Wait — the above arithmetic gives *more* dispatches if we apply norm three times (once each for Q, K, V). **Use `batchedQKVQuantizedGEMV` which already exists** if we're touching this path — it does all three in one kernel. That's a separate optimization though; see Future Work below. For the MLP path the math is clean: one norm + one projection → one fused kernel.

**MLP path is the clean win.** Do that first and defer attention to a follow-up spec.

For the MLP pre-norm:

```swift
// Current:
return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))

// Proposed — push norm into MLP:
return h + mlp.forwardWithPreNorm(h, normWeight: postAttentionLayerNorm.weight, eps: ...)
```

This requires an extended `UnaryLayer`-style protocol or a new MLP-specific method. Simpler: have the MLP class itself own the pre-norm weight reference, initialized at build time.

#### Concrete `FusedGateUpMLP` variant

Extend spec 001's `FusedGateUpMLP` with an optional pre-norm:

```swift
public final class FusedGateUpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    private var preNormWeight: MLXArray?  // set by the decoder layer, not owned
    private var preNormEps: Float = 1e-6

    /// Called once by the enclosing decoder layer after init so the MLP can
    /// fold the pre-MLP RMSNorm into its first projection. When nil, the
    /// caller is responsible for applying the norm before `callAsFunction`.
    public func setPreNorm(weight: MLXArray, eps: Float) {
        self.preNormWeight = weight
        self.preNormEps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp: MLXArray
        if let normWeight = preNormWeight {
            gateUp = applyNormLinear(x, normWeight: normWeight,
                                     eps: preNormEps, proj: gateUpProj)
        } else {
            gateUp = gateUpProj(x)
        }
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        return downProj(activation(parts[0]) * parts[1])
    }
}
```

The decoder layer passes the `postAttentionLayerNorm`'s weight into the MLP at layer init, and the MLP's forward no longer takes pre-normed input.

**Decoder layer change** (Qwen3.5 decoder):
```swift
init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
    // ... existing init ...
    mlp.setPreNorm(weight: postAttentionLayerNorm.weight,
                   eps: args.rmsNormEps)
}

func callAsFunction(_ x: MLXArray, ...) -> MLXArray {
    let r: MLXArray = ... // attention branch
    let h = x + r
    // MLP absorbs the post-attn norm:
    return h + mlp(h)     // was: h + mlp(postAttentionLayerNorm(h))
}
```

Same shape change applies to the input-norm → attention path once we do the attention fusion spec.

#### Gemma 4 reuse

`Gemma4AttentionBlock.fusedNormProj` becomes `applyNormLinear` (shared helper). `Gemma4SharedMLP` gets the same `setPreNorm` / `callAsFunction` treatment as `FusedGateUpMLP`. Attention-side usage is already wired; MLP-side becomes the new frontier.

### Alternative considered: kernel in MLP's own init

Have the MLP hold a reference to the norm weight as `@ModuleInfo` instead of a plain property. Benefit: serialization-friendly. Downside: double-ownership with the decoder layer's existing `postAttentionLayerNorm` causes weight-loading ambiguity. Reject — keep the pattern of a non-owning pointer set by the layer.

## Acceptance criteria

1. All covered models produce coherent output at ctx=1024: Qwen3.5-{0.8B, 2B, 4B, 9B}, Gemma 4 31B, Gemma 4 E2B, Gemma 4 E4B.
2. Decode tok/s on Qwen3.5-4B (M1 Max, 4bit, no-quant KV, ctx=1024): **≥ 88 tok/s** assuming spec 001 lands first (baseline after 001 ≈ 85).
3. Gemma 4 31B decode ≥ 14.7 tok/s at ctx=1024 (baseline after 001 ≈ 14.3).
4. Non-quantized (`.quant bf16`) decode unchanged — the fused primitive only fires for `.bits == 4`.
5. KL divergence vs bf16 baseline **unchanged within measurement noise** — the kernel is mathematically equivalent to norm + matmul.
6. Prefill unchanged (guards force fallback at T > 1).

## Measurement plan

Same matrix as spec 001, plus a 4bit-specific smoke for the 8bit fallback path (Qwen3.5-0.8B ships an 8bit MLX variant — confirm it still runs and matches bf16 KLD-wise).

```bash
scripts/benchmark.sh --model qwen35-4b,gemma4-31b --method summarization \
    --quant 4bit --kv none --context 128,1024,4096 --kld
scripts/benchmark.sh --model qwen35-0.8b --method summarization \
    --quant 8bit --kv none --context 1024
```

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Guards in `applyNormLinear` reject silently → no fusion, no regression, no perf win either | Add a one-shot diagnostic log (`MLX_DEBUG_FUSED_NORM=1`) in the helper that prints the first rejection reason per call site. Verify the fast path fires for every dense model at 4bit decode. |
| `MLXFast.rmsNormQuantizedGEMV` and manual `rmsNorm + matmul` diverge at the last ULP and cause KLD regressions on long generations | Block shipping on KLD-equivalence test (already part of existing benchmark harness via `--kld`). |
| Adding `setPreNorm` after init breaks the sendable / async flow | `setPreNorm` mutates non-MLX state only (stores a ref and a Float). Should not trip sendability. Verify under `-enable-testing` build. |
| Gemma 4's existing helper gets duplicated | Delete the private `fusedNormProj`; redirect all Gemma 4 call sites to the shared `applyNormLinear`. |

## Out of scope / follow-ups

- **Attention QKV fusion via `MLXFast.batchedQKVQuantizedGEMV`**: a bigger refactor because it produces a single concatenated result that the attention block has to split. Deferred to spec 003. Orthogonal to this spec on the MLP side.
- **MoE `gatherQuantizedMM` + pre-norm fusion**: would need a new MLX kernel (`gather_rms_norm_qgemv`). Not currently shipped. Investigate upstream feasibility.
- **`fusedNormProj` for Phi / Olmo2 / other pre-norm dense models**: once the shared helper lands, those models get it for free by switching their decoder layers to the `setPreNorm` pattern. Not blocking.

## Open questions

1. Is the per-layer `setPreNorm` ref-passing pattern tolerable, or do we want a full refactor where the MLP owns its own pre-norm weight via checkpoint key aliasing? Latter is more intrusive.
2. On bf16-loaded models (`.quant bf16` runs), guards reject and we hit the fallback path — is that acceptable, or do we also want an `rmsNormGEMV` (non-quantized) variant? (mlx-swift doesn't ship one; would need to upstream.)
3. For attention: is the MLP win alone worth this spec, or should we gate shipping on also doing the attention-side fusion? I'd recommend ship MLP, measure, then decide.
