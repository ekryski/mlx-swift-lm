# 037 — TEAL: training-free activation thresholding for MLP sparsity

**Status:** spec, exploratory (real win, but Metal kernel work is the bulk of the project)
**Branch:** new branch off alpha
**Depends on:** none architecturally — pairs naturally with TurboQuant (composes with quantized weights). The `FusedGateUpMLP` consolidation in [`Libraries/MLXLMCommon/FusedGateUpMLP.swift`](../Libraries/MLXLMCommon/FusedGateUpMLP.swift) is the integration point.
**Origin:** [`papers/beyond-quadratic-attention-on-apple-silicon.md`](../papers/beyond-quadratic-attention-on-apple-silicon.md) §5.1 + §2; [TEAL: Training-Free Activation Sparsity, ICLR 2025 (arXiv 2408.14690)](https://arxiv.org/abs/2408.14690); [Together AI writeup](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models); [reference kernel](https://github.com/FasterDecoding/TEAL)

## The insight

In a SwiGLU MLP, the post-activation hidden vector `h = silu(gate) * up` has a long-tail magnitude distribution. Most entries are small and contribute negligibly to the output of `down_proj(h)`. **Zeroing the smallest 40–50% of entries before `down_proj`** loses near-zero quality but:

1. Cuts the bytes read from `down_proj.weight` by 40–50% — the dominant cost in MLP decode is loading the down-projection weight matrix.
2. Cuts the FLOPs in the down-projection matmul by 40–50%.
3. Composes with weight quantization — sparse activations × Int4 weights both win.

The published TEAL numbers: **40–50% model-wide activation sparsity, 1.53–1.8× wall-clock decode** on Llama-2/3 + Mistral 7B–70B, training-free, ICLR 2025. The reference kernel is open (FasterDecoding/TEAL) but CUDA-only.

This is the Apple-Silicon-friendliest "skip weights per token" technique that survives honest benchmarking on modern SwiGLU stacks. Deja Vu (~1.2× on SwiGLU) and CATS (~1.15×) underperform because they predict sparsity *before* computing activations; TEAL applies the threshold *after*, which is both simpler and more accurate. The cost is needing a sparse-aware down-projection kernel.

## Why Apple Silicon specifically

Decode on M-series is memory-bandwidth bound (see [speculative-decoding-on-apple-silicon.md §1](../papers/speculative-decoding-on-apple-silicon.md)). The MLP is the dominant per-layer cost: on Qwen 3.5-9B the gate+up projection reads ~2× the input dimension worth of weight (typically 11008 × 4096 in fp16 = ~88 MB) and the down projection reads back the same. **Eliminating 40% of the down-projection weight reads is exactly the regime where M-series gets free decode throughput.**

Three Apple-specific properties matter:

1. **No tensor cores** — block-sparse matmul on Metal SIMD groups doesn't fight a fixed structured-sparsity pattern (unlike NVIDIA's 2:4). We can choose any sparsity layout that fits the simdgroup-async-copy pattern.
2. **Unified memory** — sparse activations don't need to be staged through a second buffer; the threshold mask is computed and consumed in-kernel.
3. **Composes with TurboQuant Int4 path** — the FasterDecoding reference dequantizes Int4 inside the sparse-matmul kernel; our TurboQuant kernel can be extended the same way. The dispatch count stays at one per layer.

## What we already ruled out

The investigation that produced the survey paper confirmed two adjacent dead ends. Documenting here so we don't re-tread:

- **Deja Vu / contextual-sparsity predictors on SwiGLU.** The 2023 line predicted activation sparsity *before* computing it, using a small MLP. On ReLU MLPs that worked (~80% sparsity, ~2× decode). On SwiGLU the smooth `silu` gate spreads activity across all neurons; predictors hit <50% accuracy and the win collapses to ~15% (CATS) or worse. Don't invest here.

- **TurboSparse / ProSparse via continued pretraining.** These get 85–90% sparsity by *replacing* the activation function with ReLU² and continuing pretraining. Real win in throughput; non-zero quality cost unless training compute is significant; not training-free; would require a fork of every model checkpoint we ship. Out of scope for this spec — TEAL is the no-retraining alternative.

## What this composes with

- **TurboQuant Int4/Int8 weights** — the down-projection sparse-matmul kernel can dequantize on the fly, same as the dense path.
- **`FusedGateUpMLP` (Libraries/MLXLMCommon/FusedGateUpMLP.swift)** — TEAL plugs in between the gate-up output and the `down_proj` call. The fused gate-up dispatch is unaffected.
- **MoE / `FusedGateUpSwitchGLU`** — TEAL applies per-expert; the threshold is per-layer-per-expert. Memory savings on MoE are smaller relative because experts are already sparse-by-routing, but the per-expert FLOPs win still applies.
- **Spec-decode (013, 023, etc.)** — TEAL applies to both the draft path and the target path independently. No interaction with KV cache, no rollback concern.
- **DuoAttention (spec 036) / Quest (spec 035) / KV cache eviction** — orthogonal. TEAL is MLP-side; everything else above is attention-side.

## What this does NOT compose with

- **Models that aren't SwiGLU** (older MLPs, ReLU-based small models). The thresholded activations need to be the *output* of `silu(gate) * up`. Plain ReLU MLPs already have sparsity for free; non-gated activations need their own variant.
- **Quantized activation paths** that already operate in the activation domain (e.g., TurboFlash's compressed-domain decode for sinks models). Composition needs explicit verification.
- **Models below ~3B params.** The published TEAL paper finds the sparsity dividend below ~3B is small (~1.2× at 50% sparsity vs ~1.7× at 70B+) — fewer dead activations to find. Don't spend kernel-engineering effort on small models; ship for 7B+.

## Design

### Phase 1 — Threshold-and-mask hook in `FusedGateUpMLP`

Extend `FusedGateUpMLP.callAsFunction` to apply a learned per-layer threshold:

```swift
public final class FusedGateUpMLP: Module, UnaryLayer {
    // ... existing fields ...

    /// Per-layer activation magnitude threshold. nil → dense path (current behavior).
    public var teal: TEALConfig?

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated: MLXArray
        if let twoArgActivation {
            activated = twoArgActivation(parts[1], parts[0])
        } else {
            activated = activation(parts[0]) * parts[1]
        }
        if let teal {
            return downProj.applySparse(
                activated, threshold: teal.threshold, axis: -1
            )
        }
        return downProj(activated)
    }
}

public struct TEALConfig: Sendable {
    /// Magnitude threshold below which activations are zeroed.
    public let threshold: Float
    /// Target sparsity (0.0 — 0.7). Recorded for diagnostics; not used at runtime.
    public let targetSparsity: Float
}
```

`Linear.applySparse` is a new extension that dispatches to the sparse Metal kernel (Phase 3), or falls back to a dense matmul on a masked input as a correctness reference (Phase 2).

### Phase 2 — Reference (slow but correct) sparse path

Phase 2 ships a correctness-first implementation:

```swift
extension Linear {
    func applySparse(_ x: MLXArray, threshold: Float, axis: Int) -> MLXArray {
        let mask = abs(x) .>= threshold
        let masked = x * mask    // hard-zero below threshold
        return self(masked)
    }
}
```

This gives **zero throughput improvement** (still runs the dense matmul) but unblocks Phase 3 + 4 calibration. The threshold value works correctly; only the kernel is missing.

### Phase 3 — Sparse-aware Metal kernel for `down_proj`

The kernel ingests the masked activation vector + the (possibly quantized) weight + threshold, and skips columns of the weight whose corresponding activation is zero. Two implementation tiers:

**3a — Block-sparse with simdgroup-aligned blocks.** Group activations into blocks of 32 (matching simdgroup width). A block is "active" if any element exceeds threshold; otherwise the entire 32-wide weight column block is skipped. This is the simdgroup-friendliest layout and matches Apple's threadgroup tile size.

**3b — Per-element sparse (gather pattern).** Variable-width sparse matmul; predicate per element. More flexible (theoretical max sparsity), but loses simdgroup coalescing on Metal — wins less than 3a in practice. **Skip this option** unless 3a misses headline targets.

Phase 3 ships 3a only.

The kernel composes with TurboQuant Int4 weights — the dequantization happens inside the sparse-matmul kernel as today, just with the masked activation as input.

### Phase 4 — Per-layer threshold calibration

For each supported model, run a calibration pass that:

1. Records activation magnitudes across a small calibration corpus (~5K tokens of representative text — same style as the bench prompts).
2. Picks the per-layer threshold that hits a target sparsity (typically 50%).
3. Optionally adjusts per-layer to equalize quality cost — early/late layers may tolerate higher sparsity.

Calibration result is a per-model JSON sidecar:

```json
{
  "model": "Qwen/Qwen3-9B",
  "calibration_corpus": "wikitext-2-validation",
  "target_sparsity": 0.5,
  "per_layer_thresholds": [0.012, 0.015, 0.011, ...],   // 32 entries (one per layer)
  "measured_sparsity": [0.49, 0.51, 0.48, ...],
  "ppl_dense": 6.42,
  "ppl_sparse": 6.51
}
```

Recipes ship under `recipes/teal/<model>.json`, parallel to spec 027 + 036. Calibration runs in ~10 minutes on any GPU; results check into the repo. Apple-Silicon users don't re-calibrate.

### Phase 5 — Per-model integration

Wire calibrated thresholds into model loaders for the priority models (in this order):

1. **Qwen 3.5-9B** — uniform SwiGLU, 32 layers, simplest case.
2. **GPT-OSS-20B** — same SwiGLU pattern.
3. **Gemma 4 26B-A4B (MoE)** — TEAL per-expert; calibration is more expensive (per-expert sample budget), but per-expert thresholds slot into `FusedGateUpSwitchGLU` the same way.

Integration is ~30 lines per model: load the recipe in `sanitize()`, attach the `TEALConfig` to each layer's MLP.

### Phase 6 — Bench sweep

Run decode/prefill tok/s, GenPPL, GenKLD on:

- 3 models × 3 quantization settings (4-bit, 8-bit, bf16) × {dense, TEAL-50%, TEAL-65%}
- Context lengths: 1 K, 8 K, 32 K
- Tasks: summarization (existing benchmark), GSM8K-style reasoning, RULER 32 K

Document in `benchmarks/notes/spec-037-teal-2026-MM-DD.md`. Decision gate: ship per-model only when no >1% PPL/KLD regression at the chosen sparsity.

## Implementation phases

1. **Phase 1 — `FusedGateUpMLP` hook + `TEALConfig` plumbing** (~3 days). ~80 lines + tests.

2. **Phase 2 — Reference (slow) sparse path** (~3 days). Bit-exact under `--teal-reference` flag. Lets calibration scripts run with correct numerical behavior before the kernel ships.

3. **Phase 3 — Block-sparse Metal kernel for `down_proj`** (~3–4 weeks). The biggest piece of this spec. Variants needed: dense weight, TurboQuant Int4 weight, TurboQuant Int8 weight. ~600–900 lines of Metal + ~100 lines of MLXFast wiring + correctness tests against the Phase 2 reference path.

4. **Phase 4 — Calibration script** (~1 week). Standalone Python (PyTorch reference, runs anywhere). Generates per-model recipes. Lives in `tools/teal-calibrate/`.

5. **Phase 5 — Per-model integration** (~3 days per model). 3 models = ~2 weeks.

6. **Phase 6 — Bench sweep + decision** (~1 week). Per-model decision on whether to ship.

**Total scope: ~7–9 weeks of engineering.** Phase 3 is the long pole.

## Expected impact

**On supported models, post-calibration:**

- **Qwen 3.5-9B-4bit at 50% sparsity, 1 K ctx:** projected **+30–40% decode tok/s** (from current ~54 → ~70–76 tok/s on M1 Max). The MLP fraction of decode dominates at small contexts; sparse down-proj is exactly the bottleneck.
- **At 32 K ctx the relative win shrinks to ~+15–20%** because KV-cache reads start to compete with weight reads. TEAL only helps weight reads.
- **GPT-OSS-20B:** similar relative win (+30–40% decode at 1 K).
- **Gemma 4 26B-A4B (MoE):** smaller relative win (+10–20%) — experts are already activation-sparse via routing, so TEAL's per-expert reduction is a smaller marginal gain.
- **Quality:** PPL/KLD drift <1% at 50% sparsity, per the TEAL paper's published numbers on Llama-2/3 + Mistral. Verified per model in Phase 6.
- **Memory:** zero reduction. TEAL is a *FLOPs and bandwidth* sparsifier, not a *memory* sparsifier.

**Where this DOESN'T help:**

- Long-context (>32 K) where KV is already dominant.
- Small models (<3B) where the activation distribution is denser.
- Pure-Mamba / pure-GDN layers (no SwiGLU MLP to sparsify in the recurrent paths).

## Risks

1. **Phase 3 kernel work is the load-bearing piece.** A block-sparse Metal kernel that handles TurboQuant Int4 dequantization in-flight is non-trivial (~3–4 weeks). If it lands at <40% sparsity-utilization (i.e., the kernel runs slower per-element than dense) the headline win evaporates. Mitigation: prototype the simdgroup-block path against `mlx::matmul` baseline at 50% sparsity *before* committing to per-model integration. If the prototype doesn't show ~+30% on a synthetic benchmark, revisit.

2. **Threshold-calibration drift across workloads.** A threshold tuned on wikitext may produce unexpected quality drops on code generation or reasoning chains. Mitigation: Phase 4 includes a multi-domain calibration set (wikitext + code + math); Phase 6 verifies on each domain.

3. **Reasoning-model quality.** Reasoning chains stress MLP layers more uniformly than declarative text. The 50% sparsity default may be too aggressive for o1/R1-class models. Mitigation: bench includes GSM8K + AIME-style prompts; lower the sparsity target for reasoning models if regression appears.

4. **Composition with the existing `FusedGateUpSwitchGLU` MoE path.** Per-expert thresholds need to be loaded as ragged `[E]` tensors and applied per-expert. The reference TEAL paper doesn't cover MoE; we're extrapolating. Mitigation: Phase 5 ships Qwen 3.5 + GPT-OSS first (dense), then Gemma 4 (MoE) only after the dense path is stable.

5. **Cliff at the threshold.** Hard-zeroing creates non-smooth gradients (irrelevant for inference, but) and can amplify quantization noise on already-quantized weights. The Together AI writeup notes the effect is small at 50% but grows at 70%+. Mitigation: Phase 6 sweeps 50/55/60/65% sparsity; pick per-model based on the PPL/KLD vs throughput Pareto frontier.

6. **Apple's compiler may already strip masked activations on the dense path.** Modern MLX compile passes do dead-code elimination on zero-multiplied paths in some kernels. If the dense matmul kernel already short-circuits zero activations efficiently, Phase 2 (reference) might be most of the win and Phase 3 (kernel) is overkill. Mitigation: Phase 2 includes a baseline measurement — if the masked-input dense matmul is already +20% faster than non-masked, Phase 3 is a smaller (and easier) marginal step.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/FusedGateUpMLP.swift` | `TEALConfig` field; threshold-and-mask call site |
| `Libraries/MLXLMCommon/TEAL.swift` (new) | `TEALConfig`, recipe loader, `Linear.applySparse(...)` extension |
| `Libraries/MLXLMCommon/MLXFast+TEAL.swift` (new) | `sparseMatmulMetal(...)` wrapper around the new kernel |
| `Libraries/MLXFast/Sources/Metal/teal_sparse.metal` (new) | Block-sparse `down_proj` kernel (Phase 3) |
| `Libraries/MLXLLM/Models/Qwen35.swift` | `sanitize()` loads recipe + attaches `TEALConfig` per layer |
| `Libraries/MLXLLM/Models/GPTOSS.swift` | Same |
| `Libraries/MLXLLM/Models/Gemma4.swift` | MoE per-expert threshold (Phase 5) |
| `Libraries/MLXLMCommon/FusedGateUpSwitchGLU.swift` | Mirror the TEAL hook for MoE |
| `recipes/teal/qwen35-9b.json` (new) | Phase 4 calibration output |
| `recipes/teal/gpt-oss-20b.json` (new) | |
| `recipes/teal/gemma4-26b-a4b.json` (new) | |
| `tools/teal-calibrate/` (new) | Standalone calibration script (PyTorch reference) |
| `scripts/benchmark.sh` | `--teal <recipe>` CLI flag, `--teal-reference` (Phase 2 fallback) |
| `Tests/MLXLMCommonTests/TEALTests.swift` (new) | Bit-exact reference path; Phase 3 kernel matches reference |
| `benchmarks/notes/spec-037-teal-2026-MM-DD.md` (new) | Phase 6 sweep results |

## Why this is Tier 4

Three reasons it ranks below Quest (035) and DuoAttention (036):

1. **Bigger kernel project.** Phase 3 is a ~3–4 week Metal kernel; specs 035 + 036 each need ~1–2 weeks of kernel work (or none, in 035's MLX-only Phase 2 path).
2. **Doesn't help at long context.** KV-cache reads dominate beyond 32 K; TEAL only helps weight reads. Quest and DuoAttention attack the long-context regime directly. TEAL's headline win is at short context where MLP dominates.
3. **More uncertain quality story.** Quest is lossless-by-construction (top-k over a fully retained cache); DuoAttention is well-validated on Llama-2/3 in the published paper; TEAL's published numbers are 50% sparsity at <1% PPL drift, but the calibration corpus matters and reasoning models specifically aren't in the published benchmarks.

That said, **TEAL is the only spec in this trio that helps the small-context decode-tok/s regime** — exactly the M1 Max measurement that anchors `speculative-decoding-on-apple-silicon.md`. If the M-series user's primary workload is short-prompt agent loops (where the existing 54 tok/s is the bottleneck and KV is small), TEAL is the bigger win for them than 035 or 036.

Suggested ordering: **after Tier 3 stabilises and after Quest (035) ships** (which establishes the kernel-engineering pattern for sparse-matmul on Metal). The kernel work pattern transfers between specs.
