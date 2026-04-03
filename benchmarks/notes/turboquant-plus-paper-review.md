# TurboQuant Plus Paper Review: Block Size, Layer-Aware V Compression, MoE V Frontier

**Date**: 2026-04-02
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Sources**:
- [Block Size Experiment](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/block-size-experiment.md) — Tom Turney
- [Layer-Aware V Compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md) — Tom Turney
- [MoE V Compression Frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md) — Tom Turney

## References

- [TurboQuant Plus repo](https://github.com/TheTom/turboquant_plus) — Tom Turney's research workspace
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — Algorithm 1 (MSE-optimal), Algorithm 2 (QJL correction)
- [SwiftLM](https://github.com/SharpAI/SwiftLM) — Swift/Metal TurboQuant implementation with fused SDPA dequant
- [Draw Things: Metal Quantized Attention](https://releases.drawthings.ai/p/metal-quantized-attention-pulling) — Int8 quantized attention on Metal
- Our implementation notes: [turbo-opt-4-hot-window-and-6-boundary-layers.md](turbo-opt-4-hot-window-and-6-boundary-layers.md), [turbo-opt-turboflash-attention.md](turbo-opt-turboflash-attention.md)

## Purpose

Review three TurboQuant Plus papers for relevance to our implementation. Specifically:
1. Should we change our TurboFlashAttention block size?
2. Should we re-enable boundary layer protection?
3. Are there MoE-specific compression strategies to adopt?
4. Is inference batching applicable?

---

## Paper 1: Storage Block Size Optimization

### What "Block Size" Means in This Paper

**This is NOT the same as our TurboFlashAttention token block size.**

- **Their block size**: Storage packing granularity — how many quantized elements share a single norm value. Controls the ratio of packed data to metadata (norms) in the on-disk/in-memory turbo struct.
- **Our flash block size (B=64)**: How many tokens each SIMD group processes in the fused attention kernel. Controls GPU parallelism vs per-block overhead.

These are orthogonal parameters.

### Key Findings

- Block sizes tested: 32, 64, 128 elements per norm
- **Quality**: Identical PPL across all block sizes, all models, all context lengths
- **Compression**: block_size=128 achieves 5.12x vs 4.57x at block_size=32 (amortizing 2-byte norm over more elements)
- **Speed**: No measurable change on M4 Max. M2 Pro showed 3-7% decode improvement at block_size=128
- **Recommendation**: Adopt block_size=128 as default

### Relevance to Our Implementation

**We're already at block_size=128.** Our storage format stores one norm per token per head, and each vector has D=128 elements. So 128 elements share one norm value — exactly their recommended setting.

No changes needed to our storage format. No changes needed to our flash attention B=64 token block size (different concept entirely).

---

## Paper 2: Layer-Aware V Compression (Boundary Layer Protection)

### What They Did

**LA-V7 policy**: First 2 and last 2 KV layers get V cache at q8_0 (8.5 bits/val). All other layers get V cache at turbo2 (2.5 bits/val). K cache stays at q8_0 throughout (unchanged).

### Results on Pure-Attention Models

| Model | q8_0/q8_0 | q8_0/turbo3 | q8_0/turbo2 | LA-V7 (boundary) |
|-------|-----------|-------------|-------------|-------------------|
| phi-4 (14B) | 4.690 | 4.742 | 4.835 | 4.784 |
| Qwen2.5-7B Q4_K_M | 6.577 | 6.707 | 6.911 | 6.835 |

LA-V7 recovers ~50% of the PPL gap between turbo2 and turbo3 on pure-attention models. No speed penalty.

### Critical Finding: Hybrid Architecture Bug

Their boundary selection used raw transformer layer index:
```cpp
const bool is_boundary = (il < 2 || il >= n_layer - 2);
```

On Qwen3.5 hybrid models (GatedDeltaNet), KV attention only occurs on every Nth layer. This means:
- **Qwen3.5-27B** (64 total layers, 16 KV layers): With 2+2 boundary, only 1 of 16 KV layers actually gets protection
- The intended policy was never actually executed on hybrid models

**Our implementation did NOT have this bug.** We used `kvOrdinal` (index within KV-only layers) and `numKVLayers`:
```swift
let isBoundary = protectedLayers > 0
    && (kvOrdinal < protectedLayers || kvOrdinal >= numKVLayers - protectedLayers)
```

This correctly targeted the first/last KV-layer ordinals, not raw layer indices.

### Why Boundary Protection Didn't Help Us

Despite correct KV-ordinal targeting, our Qwen3.5-2B benchmarks showed:
- **turbo3v2 vs turbo3v2-p2**: -5% to -17% gen tok/s, no KLD improvement
- **turbo4v2 vs turbo4v2-p2**: -5% to -15% at long contexts, no KLD improvement

Reasons this differs from the paper's positive results:
1. **Model architecture**: Qwen3.5 hybrid (GatedDeltaNet) may be less sensitive to boundary layer quality than pure-attention models (phi-4, Qwen2.5)
2. **Protection scope**: Paper protects V-only at q8_0 with q8_0 K throughout. We protected both K and V at full FP16, which breaks the fully-compressed fast path (all-turbo attention kernels can't be used for FP16 boundary layers)
3. **Context length**: Paper notes benefit diminishes with context (at 16K, only -0.006 PPL improvement on phi-4). Our benchmarks test up to 131K.
4. **Our compression regime**: We use turbo3v2/turbo4v2 (MSE-optimal rotation + codebook), which may already handle boundary layers well enough that additional protection is redundant

### Recommendation

**Do not re-enable for Qwen3.5.** The evidence is clear: correctly-targeted boundary protection showed no quality benefit on this hybrid architecture.

**Future consideration**: If we test on pure-attention models (phi-4, Qwen2.5-7B), boundary V protection could be worth re-enabling as an opt-in config (e.g., `turbo4v2-bv2`). The implementation would need to:
- Protect V-only (not K) at boundary layers
- Use a lighter protection level (q8_0 or turbo4 for V) rather than full FP16
- Ensure boundary layers still use compressed-domain attention (not falling back to FP16 SDPA)

---

## Paper 3: MoE V Compression Frontier

### Key Findings

- On Qwen3.5-35B-A3B MoE: `q8_0-K + turbo2-V` with boundary V protection achieves 7.53x V compression (vs 5.12x for turbo3-V)
- PPL within 0.4-1.0% of q8_0 baseline
- 32K decode 2-3% faster than q8_0/turbo3
- **K-side quantization dominates MoE decode costs** (6.3% overhead) while V-side contributes only 2.1% — validates aggressive V compression with conservative K handling
- Asymmetric `q8_0-K + turbo-V` rescues sensitive Q4_K_M models where symmetric turbo fails

### Relevance to Our Implementation

Our turbo3v2 (3-bit K + 2-bit V) and turbo4v2 (4-bit K + 2-bit V) already implement the asymmetric philosophy. The paper validates that aggressive V compression is the right direction.

Key difference: they use **q8_0 for K** (MLX's built-in affine 8-bit quantization), not turbo-compressed K. This is a hybrid approach we don't currently support — it would require mixing affine-quantized K with turbo-quantized V within the same cache layer. Not trivial, but could be worth exploring for MoE models where K quality is critical.

---

## Inference Batching Analysis

### Current State

mlx-swift-lm is single-user, single-sequence inference. B=1 always during generation. No multi-request batching.

### Applicable Batching Forms

| Form | Applicable? | Impact | Notes |
|------|-------------|--------|-------|
| Multi-request batching | No | N/A | Single-user on-device library |
| Prefill chunking | Already done | N/A | `fusedEncodeWHT()` batch-encodes all prefill tokens |
| Cross-layer encode batching | Theoretically | Minimal | Data dependencies between layers prevent meaningful batching |
| Speculative decoding | Yes | ~2-3x decode | Generate N candidates with draft model, verify in one pass. Major feature, orthogonal to KV compression |
| Token-level multi-turn batching | No | N/A | Single-sequence generation |

### Recommendation

Traditional inference batching is not applicable to our single-sequence use case. The biggest "batching" opportunity is **speculative decoding**, which is a fundamentally different feature (generation strategy, not kernel optimization). Not in scope for TurboQuant work.

---

## Summary of Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Change flash attention block size? | **No** | Paper's "block size" is storage norm sharing, not token blocking. Different concept. Our B=64 is well-tuned from sweep. |
| Change storage block size? | **No** | Already at block_size=128 (1 norm per D=128 element vector). Matches their recommendation. |
| Scale flash block size by context? | **No** | Our sweep showed B=64 optimal across 512-8K tokens. No evidence for adaptive sizing. |
| Re-enable boundary layer protection? | **No** (for Qwen3.5) | Correctly implemented, still showed no benefit on hybrid architecture. Paper confirms benefits are model-dependent. |
| Add boundary protection as opt-in flag? | **Future** | Worth testing on pure-attention models (phi-4, Qwen2.5) if we expand model support. V-only protection at q8_0, not full FP16. |
| Implement inference batching? | **No** | Not applicable to single-sequence on-device inference. Speculative decoding is separate feature. |
| Hybrid q8_0-K + turbo-V config? | **Future** | MoE paper shows this works well. Would require mixing affine K cache with turbo V cache in same layer. |
