# Gemma 4 26B A4B IT — Optimization Plan

**Date**: 2026-04-05
**Status**: Pre-implementation (model not yet benchmarked)

## Architecture

- **Type**: Pure attention MoE (NO GatedDeltaNet/SSM — fully parallelizable)
- **Layers**: 30 total — 25 local (sliding window) + 5 global (full attention)
- **Local attention**: 8 KV heads, head_dim=256, sliding_window=1024
- **Global attention**: 2 KV heads, global_head_dim=512
- **MoE**: 128 experts, 8 routed + 1 shared per token
- **Vocabulary**: 262,144
- **Context window**: 256K tokens
- **Total params**: 26B, ~4B active per token

## Key Architectural Differences from Qwen3.5-35B-A3B

| Aspect | Qwen3.5-35B-A3B | Gemma 4 26B A4B |
|--------|-----------------|-----------------|
| SSM/recurrence | 30 GatedDeltaNet layers (75%) | **None** — fully parallel |
| Attention | 10 full attention layers (25%) | **30 layers** (25 sliding + 5 global) |
| Head dim | 128 (uniform) | **256 local, 512 global** |
| MoE experts | 256 experts, top-8 | 128 experts, top-8 + 1 shared |
| KV cache pattern | Uniform (all layers same) | **Mixed** (sliding window + full) |
| Prefill bottleneck | GatedDeltaNet sequential O(T) | **None** — fully parallelizable |
| Decode bottleneck | Dispatch overhead (43%) | Likely similar dispatch overhead |

## Why Gemma 4 Is a Better Target for Fused Kernels

Since there's no sequential recurrence, the **entire forward pass is parallelizable**.
Every layer is attention + MoE — both are matmul-based. This means:

1. **Prefill scales to full GPU TFLOPS** — no GDN bottleneck capping at ~500 tok/s
2. **Dispatch overhead is the ONLY bottleneck** — our Metal trace showed 43% of
   decode time is inter-encoder gaps. Reducing dispatches has proportionally more
   impact when there's no sequential kernel eating into the savings.
3. **Fused MoE kernel has maximum ROI** — every dispatch eliminated translates
   directly to wall-clock improvement (no sequential work to mask the gains).

## Optimization Strategy

### Tier 1: Direct transfers from Qwen3.5 work (zero effort)

These optimizations are model-agnostic and apply immediately:

1. **FusedGateUpSwitchGLU** — If Gemma 4's model file has fused gate_up weights
   (or split gate_proj + up_proj that can be concatenated), our single-dispatch
   gatherQuantizedMM path eliminates 1 dispatch per MoE block × 30 layers.
   Expected: +3-7% decode.

2. **MoE sort threshold** — Same `SwitchGLU` / `gatherQuantizedMM` path.
   Threshold=128 is model-agnostic. Expected: same behavior.

3. **trackPerplexity flag** — Universal. Production callers skip softmax chain.

4. **.item() sync skip** — If Gemma 4 is used as a non-thinking model (no
   thinkStartTokenId), the decode pipeline benefits from the sync elimination.

### Tier 2: Gemma 4-specific optimizations

5. **Fused MoE Metal kernel** — Full gate→sort→gatherQMM×2→activation→gatherQMM→
   weighted_sum in a single dispatch. For Gemma 4 with 128 experts and top-8:
   - Current: ~5-6 dispatches per MoE block × 30 layers = ~150-180 dispatches
   - Fused: 1 dispatch × 30 = 30 dispatches
   - Savings: ~120-150 dispatches × ~123us gap = ~15-18ms saved per token
   - At current 22ms/token decode → **40-45% improvement → potentially 70+ tok/s**

   This is the single highest-impact optimization for Gemma 4. The absence of
   GDN sequential overhead means ALL saved dispatch time converts to speed.

6. **Sliding window KV cache optimization** — 25/30 layers use sliding_window=1024.
   At inference time, these layers' KV cache is bounded at 1024 tokens regardless
   of context length. This means:
   - TurboQuant compression is less valuable for local layers (small cache already)
   - Memory pinning can be tighter (KV budget is deterministic for local layers)
   - Boundary layer protection should only apply to the 5 global layers

7. **Large head dim attention optimization** — head_dim=256 (local) and 512 (global)
   are much larger than typical (64-128). This affects:
   - Attention compute: O(T × head_dim) per query-key dot product
   - The attention kernel's tiling strategy may need adjustment for these sizes
   - TurboQuant KV compression is more impactful (larger heads = more data per token)

8. **Morton order for expert weights** — With 128 experts and top-8 routing,
   Z-order curve indexing could improve gatherQuantizedMM cache locality.
   The sort A/B test showed 38-48% regression without sorting for Qwen3.5 —
   similar benefit expected for Gemma 4.

### Tier 3: Research-level

9. **Memory pinning by model config** — Pre-compute wired budget:
   - Model weights: 26B × 4 bits / 8 = ~13GB
   - KV cache (global layers only at max context): 5 layers × 2 heads × 512 dim × 256K × 2 bytes = ~2.6GB
   - KV cache (local layers, bounded): 25 layers × 8 heads × 256 dim × 1024 × 2 bytes = ~105MB
   - Workspace: ~1-2GB peak intermediate tensors
   - Total: ~17GB — fits comfortably in 64GB M1 Max

## Benchmarking Plan

```bash
# Step 1: Baseline (when model is available in mlx-community)
scripts/benchmark.sh --model gemma4-26b-a4b --quant 4bit --kv none --method summarization --quick
scripts/benchmark.sh --model gemma4-26b-a4b --quant 4bit --kv turbo4v2 --method summarization --quick

# Step 2: Metal System Trace for dispatch analysis
# (same xctrace approach as Qwen3.5)

# Step 3: Compare with Qwen3.5 to validate our optimization transfer
```

## Performance Expectations (Theoretical)

Assuming similar M1 Max hardware (400 GB/s, 10.4 TFLOPS):

**Active parameters**: ~4B at 4-bit = ~2GB weight reads per decode token
**Theoretical decode max**: 400 / 2.0 = ~200 tok/s (bandwidth-limited)
**Realistic with dispatch overhead**: ~100-120 tok/s (with fused MoE kernel)
**Without fusion**: ~60-70 tok/s (similar to GPT-OSS-20B which is also pure attention MoE)

**Prefill**: With no sequential bottleneck, prefill should approach theoretical
compute-bound maximum: 10.4 TFLOPS / (2 × 4B × 4/16) = ~2,600 tok/s
Realistic with overhead: ~1,000-1,500 tok/s

These would be significantly faster than Qwen3.5 (~500 tok/s prefill, ~50 tok/s decode)
because there's no GDN sequential kernel limiting throughput.

## Notes

- Gemma 4 uses SiGLU activation (similar to SiLU-gated, like Qwen3.5) — our
  FusedGateUpSwitchGLU activation function is compatible
- The shared expert (1 per token, always active) is the same pattern as Qwen3.5's
  `Qwen35SparseMoeBlock.sharedExpert` — no changes needed
- The sliding window attention pattern is similar to GPT-OSS-20B's alternating
  full/sliding pattern — our existing `RotatingKVCache` handles this
- Vocabulary size (262,144) is similar to Qwen3.5 (248,320) — the tied embeddings
  / LM head optimization applies equally
