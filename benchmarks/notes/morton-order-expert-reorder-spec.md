# Morton Order & Profile-Guided Expert Reordering — Design Spec

**Date**: 2026-04-05
**Status**: Speculative — A/B test confirms ordering matters, needs calibration infrastructure

## Background

The sort threshold A/B test showed expert ordering for gatherQuantizedMM matters
enormously: 38-48% prefill regression without sorting. The expert reorder A/B test
showed random shuffling causes -3.2% prefill / -1.7% decode regression, confirming
the original training order has natural locality.

This raises the question: can we do BETTER than the original order with an
intelligent reordering based on actual expert co-selection patterns?

---

## Does Morton Ordering Apply to Non-MoE Models?

**Short answer: Yes, but the opportunity is different.**

### MoE Models (Qwen3.5, GPT-OSS, Gemma 4, etc.)

The primary memory layout concern is **expert weight locality** in gatherQuantizedMM.
With 128-256 experts and top-8 routing, the selected experts are scattered across
the weight tensor. Morton/Z-order or profile-guided reordering directly addresses
this by clustering frequently co-selected experts.

**Impact**: High (38-48% from sort alone, potentially more from intelligent layout)

### Dense Transformer Models (Llama, Phi, Gemma dense, etc.)

Dense models don't have expert selection — every weight is read every token.
Morton ordering doesn't apply to the weight layout itself.

However, there ARE memory layout opportunities:

1. **KV Cache Layout**: The KV cache grows linearly with context. The memory layout
   of `[batch, heads, tokens, head_dim]` could be reordered. For GQA models where
   `num_kv_heads < num_attention_heads`, the repeat pattern means multiple query heads
   access the same KV head data. Ensuring KV heads are contiguous in memory (which
   they already are) is the equivalent optimization.

2. **Attention Score Tiling**: For long-context attention (T > 4096), the attention
   score matrix `[heads, T, T]` exceeds L2 cache. Tiled attention (FlashAttention-style)
   addresses this — similar in spirit to Morton ordering but at the algorithm level.

3. **Weight Quantization Group Layout**: For quantized weights with groupSize=64,
   the group boundaries determine cache line alignment. Ensuring group boundaries
   align with cache line sizes (128 bytes on Apple Silicon) can reduce partial
   cache line reads. This is a minor but measurable optimization.

### Hybrid Models (Qwen3.5, NemotronH, Jamba, etc.)

Hybrid models have BOTH MoE and SSM/attention layers. Morton ordering applies to
the MoE expert weights (same as pure MoE). The SSM layers (GatedDeltaNet, Mamba)
don't benefit from Morton ordering because their projections are dense (all weights
read every token).

---

## Profile-Guided Expert Reordering — Calibration Spec

### Goal

Reorder expert indices so that frequently co-selected experts are adjacent in
memory. This provides the sort's cache locality benefit WITHOUT runtime sort overhead.

### Calibration Step

**What**: Run a calibration pass over a representative text corpus (e.g., 1000
sentences from WikiText-2 or a model-specific evaluation set). For each token,
record which top-K experts are selected by the gate.

**Output**: A permutation table `perm[0..numExperts-1]` that maps original expert
index to reordered position.

**Algorithm**:

```
1. COLLECT co-selection counts:
   For each layer l:
     co_select[l] = zeros(numExperts, numExperts)  // co-selection matrix
     For each token t in calibration corpus:
       selected = gate(x_t)  // top-K expert indices
       For each pair (i, j) in selected:
         co_select[l][i][j] += 1
         co_select[l][j][i] += 1

2. BUILD affinity graph:
   For each layer l:
     affinity[l] = co_select[l] normalized by diagonal (Jaccard or cosine)

3. FIND optimal ordering (per layer or global):
   Option A: Spectral ordering
     - Compute Fiedler vector (2nd eigenvector of graph Laplacian)
     - Sort experts by Fiedler vector value
     - This minimizes total "distance" between co-selected experts

   Option B: Greedy nearest-neighbor
     - Start with most-selected expert
     - Greedily add the expert most co-selected with current cluster
     - O(E²) but simple and effective

   Option C: Hierarchical clustering
     - Cluster experts by co-selection affinity
     - Within each cluster, sort by sub-cluster affinity
     - Produces a dendrogram that maps naturally to Morton order

4. APPLY permutation:
   - Reorder expert weights along axis 0: weight[perm] for each projection
   - Reorder gate weight columns: gate_weight[:, perm]
   - Both done at load time in sanitize() — zero runtime cost
```

### Per-Layer vs Global Ordering

Each layer has its own gate and expert specialization. The optimal ordering may
differ per layer. Options:

- **Per-layer ordering**: Best locality but requires 40 different permutation tables.
  Each layer's weights are reordered independently. Gate weights adjusted per layer.

- **Global ordering**: Single permutation across all layers. Simpler but suboptimal
  for individual layers. Uses averaged co-selection across all layers.

- **Hybrid**: Cluster experts globally (for memory layout), fine-tune within clusters
  per layer. Best tradeoff — the coarse ordering is global (cache pages), the fine
  ordering is per-layer (cache lines).

**Recommendation**: Start with global ordering (simpler), measure impact, then
try per-layer if there's a signal.

### Calibration Corpus Requirements

- **Size**: 1000-10000 sentences (enough for stable co-selection statistics)
- **Content**: Representative of deployment domain (general text, code, etc.)
- **Processing**: Prefill only (no generation needed — just gate routing decisions)
- **Time**: ~30-60 seconds per layer on M1 Max for 10K sentences
- **Storage**: Co-selection matrix per layer: numExperts² × 4 bytes = 256² × 4 = 256KB
  Total for 40 layers: ~10MB

### Implementation Plan

```
Phase 1: Calibration infrastructure
  - Add CalibrateExpertOrder utility that runs gate on calibration text
  - Collect per-layer co-selection matrices
  - Save to JSON/binary file alongside model weights

Phase 2: Ordering algorithm
  - Implement spectral ordering (Fiedler vector via eigendecomposition)
  - Generate permutation table per layer (or global)
  - Save permutation as part of calibration output

Phase 3: Apply at load time
  - Extend sanitize() to read permutation file
  - Reorder expert weights + gate columns using saved permutation
  - Verify model output is identical (permutation is a bijection)

Phase 4: Benchmark
  - Compare reordered vs original at various contexts
  - Measure with and without runtime sort (reordered should work WITHOUT sort)
  - If reordered + no-sort > original + sort: permanent win
```

### Expected Impact

The sort threshold A/B showed:
- Sort ON vs OFF at T=1024: 472.7 vs 268.5 (+76% from sort)
- The sort cost at T=128: -25% overhead

Profile-guided reordering would provide the sort's locality benefit permanently,
without the runtime sort overhead. Expected improvement over current baseline:
- **At T=128**: +25% (recover sort overhead currently saved by threshold=128)
- **At T≥1024**: +0-5% (sort already provides locality; reorder is equivalent)
- **At decode (T=1)**: +0-3% (sort skipped at decode; reorder helps marginally)

The biggest win would be for models where the sort threshold causes a tradeoff
(small T hurt by sort overhead, large T need sort for locality). Profile-guided
reordering eliminates this tradeoff entirely.

### Morton Order Variant

Instead of profile-guided reordering, a purely structural approach:

**Z-order curve** on expert_index × output_dimension:
- Interleave bits of expert index and row index in the weight tensor
- This creates a fractal-like memory layout where "nearby" (expert, row) pairs
  are adjacent in memory
- Doesn't require calibration data — purely mathematical

**Pros**: No calibration needed, deterministic, benefits all access patterns
**Cons**: Doesn't account for actual co-selection patterns, may be suboptimal
for specific models, requires modifying gatherQuantizedMM indexing

**Recommendation**: Profile-guided > Morton for MoE (data-dependent is better).
Morton is better for dense models where access patterns are uniform.

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `Libraries/MLXLMCommon/ExpertCalibration.swift` (NEW) | Calibration runner + ordering algorithm |
| `Libraries/MLXLLM/Models/Qwen35MoE.swift` | Apply permutation in sanitize() |
| `Libraries/MLXLMCommon/SwitchLayers.swift` | Optional: skip sort when reordered |
| `scripts/calibrate-experts.sh` (NEW) | CLI wrapper for calibration |
| `benchmarks/notes/expert-reorder-calibration-results.md` (NEW) | Results |

## Summary

**Morton order for non-MoE models**: Limited. Dense models read ALL weights every
token — there's no selection pattern to optimize. The analog optimizations are KV
cache layout (already good), attention tiling (FlashAttention-style), and quantization
group alignment with cache lines. None have the 38-48% impact that MoE expert
ordering provides.

**Profile-guided calibration** would involve:
1. **Collect**: Run gate routing on 1K-10K calibration sentences, record which
   experts are co-selected
2. **Analyze**: Build co-selection affinity matrix per layer (256×256 for Qwen3.5)
3. **Order**: Spectral ordering (Fiedler vector of graph Laplacian) or greedy
   nearest-neighbor to find optimal permutation
4. **Apply**: Reorder weight tensors + gate columns at load time in `sanitize()`
   — zero runtime cost

**The killer feature**: eliminates the sort/no-sort tradeoff. Currently we sacrifice
25% at T=128 (sort overhead) to gain 48% at T≥1024 (sort locality). Profile-guided
reordering gives the locality benefit at ALL batch sizes with zero runtime overhead
— permanent static layout.

---

## Open Questions

1. **Does the optimal ordering change with quantization?** The 4-bit weight packing
   changes memory access patterns. Calibration should be done at the target precision.

2. **Does the ordering transfer across similar models?** If Qwen3.5-35B-A3B's
   ordering works for Qwen3.5-27B, we could share calibration data.

3. **How stable is the ordering across different prompts?** If co-selection patterns
   are highly input-dependent, a global ordering may not help. Need to measure
   variance across calibration subsets.

4. **Can the gate be fine-tuned to prefer adjacent experts?** A regularization term
   during training that penalizes selecting far-apart experts (in reordered space)
   would make the model naturally cache-friendly. This is a training-time optimization.
