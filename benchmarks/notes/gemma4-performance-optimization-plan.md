# Gemma 4 Performance Optimization Plan

**Date**: 2026-04-07
**Status**: Analysis complete, implementation pending
**Models**: E2B (dense+PLE), E4B (dense+PLE), 26B-A4B (MoE), 31B (dense)

## Current Baselines (M1 Max, 4-bit, no-quant KV)

| Model | Prefill @4K | Decode @4K | Architecture |
|-------|:-----------:|:----------:|:-------------|
| E2B | 1464 tok/s | 55 tok/s | 35 layers, PLE, KV sharing (20 layers) |
| 26B-A4B | 567 tok/s | 25 tok/s | 30 layers, 128 experts top-8, MoE |
| 31B | 68 tok/s | 13 tok/s | 60 layers, dense, large head dims |

## Optimization Opportunities

### Tier 1: Quick Wins (low effort, high impact)

#### 1. Increase Prefill Chunk Size

Gemma 4 is pure attention (no GatedDeltaNet sequential bottleneck). Default `prefillStepSize`
is `windowSize` (512-1024) or 512. Pure attention models can safely use 4096+.

**Implementation**: Override `prepare()` in Gemma4TextModel or set `prefillStepSize: 4096`
in the benchmark's GenerateParameters.

**Expected**: 15-25% TTFT reduction (fewer prefill passes, better GPU utilization).

#### 2. Replace Manual v_norm with MLXFast.rmsNorm

Current `rmsNormNoScale()` is hand-rolled (3 dispatches: square, mean, rsqrt*multiply):
```swift
private func rmsNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let meanSquare = x.square().mean(axis: -1, keepDims: true)
    return x * rsqrt(meanSquare + eps)
}
```

**Implementation**: Replace with `MLXFast.rmsNorm(x, weight: MLXArray.ones([headDim]), eps: eps)`.
Need to verify MLXFast fuses into a single kernel. Alternatively, create a `weight = ones` once
in init and reuse.

**Expected**: 2-4% total decode (1-2% per layer x 30-60 layers).

#### 3. FusedGateUpSwitchGLU for 26B MoE

Currently uses `SwitchGLU` (2 separate gatherQuantizedMM dispatches for gate + up). The
FusedGateUpSwitchGLU pattern saves 1 dispatch per MoE block by fusing gate+up into a single
gatherQuantizedMM call.

**Implementation**: In `Gemma4.swift` sanitize, fuse `experts.gate_proj` + `experts.up_proj`
into `experts.gate_up_proj` (concatenate weights on axis 1). Switch the module from `SwitchGLU`
to `FusedGateUpSwitchGLU` with `geluApproximate` activation.

**Expected**: 10-15% decode for 26B (saves 30 Metal dispatches per token, 1 per MoE layer).

#### 4. Cache PLE Embeddings During Decode

`embedTokensPerLayer(inputs)` is called every forward pass. During decode, `inputs` is a single
token ID. The embedding lookup is cheap, but `perLayerModelProjection(h)` is a full Linear
matmul projecting hidden_size -> numLayers * plDim (e.g., 1536 -> 35*256 = 8960).

**Implementation**: The embedding part (`embedPL(inputs)`) can be cached since token IDs don't
change during a decode step. The projection part depends on `h` which changes every step, so
it can't be cached. However, the projection could potentially be split per-layer and computed
lazily only for layers that use PLE (skip for KV-shared layers that don't need it).

**Expected**: 5-10% decode for E2B/E4B models with PLE enabled.

---

### Tier 2: Medium Effort, Significant Gains

#### 5. RotatingKVCache peek() Temporal Order Caching

`peek()` calls `temporalOrder()` which reorders the circular buffer into sequential order via
concatenation. For KV-shared models, multiple layers call `peek()` on the same donor cache
per token, redundantly recomputing the same reordering.

**Implementation**:
```swift
private var cachedTemporalKeys: MLXArray?
private var cachedTemporalValues: MLXArray?
private var peekCacheValid = false

public override func peek() -> (MLXArray, MLXArray)? {
    guard let keys, let values else { return nil }
    if !peekCacheValid {
        cachedTemporalKeys = temporalOrder(keys)
        cachedTemporalValues = temporalOrder(values)
        peekCacheValid = true
    }
    return (cachedTemporalKeys!, cachedTemporalValues!)
}

// Invalidate in update():
peekCacheValid = false
```

**Expected**: 3-5% decode for KV-shared models (E2B: 20 shared layers, E4B: 18 shared layers).

#### 6. Symbolic Sliding Window Mask

Currently materializes a full N x N boolean array for sliding window attention:
```swift
var mask = linds .>= rinds  // Full materialization
if let windowSize {
    mask = mask & (linds .< rinds + windowSize)  // Another full array
}
```

For 4K context with 28 sliding layers: 16M elements per mask, ~130MB memory traffic per pass.

**Implementation**: Add a `.slidingWindow(size: Int)` case to `ScaledDotProductAttentionMaskMode`
that the SDPA kernel handles symbolically (like `.causal`). Requires MLX framework change.

**Expected**: ~15% overall speedup from reduced memory bandwidth. Larger impact at longer contexts.

#### 7. Steel Attention Kernels for head_dim=256/512

MLX's Steel attention kernels currently only support head_dim = 64, 80, 128. Gemma 4 uses
head_dim=256 (sliding) and head_dim=512 (global). Falls back to SDPA vector kernels which
are less efficient for longer sequences.

**Implementation**: Add Steel kernel instantiations for head_dim=256 and head_dim=512 in
`steel_attention.h`. Requires tuning tile sizes (BM, BN, BK) for these larger dimensions.
The block size configuration in `scaled_dot_product_attention.cpp:34-36` uses `bk=16` for
head_dim >= 128, which may be too small for 256/512.

**Expected**: 15-25% faster attention on sequences > 256 tokens. Larger models (31B with
32 heads x 512 dim) benefit most.

---

### Tier 3: Custom Metal Kernel Work

#### 8. Fused Q-Norm + RoPE Kernel

Combine RMSNorm and RoPE into a single Metal dispatch. Currently 4 separate dispatches per
layer (norm_q, norm_k, rope_q, rope_k). Fused version: 2 dispatches (fused_norm_rope_q,
fused_norm_rope_k).

**Implementation**: New Metal kernel that loads Q/K, applies RMS normalization in-register,
then applies rotary embedding. Write to output once. For partial RoPE (global attention with
128 of 512 dims rotated), the fused kernel handles the passthrough dims efficiently.

**Expected**: 8-12% decode latency reduction (saves 84+ dispatches across 42 layers).

#### 9. Partial RoPE Optimization

When `dimensions < head_dim` (e.g., RoPE on 128 of 512 dims for global attention), the
current MLX implementation copies the FULL input array, then only rotates the first
`dimensions` elements. The copy of the non-rotated 384 dims is wasted bandwidth.

**Implementation**: In `rope.cpp`, use in-place rotation for the first `dimensions` elements
and skip the initial full copy. Requires Metal kernel modification to handle the split.

**Expected**: 5-10% for full-attention layers (only 5-7 layers per model, but 512-dim heads
make each one expensive).

#### 10. Circular KV Cache (Eliminate Physical Rotation)

Replace RotatingKVCache's physical array reordering (`temporalOrder()`, `trim()`, `concat()`)
with logical circular buffer indexing. The cache stays in fixed memory positions; only an index
pointer advances. Attention kernels use modular arithmetic for position lookup.

**Implementation**: Major refactor of RotatingKVCache + attention mask generation. The Steel
attention kernel would need a `circular_offset` parameter. All mask creation code would use
`(cache_offset + relative_position) % maxSize` instead of physical reordering.

**Expected**: 30% KV cache update latency reduction. Eliminates O(maxSize) concatenation per
token. Most impactful for long-context generation where cache is constantly rotating.

---

## Implementation Priority

For maximum immediate impact on Gemma 4 benchmarks:

1. **Prefill chunk size** (#1) — trivial, 15-25% TTFT
2. **FusedGateUpSwitchGLU for 26B** (#3) — proven pattern, 10-15% decode
3. **v_norm replacement** (#2) — simple, 2-4% decode
4. **peek() caching** (#5) — small change, 3-5% for E2B/E4B

For framework-level improvements benefiting ALL models:

5. **Symbolic sliding window mask** (#6) — 15% across the board
6. **Steel attention for large head dims** (#7) — 15-25% attention
7. **Fused norm+rope** (#8) — 8-12% decode
8. **Circular KV cache** (#10) — 30% cache update

## Notes

- Gemma 4's pure-attention architecture makes it an ideal target for dispatch reduction
  optimizations since there's no sequential bottleneck (GatedDeltaNet) to mask the gains.
- The 31B model (60 layers, head_dim=512) would benefit most from Steel attention and
  partial RoPE optimizations.
- The E2B/E4B models (PLE + KV sharing) benefit from peek() caching and PLE optimization.
- The 26B MoE model benefits from FusedGateUpSwitchGLU and sort threshold tuning.
- All models benefit from prefill chunk size, symbolic masks, and fused norm+rope.
