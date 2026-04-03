# SIMD Analysis & Remaining Performance Gains

**Date**: 2026-04-02
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Model**: Qwen3.5-2B 8-bit (28 layers, 24 QHeads, 4 KVHeads, D=128)

## Question

Can we speed up processing by increasing the SIMD count in our Metal kernels? What gains remain in the architecture?

## Current SIMD Layout

| Kernel | Grid | Threadgroup | SIMD Groups Created | What Each SIMD Group Does |
|--------|------|-------------|---------------------|--------------------------|
| **Score** | (32, totalQ, T) | (32,1,1) | totalQ × T | 1 dot product: 32 lanes × ceil(D/32) dims, `simd_sum` reduces |
| **Value** | (32, totalHeads, ceil(D/32)) | (32,1,1) | totalHeads × 4 | 1 (head, dim_block): each lane handles 1 dim, loops over T tokens |
| **Flash Pass 1** | (32, totalQ, numBlocks) | (32,1,1) | totalQ × numBlocks | 1 (query, block): 32 lanes × 4 dims, loops BlockSize tokens with online softmax |
| **Flash Pass 2** | (32, totalQ, 1) | (32,1,1) | totalQ | 1 query: loops over blocks to merge partial (m, l, o) states |
| **Fused Encode** | (D, numRows, 1) | (D,1,1) | numRows × ceil(D/32) | 1 vector: norm + WHT rotate + quantize + pack + norm correct |

Example at T=4096, B=64: Flash Pass 1 creates 64 blocks × 24 queries = **1536 SIMD groups**. M1 Max has ~128 execution units, each running one SIMD group — reasonable saturation.

## Why Increasing SIMD Count Won't Help

### The Metal SIMD Constraint

Metal's SIMD group is fixed at **32 lanes**. `simd_sum()` — used for every dot product reduction — only works within a single 32-lane group. To use more threads cooperatively requires **threadgroup** shared memory + `threadgroup_barrier()`, adding synchronization overhead.

### Options Considered

**Option A — Smaller blocks (B=32):** Already tested in block size sweep. B=32 creates 2x more SIMD groups but was slower than B=64 at most token counts. Extra dispatch overhead and reduced per-group amortization outweigh the parallelism gain.

**Option B — Multiple SIMD groups per block:** Use 2 groups per (query, block), each handling 32 tokens, merge online softmax states via threadgroup shared memory. This is essentially what the two-pass architecture already does at a coarser granularity. With only B=64 tokens per block, the sync overhead (barriers, shared memory loads) exceeds the benefit of halving the serial loop.

**Option C — More lanes per dot product (64 threads for D=128):** `simd_sum` only works within 32 lanes. Would need two `simd_sum` calls + threadgroup reduction. Slower than current approach where 32 lanes handle 4 dims each with one `simd_sum`.

### The Real Bottleneck

| Component | % of Decode Time | Amenable to SIMD Changes? |
|-----------|:---:|:---:|
| **FFN (feed-forward)** | 50-60% | No — MLX matmul, already optimized |
| **Attention (our kernels)** | 15-25% | Already well-parallelized |
| **Rotation matmuls** | ~5% | No — MLX matmul |
| **Layer norms, embeddings, sampling** | 15-25% | No — MLX ops |

Even if attention kernels were infinitely fast (0ms), maximum possible speedup is ~20-25%. FFN dominates decode time and uses MLX's optimized matmul that we can't improve.

## Remaining Gains

### Gain 1: Rotation Fusion (~2-5% total decode improvement)

**Status**: Not yet implemented
**Effort**: MEDIUM

Fuse query pre-rotation into flash pass 1 and output inverse-rotation into flash pass 2. Eliminates 2 MLX matmul kernel dispatches per layer per step (56 dispatches total across 28 layers).

Implementation: Load rotation matrix row into threadgroup shared memory (128 floats = 512 bytes), apply in-register before/after main computation. Same FLOPs but eliminates dispatch overhead.

Current:
```
3. Pre-rotate query  [MLX matmul: q × Π_key^T]     ← eliminate
4. Flash Pass 1      [Metal: per-block partial attention]
5. Flash Pass 2      [Metal: cross-block reduction]
6. Inverse rotate    [MLX matmul: rot_out × Π_val]  ← eliminate
```

After:
```
3. Flash Pass 1      [Metal: rotate query in-register + partial attention]
4. Flash Pass 2      [Metal: cross-block reduction + inverse rotate output]
```

### Gain 2: Float16 V Accumulation (~5-10% attention kernel improvement)

**Status**: Not yet implemented
**Effort**: LOW

Kernels currently accumulate everything in float32. The V accumulation (`o[i]` in flash pass 1) can safely use `half` (float16):
- Halves register pressure for the V accumulator
- Potentially doubles V accumulation throughput
- Score computation stays float32 for numerical stability (softmax is sensitive)
- Online softmax state (m, l) stays float32

Trade-off: Slight precision loss in V accumulation, but this is post-softmax weighted sum — much less sensitive than the score computation. Need to validate max diff vs float32 accumulation stays acceptable (< 1e-3).

### Gain 3: Prefill Flash Attention (faster TTFT for long prompts)

**Status**: Not yet implemented
**Effort**: MEDIUM-HIGH

Currently L>1 prefill uses separated score → softmax → value kernels which materialize the full [nQHeads, L, T] score matrix. A causal flash attention kernel for L>1 would:
- Eliminate the intermediate score matrix (saves memory at long contexts)
- Reduce 3 dispatches to 2 (pass 1 + pass 2 with causal masking)

Implementation: Extend flash pass 1 with per-query causal masking. Each query position q only attends to tokens where `t <= q + offset`. Add a `causal_offset` parameter so the kernel can mask correctly.

Lower priority than gains 1-2 because prefill only happens once per generation and is already fast.

### Gain 4: Speculative Decoding (~2-3x total improvement)

**Status**: Not implemented (separate feature)
**Effort**: HIGH

Generate N candidate tokens with a small draft model, verify with the large model in one forward pass. Orthogonal to KV cache compression but multiplicative with it. This is the single biggest remaining win but is a fundamentally different feature, not a kernel optimization.

### Gain 5: Long-Context Memory Bandwidth

At 32K+ tokens, kernels become memory-bandwidth-bound (reading all packed K/V). The sparse V threshold (`w < 1e-6f`) helps for values, but score computation reads all K tokens. No algorithmic fix — attention is fundamentally O(T) per query. Hardware-limited.

## Decision

Gains 1-3 are implementable within the current architecture. Expected combined impact: ~5-10% total decode improvement, primarily from eliminating dispatch overhead (rotation fusion) and reducing register pressure (float16 accumulators).

The SIMD configuration is already near-optimal. We are in diminishing returns territory for attention kernel optimization — the 2x+ gains from compressed-domain attention and TurboFlashAttention have already been captured. Remaining wins are incremental.
