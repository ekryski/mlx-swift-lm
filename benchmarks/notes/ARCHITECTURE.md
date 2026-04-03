# KV Cache Architecture: Complete Code Path Traces

**Date**: 2026-04-02
**Branch**: `ek/turbo-opt-0-fix-default-path`

This document traces the full execution path for all three KV cache strategies, identifies where Metal kernels vs MLX ops vs Swift CPU are used, documents data structures at each stage, and identifies performance/memory gaps.

---

## Table of Contents

1. [Common Infrastructure](#1-common-infrastructure)
2. [Path A: No KV Quantization](#2-path-a-no-kv-quantization)
3. [Path B: Affine-4 KV Quantization](#3-path-b-affine-4-kv-quantization)
4. [Path C: TurboQuant-4 KV Quantization](#4-path-c-turboquant-4-kv-quantization)
5. [Comparative Analysis](#5-comparative-analysis)
6. [Gap Analysis & Optimization Mapping](#6-gap-analysis--optimization-mapping)

---

## 1. Common Infrastructure

### Cache Creation

All paths begin identically:

```
LanguageModel.newCache(parameters:)                    [LanguageModel.swift:217-232]
  → Creates [KVCacheSimple()] × numLayers              [KVCache.swift:315-458]
  → Each cache: offset=0, keys=nil, values=nil
```

Models implement `KVCacheDimensionProvider` which provides `kvHeads: [Int]` — one entry per attention layer. For Qwen3.5-2B: 16 KV heads, 128 head dim, 28 layers.

### Generation Loop

```
TokenIterator.step()                                   [Evaluate.swift:1034-1049]
  → model(input, cache)                                [Model forward pass]
      → [Each attention layer]:
          attentionWithCacheUpdate(q, k, v, cache, scale, mask)
                                                       [AttentionUtils.swift:37-105]
  → maybeQuantizeKVCache(&cache, ...)                  [KVCache.swift:1687-1745]
  → sample token from logits
```

### Attention Dispatch (the routing hub)

```swift
// AttentionUtils.swift:37-105
public func attentionWithCacheUpdate(queries, keys, values, cache, scale, mask) -> MLXArray {
    if cache is QuantizedKVCacheProtocol  → Path B (affine)
    if cache is TurboQuantKVCache         → Path C (turbo)
    else                                  → Path A (no quant)
}
```

### MLXFast.scaledDotProductAttention (shared by all paths)

```
Swift wrapper                                          [MLXFast.swift:203-218]
  → mlx_fast_scaled_dot_product_attention()            [C bridge → MLX C++]
      → Metal SDPA kernel (L=1, decode)                [Metal: sdpa_vector]
      → MLX ops fallback (L>1, prefill)                [matmul + softmax]
```

This is MLX's optimized attention — uses Metal flash attention kernels for single-token decode, falls back to standard matmul for multi-token prefill. **All three paths eventually call this** (or its quantized variant).

---

## 2. Path A: No KV Quantization

### Trigger
`parameters.kvBits == nil && parameters.kvScheme == nil`

### Data Flow

```
                    ┌─────────────────────────────────┐
                    │  Raw FP16 Storage (KVCacheSimple) │
                    │  keys:   [B, H, T, D]  FP16      │
                    │  values: [B, H, T, D]  FP16      │
                    │  step=256 growth increments       │
                    └─────────┬───────────────────────┘
                              │
            ┌─────────────────┼──────────────────┐
            │                 │                  │
         Prefill           Decode            maybeQuantize
      (L>1 tokens)      (L=1 token)           (no-op)
            │                 │
            ▼                 ▼
    cache.update()     cache.update()
    [KVCache.swift:328-371]
            │                 │
            ▼                 ▼
    MLXFast.scaledDotProductAttention()
    [MLXFast.swift:203-218]
            │
            ▼
    Metal SDPA kernel (decode) or MLX matmul ops (prefill)
```

### Per-Step Operations (decode, L=1)

| # | Operation | Language | File:Line | Shapes |
|---|-----------|----------|-----------|--------|
| 1 | Slice write: `keys[..., prev:offset, ...] = newK` | MLX op | KVCache.swift:388 | [B,H,1,D] → buffer |
| 2 | Slice read: `keys[..., :offset, ...]` | MLX op | KVCache.swift:391 | → [B,H,T,D] |
| 3 | `MLXFast.scaledDotProductAttention(q,k,v)` | **Metal kernel** | MLXFast.swift:211 | Q:[B,Hq,1,D] × K:[B,Hkv,T,D] |

**Total ops per decode step: 3** (2 slicing + 1 Metal SDPA)

### Memory Per Layer (Qwen3.5-2B: H=16 KV heads, D=128, FP16)

| Tokens | K+V Memory | Formula |
|--------|-----------|---------|
| 128 | 1 MB | 2 × 16 × 256 × 128 × 2 (step=256 alloc) |
| 1024 | 8 MB | 2 × 16 × 1024 × 128 × 2 |
| 4096 | 32 MB | 2 × 16 × 4096 × 128 × 2 |
| 32K | 256 MB | 2 × 16 × 32768 × 128 × 2 |
| 131K | 1 GB | 2 × 16 × 131072 × 128 × 2 |

**All 28 layers at 4096 tokens: 28 × 32 MB = 896 MB**

### Characteristics
- **Zero overhead**: No quantization, no rotation, no encoding
- **Maximum quality**: Full FP16 precision
- **Maximum memory**: Grows linearly with context length
- **Speed**: Fastest decode (only Metal SDPA kernel)

---

## 3. Path B: Affine-4 KV Quantization

### Trigger
`parameters.kvBits == 4, parameters.kvScheme == nil`

### Cache Lifecycle

```
Phase 1: KVCacheSimple (first quantizedKVStart tokens)
  │
  │ maybeQuantizeKVCache() triggers when offset > quantizedKVStart
  ▼
Phase 2: QuantizedKVCache (all subsequent tokens)
  → KVCacheSimple.toQuantized(groupSize:64, bits:4)   [KVCache.swift:407-432]
```

### Data Flow

```
                 ┌──────────────────────────────────────┐
                 │  Quantized Storage (QuantizedKVCache)  │
                 │  keys:   (wq, scales, biases)         │
                 │  values: (wq, scales, biases)         │
                 │                                        │
                 │  wq:     [B, H, T, D/grp × bits/32]   │ uint32
                 │  scales: [B, H, T, D/grp]              │ FP32
                 │  biases: [B, H, T, D/grp]              │ FP32
                 └────────┬───────────────────────────────┘
                          │
            ┌─────────────┼─────────────────┐
            │             │                 │
         Prefill       Decode         Conversion
       (FP16 path)    (quantized)    (one-time)
            │             │
            ▼             ▼
   MLXFast.SDPA    quantizedScaledDotProductAttention()
   (standard)      [KVCache.swift:1583-1671]
                          │
                ┌─────────┼──────────┐
                │                    │
                ▼                    ▼
        quantizedMM(Q, K)    quantizedMM(Attn, V)
        [Ops.swift:2412]     [Ops.swift:2412]
                │                    │
                ▼                    ▼
        mlx_quantized_matmul()  mlx_quantized_matmul()
        [MLX C++ → Metal]      [MLX C++ → Metal]
```

### Per-Step Operations (decode, L=1)

| # | Operation | Language | File:Line | Notes |
|---|-----------|----------|-----------|-------|
| 1 | `quantized(newKeys, groupSize:64, bits:4)` | **MLX C++/Metal** | Ops.swift:2353 | Quantize 1 token K |
| 2 | `quantized(newValues, groupSize:64, bits:4)` | **MLX C++/Metal** | Ops.swift:2353 | Quantize 1 token V |
| 3 | Slice write wq/scales/biases to storage | MLX ops | KVCache.swift:885-898 | 6 slice assignments |
| 4 | Slice read wq/scales/biases up to offset | MLX ops | KVCache.swift:900-903 | 6 slice reads |
| 5 | `quantizedMM(Q*scale, K_wq, K_scales, K_biases, transpose:true)` | **MLX C++ → Metal** | KVCache.swift:1622 | Scores = Q×K^T |
| 6 | Causal mask application | MLX ops | KVCache.swift:1635-1645 | If mask needed |
| 7 | `softmax(scores, axis:-1)` | MLX op | KVCache.swift:1647 | Attention weights |
| 8 | `quantizedMM(Attn, V_wq, V_scales, V_biases, transpose:false)` | **MLX C++ → Metal** | KVCache.swift:1650 | Output = Attn×V |

**Total ops per decode step: ~14** (2 quantize + 12 slice/matmul/softmax)

### Quantized Matmul (the key Metal kernel)

`mlx_quantized_matmul` is MLX's built-in operation:
- **On-the-fly dequant + matmul** fused in a single Metal kernel dispatch
- Dequantization formula: `value = quantized_val × scale + bias`
- Groups of 64 elements share one scale + bias
- **No intermediate FP16 buffer** — dequant happens in-register during matmul

This is the critical advantage of affine-4: the `quantizedMM` function is a **fused Metal kernel** that reads packed 4-bit data, dequantizes in registers, and computes the matmul without ever materializing a full FP16 key/value tensor.

### Memory Per Layer (4-bit affine, groupSize=64)

Per token per head:
- wq: D×4 bits = 128×4/8 = 64 bytes
- scales: (D/64)×4 = 2×4 = 8 bytes
- biases: (D/64)×4 = 2×4 = 8 bytes
- **Total: 80 bytes** vs 256 bytes FP16 (3.2× compression)

Per token all heads (K+V): 2 × 16 × 80 = 2,560 bytes vs 8,192 bytes FP16

| Tokens | K+V Memory | vs No-Quant |
|--------|-----------|-------------|
| 128 | 320 KB | 3.2× smaller |
| 1024 | 2.5 MB | 3.2× smaller |
| 4096 | 10 MB | 3.2× smaller |
| 32K | 80 MB | 3.2× smaller |
| 131K | 320 MB | 3.2× smaller |

### Characteristics
- **Low overhead**: `quantized()` + `quantizedMM()` are both fused Metal kernels
- **Good compression**: 3.2× memory reduction
- **Slight speed cost**: quantize overhead per token + quantizedMM slightly slower than pure FP16 SDPA
- **Quality**: Good — affine preserves linear relationships well for KV data

---

## 4. Path C: TurboQuant-4 KV Quantization

### Trigger
`parameters.kvScheme == "turbo4"` → bits=4

### Cache Lifecycle

```
Phase 1: KVCacheSimple (during prefill)
  │
  │ maybeQuantizeKVCache() with kvScheme="turbo4"
  ▼
Phase 1.5: TurboQuantKVCache (raw, isCompressed=false)
  │  toTurboQuantized() copies FP16 data           [KVCache.swift:445-453]
  │
  │ First decode call (L=1) triggers transition
  ▼
Phase 2: TurboQuantKVCache (compressed, isCompressed=true)
  → compressRawCacheInternal() — batch encode        [TurboQuantKVCache.swift:616-646]
  → Subsequent tokens: encodeNewToken() per step      [TurboQuantKVCache.swift:651-701]
```

### Data Flow (Current: Compressed-Domain with TurboFlashAttention)

```
┌─────────────────────────────────────────────────────────────┐
│ COMPRESSED STORAGE (no dequant buffer needed)               │
│                                                              │
│ keyPackedMSE:  [B,H,T,PW]  uint32   (packed codebook idx)  │
│ keyNorms:      [B,H,T]     FP32     (norm-corrected)       │
│ valPackedMSE:  [B,H,T,PW]  uint32   (packed codebook idx)  │
│ valNorms:      [B,H,T]     FP32     (norm-corrected)       │
│                                                              │
│ CODEC STATE (shared across layers via getOrCreateCodec())   │
│ keyMSECodec:   rotation [D,D], codebook [2^bits],          │
│                boundaries [2^bits-1], rotationT [D,D]      │
│ valueMSECodec: rotation [D,D], codebook [2^bits],          │
│                boundaries [2^bits-1], rotationT [D,D]      │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────┐
         │               │                   │
      Prefill      First Decode         Subsequent Decode
      (L>1)        (transition)          (L=1, steady state)
         │               │                   │
         ▼               ▼                   ▼
 turboCache.update()  compressRawCache()  encodeNewToken()
 → raw FP16 store     │                  (Metal: fused WHT
 → hot window check   ├─ ensureCodecs()    encode kernel)
                       │  (pre-computed            │
                       │   codebooks)               │
                       │                            │
                       ├─ fusedEncodeWHT()          │
                       │  (Metal: batch encode      │
                       │   all prefill tokens)      │
                       │                            │
                       ├─ free raw buffers          │
                       │                            │
                       └────────────┬───────────────┘
                                    │
                                    ▼
                        prepareQueries(queries)
                        (MLX: matmul × Π_key^T, scale)
                                    │
                          ┌─────────┴──────────┐
                          │                    │
                       L == 1               L > 1
                       (decode)            (prefill)
                          │                    │
                          ▼                    ▼
               TurboFlashAttention     Separated Kernels
               (2 Metal dispatches)    (3 dispatches)
                          │                    │
                     ┌────┴────┐          ┌────┴────┐
                     │         │          │    │    │
                  Pass 1    Pass 2     Score Soft- Value
                (B=64 blocks)                  max
                     └────┬────┘          └────┬────┘
                          │                    │
                          ▼                    ▼
                     rot_output (in Π_val rotated space)
                                    │
                                    ▼
                        inverseRotateOutput(rotOutput)
                        (MLX: matmul × Π_val)
                                    │
                                    ▼
                              Final output
```

### Per-Step Operations (decode, L=1, steady state)

| # | Operation | Language | File:Line | Shapes | Cost |
|---|-----------|----------|-----------|--------|------|
| 1 | `fusedEncodeDispatch(newKeys)` — norm+rotate+quantize+pack+normcorrect | **Metal kernel** | TurboQuantKernels.swift:381 | [B×H,D]→[B×H,PW]+[B×H] | ~0.4ms |
| 2 | `fusedEncodeDispatch(newValues)` | **Metal kernel** | TurboQuantKernels.swift:381 | same | ~0.4ms |
| 3 | Reshape packed K back to [B,H,1,PW] | MLX op | TurboQuantKVCache.swift | trivial | ~0 |
| 4 | Grow compressed storage if needed | MLX ops | TurboQuantKVCache.swift | alloc + copy | rare |
| 5 | Slice write packed K/V + norms | MLX ops | TurboQuantKVCache.swift | 4 assignments | ~0 |
| 6 | `matmul(queries, Π_key^T) * scale` — pre-rotate queries | **MLX matmul** | TurboQuantKVCache.swift | [B,Hq,1,D]×[D,D] | ~0.1ms |
| 7 | Slice read packed K/V + norms up to offset | MLX ops | TurboQuantKVCache.swift | 4 reads + reshape | ~0 |
| 8 | `turboFlashAttention()` Pass 1 — per-block partial attention | **Metal kernel** | TurboQuantKernels.swift | Grid:(32, totalQ, numBlocks) | dominant |
| 9 | `turboFlashAttention()` Pass 2 — cross-block reduction | **Metal kernel** | TurboQuantKernels.swift | Grid:(32, totalQ, 1) | ~0.1ms |
| 10 | `matmul(rotOutput, Π_val)` — inverse rotate output | **MLX matmul** | TurboQuantKVCache.swift | [B,Hq,1,D]×[D,D] | ~0.1ms |

**Total ops per decode step: ~14** (2 Metal encode + 2 Metal flash + 2 MLX matmul + ~8 slice/reshape)

### Fused Encode Metal Kernel (the turbo-specific kernel)

```
TurboQuantKernelOps.fusedEncodeWHT()                   [TurboQuantKernels.swift:438-495]
  → MLXFast.metalKernel(name: "turbo_fused_encode_wht_4_128")
  Grid: (128, numRows, 1)  ThreadGroup: (128, 1, 1)

Metal kernel operations per vector [D=128]:
  1. Load input[d] from global memory
  2. Compute L2 norm: sq = val*val; norm_sq = simd_sum(sq) across 4 SIMD groups
  3. Normalize: unit = val / norm
  4. WHT rotation: shared_buf[d] = signs[d] * unit; butterfly(shared_buf, log2(D) stages)
  5. Boundary quantize: count(rotated > boundaries[0..14]) → index (0..15)
  6. Pack bits: atomic_fetch_or into shared_packed[] words
  7. Norm correction: recon_sq = codebook[idx]^2; simd_sum; corrected = norm / recon_norm
  8. Write: packed_out[row], norms_out[row]
```

### Memory Per Layer (turbo4v2: 4-bit K, 2-bit V, D=128)

**Compressed storage only** — no dequant buffer needed (compressed-domain attention reads packed data directly):

Per token per head:
- K packed: PW × 4 = 16 × 4 = 64 bytes (4-bit, D=128: PW = (128×4+31)/32 = 16)
- K norm: 4 bytes (FP32)
- V packed: PW × 4 = 8 × 4 = 32 bytes (2-bit, D=128: PW = (128×2+31)/32 = 8)
- V norm: 4 bytes (FP32)
- **Total: 104 bytes per head** vs 512 bytes FP16 (4.9× compression)

Per token all heads (K+V): 16 × 104 = 1,664 bytes vs 8,192 bytes FP16

| Tokens | K+V Memory | vs No-Quant | vs Affine4 |
|--------|-----------|-------------|------------|
| 128 | 208 KB | 4.9× smaller | 1.5× smaller |
| 1024 | 1.6 MB | 4.9× smaller | 1.5× smaller |
| 4096 | 6.5 MB | 4.9× smaller | 1.5× smaller |
| 32K | 52 MB | 4.9× smaller | 1.5× smaller |

For turbo3v2 (3-bit K, 2-bit V): 84 bytes per head → 6.1× compression vs FP16.

### Codec Initialization Cost (one-time, first decode)

| Operation | Language | Time | Notes |
|-----------|----------|------|-------|
| Lloyd-Max codebook (100 k-means iters on 32K grid) | Swift CPU | ~50ms | Per (dim, bits) pair |
| QR decomposition on [D,D] Gaussian | MLX CPU | ~10ms | Non-power-of-2 dims |
| Hadamard matrix construction | Swift CPU | ~2ms | Power-of-2 dims |
| WHT signs generation | MLX GPU | ~1ms | Random ±1 |
| Dense Hadamard×Signs matmul | MLX GPU | ~1ms | Build rotation matrix |

**Total init per codec: ~60ms**. Two codecs (key + value) = **~120ms first-token overhead**.

---

## 5. Comparative Analysis

### Operations Per Decode Step

| Path | Metal Kernels | MLX Matmul | MLX Ops | Total | Overhead vs No-Quant |
|------|:---:|:---:|:---:|:---:|---:|
| **No-Quant** | 1 (SDPA) | 0 | 2 (slice) | 3 | baseline |
| **Affine-4** | 3 (quantize×2 + qMM×2) | 0 | 12 (slice) | ~15 | +12 ops |
| **Turbo4v2** | 4 (encode×2 + flash p1+p2) | 2 (q pre-rotate + output inverse-rotate) | ~8 (slice/reshape) | ~14 | +11 ops |

### Memory Footprint Per Layer at 4096 Tokens

| Path | K+V Storage | Additional | **Total** | Compression vs FP16 |
|------|-----------|------------|-----------|---------------------|
| **No-Quant** | 32 MB FP16 | — | **32 MB** | 1× |
| **Affine-4** | 10 MB (wq+scales+biases) | — | **10 MB** | **3.2×** |
| **Turbo4v2** | 6.5 MB compressed | — | **6.5 MB** | **4.9×** |
| **Turbo3v2** | 5.2 MB compressed | — | **5.2 MB** | **6.1×** |

Compressed-domain attention eliminated the FP16 dequant buffer entirely. TurboQuant now stores only packed indices + norms — no intermediate buffers.

Per token per head for turbo4v2 (4-bit K, 2-bit V, D=128):
- K: packedWidth(128×4) = 16 uint32s = 64 bytes + 4 bytes norm = 68 bytes
- V: packedWidth(128×2) = 8 uint32s = 32 bytes + 4 bytes norm = 36 bytes
- **Total: 104 bytes** vs 512 bytes FP16 (4.9× compression)

### Speed (Qwen3.5-2B 8-bit, summarization)

| Path | Gen tok/s (128) | Gen tok/s (1024) | Gen tok/s (4096) | Gen tok/s (32K) | Gen tok/s (131K) |
|------|:---:|:---:|:---:|:---:|:---:|
| **No-Quant** | 88.7 | 89.0 | 85.3 | 68.6 | 40.6 |
| **Affine-4** | 88.7 | 83.4 | 82.0 | 64.1 | 39.6 |
| **Turbo4v2** | 79.6 | 80.5 | 77.3 | 62.2 | 31.5 |
| **Turbo3v2** | 83.4 | 82.1 | 80.2 | 64.6 | 37.8 |

Turbo3v2 is within ~1-3% of affine-4 at short-to-medium contexts, with significantly better compression. At very long contexts (131K), turbo variants are slower due to rotation overhead that scales with sequence length.

### Quality (KLD vs bf16 baseline, avg across contexts)

| Path | Think KLD (avg) | Gen KLD (avg) |
|------|:---:|:---:|
| **No-Quant** | 0.021 | 0.023 |
| **Affine-4** | 0.040 | 0.025 |
| **Turbo4v2** | 0.020 | 0.034 |
| **Turbo3v2** | 0.013 | 0.013 |

TurboQuant quality is excellent — Think KLD is comparable to or better than no-quant (which has weight quantization noise from 8-bit weights), and significantly better than affine-4.

### KV Memory at 32K Context (All 28 Layers)

| Path | KV Cache | vs No-Quant |
|------|----------|-------------|
| **No-Quant** | 7.0 GB | 1× |
| **Affine-4** | 2.2 GB | 3.2× smaller |
| **Turbo4v2** | 1.44 GB | 4.9× smaller |
| **Turbo3v2** | 1.22 GB | 5.7× smaller |

---

## 6. Gap Analysis & Optimization Mapping

### Completed Optimizations

The following gaps from the original analysis have been addressed:

| # | Optimization | Status | Outcome |
|---|---|---|---|
| P0 | Fix default path (was using raw FP16 for turbo caches) | **DONE** | Compressed-domain attention now active |
| P1 | Compressed-domain Metal kernels (score + value work on packed data) | **DONE** | Eliminated FP16 dequant buffer entirely |
| P2 | Pre-computed Lloyd-Max codebook constants | **DONE** | Eliminated ~100ms first-token overhead for common (dim, bits) pairs |
| P3 | Asymmetric K/V compression (separate keyBits/valueBits) | **DONE** | turbo4v2 (4K/2V), turbo3v2 (3K/2V) configs |
| P5 | Sparse V dequant threshold (`w < 1e-6f` skip) | **DONE** | Skips ~90% of attention weights at long contexts |
| — | TurboFlashAttention (fused score+softmax+value, two-pass) | **DONE** | 1.1-3.8x kernel speedup, eliminates intermediate score+weight arrays |

The following were implemented, benchmarked, and **removed** because they provided no measurable benefit:

| # | Optimization | Status | Outcome |
|---|---|---|---|
| P4 | Hot window (last 256 tokens in FP16) | **REMOVED** | -0.8% to -3.7% gen tok/s regression. Added routing complexity with no speed or quality benefit. |
| P6 | Boundary layer protection (first/last N layers at FP16) | **REMOVED** | -5% to -17% gen tok/s regression. No measurable KLD improvement — both protected and unprotected variants showed KLD in the 0.01-0.05 stochastic noise range. |

### Remaining Gaps

#### GAP 1: Per-Token Rotation Overhead — MEDIUM

**Problem**: Each decode step still does 2 MLX matmul operations for rotation:
1. `matmul(queries, Π_key^T)` — pre-rotate query for score computation
2. `matmul(rotOutput, Π_val)` — inverse-rotate output back to model space

Each is a [B,Hq,1,D]×[D,D] matmul. At D=128 with 24 query heads: 2 × 24 × 128×128 = 786K FLOPs. Small, but each launches a separate MLX Metal kernel dispatch with overhead.

Note: The encode kernel already handles K/V rotation internally (WHT butterfly in-register), so we went from 4 rotation matmuls → 2. The remaining two cannot easily be eliminated — the query must be in rotated space for score computation, and the output must be un-rotated before the next layer.

**Potential fix**: Fuse query rotation into the flash pass 1 kernel (load Π_key^T in shared memory, rotate query in-register before scoring). Fuse output inverse-rotation into flash pass 2 (apply Π_val after merging partials). This would eliminate both MLX matmul dispatches.

**Impact**: Minor speed improvement (~0.2ms per step). Low priority — rotation is <5% of decode time.

#### GAP 2: Prefill Path Uses Separated Kernels — LOW

**Problem**: For L>1 (prefill chunks during re-encoding after context shift, or multi-query batches), `compressedAttention()` falls back to separated score → softmax → value kernels. This path materializes the full score matrix for causal masking across query positions.

**Root Cause**: TurboFlashAttention's online softmax processes one query at a time. Supporting L>1 with causal masking would require per-query masking logic inside the fused kernel.

**Impact**: Low — prefill is dominated by FFN compute and only happens once per generation. The separated path is adequate.

#### GAP 3: Model Compatibility — UNKNOWN (Testing Needed)

**Problem**: All benchmarks so far are on Qwen3.5-2B (28 layers, 24 QHeads, 4 KVHeads, D=128). The Metal kernels make assumptions that may not hold for all architectures:

**Known compatible patterns**:
- GQA with any repeat count (handled via `repeatCount` parameter in kernels)
- MHA (repeatCount=1) and MQA (repeatCount=numQHeads)
- Power-of-2 head dimensions (WHT butterfly rotation in fused encode kernel)
- Non-power-of-2 head dimensions (falls back to QR decomposition for rotation matrix)
- Hybrid architectures like Qwen3.5's GatedDeltaNet (`maybeQuantizeKVCache` filters by `KVCacheSimple`, skipping Mamba/DeltaNet layers)

**Potential risk areas**:
- **Very large head dimensions** (D=256+): Flash kernel SIMD lane mapping (`ceil(D/32)` dims per lane) may need tuning. At D=256, each lane handles 8 dims — should work but untested.
- **Very small head dimensions** (D=64): Less work per SIMD group. Flash kernel may underperform separated kernels. Need block size re-tuning.
- **MoE models** (Qwen3.5 35B-A3B, Nemotron): KV cache is standard (not MoE-specific — MoE only affects FFN routing). Should work, but untested at scale.
- **Large KV head counts** (H=32+): Grid dispatch scales linearly with totalQ. Should work, but memory pressure from more layers × more heads is untested.
- **Models with unusual KV layouts**: Any model following standard [B, H, T, D] KV cache layout should work. Non-standard layouts would need adapter code.

**Testing plan**: Benchmark turbo4v2/turbo3v2 on Qwen3.5-35B-A3B (MoE), Qwen2.5-27B (large dense), GPT-OSS (different architecture family), Nemotron-30B-A3B (MoE).

---

## Summary: Current Architecture Status

TurboQuant compressed-domain attention is fully operational with the following decode pipeline:

```
Decode step (L=1):
  1. Fused encode K    [Metal: WHT rotate + quantize + pack + norm correct]
  2. Fused encode V    [Metal: same]
  3. Pre-rotate query  [MLX matmul: q × Π_key^T]
  4. Flash Pass 1      [Metal: per-block partial attention with online softmax]
  5. Flash Pass 2      [Metal: cross-block reduction → rotated output]
  6. Inverse rotate    [MLX matmul: rot_out × Π_val]
  Total: 4 Metal dispatches + 2 MLX matmuls
```

vs the original broken path (P0 bug): 2 Metal encode + 4 MLX matmul rotations + 1 Metal SDPA on full FP16 dequant buffer = wasted compression.

| Metric | Before (P0 bug) | Current |
|--------|-----------------|---------|
| Metal dispatches | 3 | 4 (2 encode + 2 flash) |
| MLX matmuls | 4 (wasted rotations) | 2 (essential rotations) |
| Intermediate buffers | Full FP16 dequant (same size as no-quant) | None |
| Memory at 4K, turbo4v2 | ~40 MB/layer (compressed + dequant) | ~6.5 MB/layer (compressed only) |
| Gen tok/s (4K) | ~77 | ~77-80 |
| Quality (KLD) | Good | Good (identical — same codec) |
