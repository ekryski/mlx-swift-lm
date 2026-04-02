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

### Data Flow (Post-P0 Fix: Dequant-First Path)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ COMPRESSED STORAGE                        DEQUANT BUFFER (ROTATED)     │
│                                                                         │
│ keyPackedMSE:  [B,H,T,PW]  uint32        dequantKeys:  [B,H,T,D] FP16 │
│ keyNorms:      [B,H,T]     FP32          dequantValues:[B,H,T,D] FP16 │
│ valPackedMSE:  [B,H,T,PW]  uint32        (in Π-rotated space)         │
│ valNorms:      [B,H,T]     FP32                                        │
│                                                                         │
│ CODEC STATE (per layer, initialized once)                               │
│ keyMSECodec:   rotation [D,D], codebook [2^bits], boundaries [2^bits-1]│
│ valueMSECodec: rotation [D,D], codebook [2^bits], boundaries [2^bits-1]│
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          │               │                   │
       Prefill      First Decode         Subsequent Decode
       (L>1)        (transition)          (L=1, steady state)
          │               │                   │
          ▼               ▼                   ▼
  turboCache.update()  turboCache.         turboCache.
  → raw FP16 store     updateAndDequant()  updateAndDequant()
  → standard SDPA      │                   │
                        ├─ ensureCodecs()   ├─ encodeNewToken()
                        │  (CPU: codebook,  │  (Metal: fused encode)
                        │   rotation gen)   │
                        │                   ├─ rotate new token
                        ├─ compress raw     │  (MLX: matmul × Π^T)
                        │  (Metal: fused    │
                        │   encode batch)   ├─ append to dequant buffer
                        │                   │  (MLX: slice assignment)
                        ├─ rotate raw → Π   │
                        │  (MLX: matmul)    │
                        │                   │
                        ├─ init dequant buf │
                        │                   │
                        └───────────┬───────┘
                                    │
                                    ▼
                        prepareQueries(queries)
                        (MLX: matmul × Π_key^T)
                                    │
                                    ▼
                        MLXFast.scaledDotProductAttention(
                            rotQueries, dequantKeys, dequantValues)
                        (Metal SDPA kernel — standard FP16)
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
| 3 | Reshape packed K back to [B,H,1,PW] | MLX op | TurboQuantKVCache.swift:673 | trivial | ~0 |
| 4 | Grow compressed storage if needed | MLX ops | TurboQuantKVCache.swift:679-694 | alloc + copy | rare |
| 5 | Slice write packed K/V + norms | MLX ops | TurboQuantKVCache.swift:697-700 | 4 assignments | ~0 |
| 6 | `matmul(newKeys, Π_key^T)` — rotate new K | **MLX matmul** | TurboQuantKVCache.swift:756 | [B,H,1,D]×[D,D] | ~0.1ms |
| 7 | `matmul(newValues, Π_val^T)` — rotate new V | **MLX matmul** | TurboQuantKVCache.swift:757 | [B,H,1,D]×[D,D] | ~0.1ms |
| 8 | Grow dequant buffer if needed | MLX ops | TurboQuantKVCache.swift:760-785 | alloc + copy | rare |
| 9 | Slice write rotated K/V to dequant buffer | MLX ops | TurboQuantKVCache.swift:788-789 | 2 assignments | ~0 |
| 10 | Slice read dequant buffer up to offset | MLX ops | TurboQuantKVCache.swift:791-794 | 2 reads | ~0 |
| 11 | `matmul(queries, Π_key^T)` — pre-rotate queries | **MLX matmul** | TurboQuantKVCache.swift:798 | [B,Hq,1,D]×[D,D] | ~0.1ms |
| 12 | `MLXFast.scaledDotProductAttention(rotQ, rotK, rotV)` | **Metal kernel** | AttentionUtils.swift:83 | standard SDPA | dominant |
| 13 | `matmul(rotOutput, Π_val)` — inverse rotate output | **MLX matmul** | TurboQuantKVCache.swift:806 | [B,Hq,1,D]×[D,D] | ~0.1ms |

**Total ops per decode step: ~17** (2 Metal encode + 4 MLX matmul + 1 Metal SDPA + 10 slice/reshape)

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

### Memory Per Layer (turbo4, D=128)

**Compressed storage** per token per head:
- packed: PW × 4 bytes = 16 × 4 = 64 bytes (for 4-bit, D=128: PW = (128×4+31)/32 = 16)
- norm: 4 bytes (FP32)
- **Total K: 68 bytes**, **V: 68 bytes** = 136 bytes per head

**Dequant buffer** per token per head (the problem!):
- FP16: D × 2 = 256 bytes per head for K, 256 for V = 512 bytes per head

**Combined per token all heads (K+V):**
- Compressed: 2 × 16 × 68 = 2,176 bytes
- Dequant buffer: 2 × 16 × 256 = 8,192 bytes (same as FP16!)
- **TOTAL: 10,368 bytes** — worse than no-quant (8,192) by 26%!

| Tokens | Compressed | Dequant Buffer | **Total** | vs No-Quant | vs Affine4 |
|--------|-----------|---------------|-----------|-------------|------------|
| 128 | 272 KB | 1 MB | **1.3 MB** | 1.3× worse | 4× worse |
| 1024 | 2.2 MB | 8 MB | **10.2 MB** | 1.3× worse | 4× worse |
| 4096 | 8.5 MB | 32 MB | **40.5 MB** | 1.3× worse | 4× worse |
| 32K | 68 MB | 256 MB | **324 MB** | 1.3× worse | 4× worse |

**The dequant buffer dominates.** This is why the P0 benchmark showed KV Delta of 60MB at 4096 tokens — higher than both baselines.

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
| **Turbo4 (P0)** | 3 (encode×2 + SDPA) | 4 (rotate×3 + inverse) | 10 (slice) | ~17 | +14 ops |

### Memory Footprint Per Layer at 4096 Tokens

| Path | K+V Storage | Additional | **Total** | Compression vs FP16 |
|------|-----------|------------|-----------|---------------------|
| **No-Quant** | 32 MB FP16 | — | **32 MB** | 1× |
| **Affine-4** | 10 MB (wq+scales+biases) | — | **10 MB** | **3.2×** |
| **Turbo4 (P0)** | 8.5 MB compressed | 32 MB dequant buffer | **40.5 MB** | **0.8× (worse!)** |
| Turbo4 (compressed-only) | 8.5 MB compressed | — | **8.5 MB** | **3.8×** (theoretical) |

### Speed (from P0 benchmark, Qwen3.5-2B 8-bit)

| Path | Gen tok/s (128) | Gen tok/s (1024) | Gen tok/s (4096) |
|------|:---:|:---:|:---:|
| **No-Quant** | 88.7 | 89.0 | 85.3 |
| **Affine-4** | 88.7 | 83.4 | 82.0 |
| **Turbo4 (P0)** | 78.7 | 77.9 | 77.8 |

### Quality (KLD vs bf16 baseline)

| Path | Think KLD (avg) | Gen KLD (avg) |
|------|:---:|:---:|
| **No-Quant** | 0.021 | 0.023 |
| **Affine-4** | 0.040 | 0.025 |
| **Turbo4 (P0)** | **0.009** | **0.004** |

Turbo4 quality is excellent — significantly better KLD than both no-quant (which has weight quantization noise) and affine-4.

---

## 6. Gap Analysis & Optimization Mapping

### GAP 1: Dual Buffer (Compressed + FP16 Dequant) — CRITICAL

**Problem**: `updateAndDequant()` maintains BOTH compressed packed storage AND a full FP16 dequant buffer in rotated space. The dequant buffer is the same size as no-quant FP16 storage, so turbo4 actually uses **more** memory than no compression.

**Root Cause**: MLX's `scaledDotProductAttention` requires FP16 inputs — we can't pass packed indices directly.

**Fix → P1 (Fused SDPA Dequant Kernel)**: SwiftLM's approach — decompress K/V **inside** the SDPA Metal kernel. Read packed indices, dequant in-register, compute attention. Never materialize FP16. This eliminates the 32MB dequant buffer entirely, achieving the theoretical 3.8× compression.

**Impact**: Memory goes from 40.5 MB → 8.5 MB per layer at 4096 tokens. This is the single highest-impact optimization.

### GAP 2: Per-Token Rotation Overhead — HIGH

**Problem**: Each decode step does 4 MLX matmul operations for rotation:
1. `matmul(newKeys, Π_key^T)` — rotate new key (encode)
2. `matmul(newValues, Π_val^T)` — rotate new value (encode)
3. `matmul(queries, Π_key^T)` — pre-rotate query
4. `matmul(rotOutput, Π_val)` — inverse-rotate output

Each is a [B,H,1,D]×[D,D] matmul. At D=128, that's 4× 128×128 = 65K FLOPs per head per step × 16 heads = 1M FLOPs. Small but latency-bound — each launches a separate Metal kernel.

**Root Cause**: The encode Metal kernel already does rotation internally (step 4 in the fused encode). But then we ALSO rotate the raw token separately for the dequant buffer. We're doing rotation twice.

**Fix → P1 (Fused SDPA Dequant Kernel)**: If SDPA works directly on compressed data, we don't need the dequant buffer, so we don't need to rotate raw tokens separately. The fused encode kernel handles rotation during encoding. For queries, we can pre-rotate once (keeping this single matmul). For output, the fused value kernel outputs in rotated space, requiring one inverse rotation.

**Net effect**: 4 rotation matmuls → 2 (pre-rotate query + inverse-rotate output). The encode kernel already handles the other two internally.

### GAP 3: Codebook Generation at Runtime — MEDIUM

**Problem**: First decode step generates Lloyd-Max codebook via 100-iteration k-means on 32K grid points. This takes ~50ms per codec × 2 codecs = ~100ms one-time cost.

**Root Cause**: Codebook is dimension- and bits-dependent but deterministic. Computing it at runtime is wasteful.

**Fix → P2 (Pre-computed Constants)**: SwiftLM hardcodes 8 centroids for dim=128, 3-bit: `{-0.190685, -0.117832, -0.065717, -0.021460, 0.021460, 0.065717, 0.117832, 0.190685}`. Pre-compute for common (dim, bits) pairs and store as static constants.

**Impact**: Eliminates ~100ms TTFT overhead on first decode step.

### GAP 4: Symmetric K/V Compression — MEDIUM

**Problem**: Both K and V use the same bit-width. Research shows V compression is nearly free while K drives all quality loss.

**Root Cause**: Original design simplicity — `init(bits:)` applies equally.

**Fix → P3 (Asymmetric K/V)**: Separate `keyBits`/`valueBits`. Use 4-bit K + 2-bit V for better compression at same quality. Compressed storage for V shrinks by 50%.

**Impact**: ~25% more memory savings with negligible quality impact.

### GAP 5: No Hot Window — MEDIUM

**Problem**: All tokens get compressed immediately on first decode. Short contexts (< 256 tokens) pay full compression/rotation overhead for no memory benefit.

**Root Cause**: Two-phase design is prefill-then-compress, with no sliding window.

**Fix → P4 (Hot Window)**: Keep last 256 tokens in FP16 rotated space. Only compress tokens that age out of the window. Short contexts never see compression overhead.

**Impact**: Short-context gen tok/s matches no-quant. Long-context gets compression benefits.

### GAP 6: Value Kernel Threshold — LOW (for compressed path only)

**Problem**: In `compressedAttention()` Metal value kernel, `if (w == 0.0f) continue;` — only skips exactly-zero weights.

**Fix → P5 (Sparse V)**: Widen to `if (w < 1e-6f) continue;`. At 32K context, ~90% of attention weights are below this threshold.

**Impact**: +7-22% decode speedup at long contexts, zero quality impact. One-line change. But only applies to compressed-domain attention path, not the current dequant-first path.

### GAP 7: Uniform Layer Compression — LOW

**Problem**: All layers get same compression level. First/last layers are more sensitive.

**Fix → P6 (Boundary Layers)**: Keep first/last 2 layers at FP16 or q8_0, compress middle layers with turbo.

**Impact**: 37-91% quality gap recovery, ~14% more memory for boundary layers.

### GAP 8: Affine-4 Has Fused quantizedMM, Turbo4 Doesn't — KEY INSIGHT

**The fundamental architectural difference**: Affine-4's `quantizedMM` is a **single MLX built-in Metal kernel** that reads packed 4-bit data, dequantizes in-register, and computes the matmul. There is no intermediate buffer. MLX provides this natively.

TurboQuant's MSE-optimal coding with rotation and norm correction is mathematically superior (lower KLD), but our implementation pays for it with separate kernels for: encode → store packed → read packed → dequant → rotate → materialize FP16 → SDPA. This is **6 separate GPU dispatches** vs affine-4's **1 dispatch**.

**This is why affine-4 is faster despite worse quality.** The fused SDPA kernel (P1) must collapse these 6 dispatches into 1-2 to be competitive.

---

## Summary: Priority-Ordered Optimization Impact

| Priority | Optimization | Speed Impact | Memory Impact | Quality Impact | Complexity |
|:---:|---|:---:|:---:|:---:|:---:|
| **P1** | Fused SDPA Dequant Kernel | **+++** (eliminate 4 matmuls + buffer) | **+++** (eliminate dequant buffer) | none | HIGH |
| **P2** | Pre-computed Constants | **+** (eliminate 100ms init) | none | none | LOW |
| **P3** | Asymmetric K/V | none | **++** (25% more compression) | none/better | MEDIUM |
| **P4** | Hot Window | **++** (short ctx) | **+** (adaptive) | **+** (recent tokens full precision) | MEDIUM |
| **P5** | Sparse V Dequant | **+** (long ctx only) | none | none | LOW |
| **P6** | Boundary Layers | none | **-** (slight increase) | **++** (37-91% gap recovery) | LOW |

**P1 is the single most impactful change** — it addresses Gaps 1, 2, and 8 simultaneously by moving from "encode-then-dequant-then-SDPA" to "fused-SDPA-with-inline-dequant", matching the architectural approach that makes both affine-4 and SwiftLM's TurboQuant performant.
