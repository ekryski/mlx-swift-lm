# KV Delta Memory Analysis: Why Turbo Shows Higher Than Affine

**Date**: 2026-04-02
**Branch**: `ek/turbo-opt-0-fix-default-path`

## Question

Why does turbo4 show 267MB KV Delta vs affine4's 98MB at 32K, when turbo4's theoretical compressed size (1927MB) is actually smaller than affine4's (2267MB)?

## Investigation

Ran all turbo configs (turbo3, turbo3v2, turbo4, turbo4v2) and compared.

### Results at 32K Context

| Config | Bits/dim K | Bits/dim V | KV Delta | Theoretical | Ratio | GPU Peak | Gen tok/s | Gen KLD |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| No-Quant | 16 | 16 | 333MB | 7256MB | 4.6% | 5.00GB | 68.6 | 0.007 |
| Affine4 | 4 | 4 | **98MB** | 2267MB | 4.3% | **4.55GB** | 64.1 | 0.757 |
| Turbo4 | 4 | 4 | 267MB | 1927MB | 13.9% | 5.00GB | 63.1 | 0.034 |
| Turbo4v2 | 4 | 2 | 366MB | 1548MB | 23.6% | 5.00GB | 63.2 | 0.017 |
| Turbo3 | 3 | 3 | 398MB | 1470MB | 27.1% | 5.00GB | 63.3 | 0.017 |
| Turbo3v2 | 3 | 2 | 399MB | 1163MB | 34.3% | 5.00GB | 63.3 | 0.007 |

### Results at 4096 Context

| Config | KV Delta | GPU Peak | Gen tok/s |
|--------|:---:|:---:|:---:|
| No-Quant | 39MB | 3.95GB | 85.3 |
| Affine4 | 18MB | 3.55GB | 82.0 |
| Turbo4 | 60MB | 3.95GB | 77.2 |
| Turbo3 | 63MB | 3.95GB | 76.6 |
| Turbo3v2 | 63MB | 3.95GB | 77.0 |

## Key Findings

### 1. KV Delta Is NOT Proportional to Compressed Size

Turbo3 (3-bit) shows 398MB — nearly same as turbo3v2 (3K/2V) at 399MB. Reducing value bits from 3→2 saved zero memory. This proves KV Delta is not measuring the compressed cache size.

### 2. GPU Peak Tells the Real Story

All turbo configs AND no-quant hit **exactly 5.00GB** GPU Peak at 32K. Affine4 is the only one at 4.55GB. This means:
- Turbo configs use the same actual GPU memory as no-quant at 32K
- The compression savings are being consumed by overhead or not reflected in peak measurement
- Affine4 is genuinely more memory-efficient during peak (prefill + decode)

### 3. Why Affine4 is More Memory-Efficient at Peak

Affine4's `quantizedMM` is an MLX built-in Metal kernel. It reads packed 4-bit data and dequants in-register during matmul — **no intermediate tensors**. MLX can optimize the entire graph because it understands quantized matmul natively.

Turbo's `compressedAttention()` uses custom Metal kernels dispatched via `MLXFast.metalKernel`. Each kernel dispatch creates input/output arrays that MLX must manage. The score kernel output, softmax, and value kernel output are all separate allocations. MLX can't fuse across custom kernel boundaries.

### 4. KV Delta Measures Different Things for Each Strategy

- **No-Quant/Affine4**: KV Delta ≈ 4% of theoretical. MLX's memory pool reuses prefill allocations for decode. The delta is tiny because most KV memory was already in the pool from prefill.
- **Turbo**: KV Delta ≈ 14-34% of theoretical. Higher because:
  - Raw FP16 prefill cache → compressed conversion creates new allocations (compressed arrays) while old ones (raw FP16) are freed. The pool may not immediately reclaim.
  - Codec state (rotation matrices, codebooks) adds ~7MB persistent overhead
  - Custom kernel dispatch creates intermediate arrays that inflate active memory
  - JIT-compiled kernel caches add to active memory

### 5. Gen tok/s at 32K Is Nearly Identical Across All Turbo Configs

63.1-63.3 tok/s regardless of bit-width (2, 3, or 4 bits). This means the bottleneck at 32K is NOT the compressed data read bandwidth but rather the kernel dispatch overhead or the softmax/rotation operations.

## Conclusion

**KV Delta is not a reliable metric for comparing turbo vs affine memory efficiency.** Use GPU Peak instead. At 32K, turbo configs match no-quant's GPU Peak (5.00GB) while affine4 achieves 4.55GB. The difference is architectural:
- Affine4 benefits from MLX's native `quantizedMM` graph optimization
- Turbo uses custom Metal kernels that MLX can't optimize across

**To close the GPU Peak gap**, we would need to either:
1. Integrate turbo quant into MLX's native kernel system (like affine)
2. Reduce the number of separate kernel dispatches in `compressedAttention()`
3. Implement a single fused SDPA kernel that does encode + score + softmax + value in one dispatch

**Quality remains turbo's clear advantage**: Gen KLD at 32K ranges from 0.007-0.034 for turbo configs vs 0.757 for affine4 — turbo is 22-108x better.
