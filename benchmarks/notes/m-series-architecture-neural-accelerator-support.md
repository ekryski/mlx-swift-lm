# M-Series Architecture: Steel SDPA Support & Neural Accelerator Opportunities

**Date**: 2026-04-09
**Context**: Analysis of Steel (fused flash) attention kernel viability across Apple Silicon generations, with focus on head_dim=256/512 support for Gemma4 models and future ANE offloading.

---

> Definitions from the kernel code:
> 
> - BQ = query block size (rows of Q processed per threadgroup)
> - BK = key block size (columns of K processed per inner loop iteration)
> - BD = head dimension
> - WM × WN = warp grid (number of SIMD groups)

## 1. Steel SDPA Head Dimension Support

### What Steel SDPA Does

The Steel attention kernel computes `softmax(Q × K^T / scale) × V` in a single GPU dispatch, tiled in registers. It **never materializes the full L×L attention score matrix** in GPU memory — unlike the matmul fallback which allocates `[B, H, L, L]` intermediate tensors.

For Gemma4 E2B at 4k context with 35 layers, the matmul fallback allocates:
- 28 sliding layers (head_dim=256): ~3.7 GB of score matrices
- 7 full layers (head_dim=512): ~0.9 GB of score matrices
- **Total: ~4.6 GB of unnecessary intermediate allocations during prefill**

### Current Kernel Instantiations

| BD (head_dim) | Vector (decode, L≤8) | Steel Full (prefill, L>8) | Tile Config |
|---------------|---------------------|--------------------------|-------------|
| 64 | ✅ | ✅ | BQ=32 BK=32 WM=4 WN=1 |
| 80 | ❌ | ✅ | BQ=32 BK=32 WM=4 WN=1 |
| 96 | ✅ | ❌ | — |
| 128 | ✅ | ✅ | BQ=32 BK=16 WM=4 WN=1 |
| 256 | ✅ | ✅ (newly enabled) | BQ=16 BK=16 WM=2 WN=1 |
| 512 | ✅ (newly added) | ✅ (newly added) | BQ=8 BK=8 WM=1 WN=1 |

### BD=256 History

The BD=256 Steel full attention kernel was **compiled but deliberately disabled** in upstream MLX with the comment: "BD=256 Steel full attention is compiled but disabled — matmul fallback is faster on M1 Max for all tested context lengths (1K-32K)."

This was likely benchmarked for models with head_dim=256 AND many KV heads where the matmul path benefits from Apple's highly optimized GEMM kernels. For Gemma4 with GQA (8 query heads, 1 KV head), the tradeoff is different — the matmul fallback allocates disproportionately large score matrices relative to the model's actual compute.

**We re-enabled BD=256 for full SDPA** because the memory savings (eliminating ~3.7 GB of score matrices at 4k) far outweigh any potential compute overhead. The kernel was already compiled and tested — just disabled in dispatch.

### BD=512 Constraints

BD=512 is the most register-intensive configuration:

**Register pressure per thread:**
- Otile accumulator: TQ=1 × TD=64 frags × 8 floats = 512 floats = **2048 bytes**
- Stile scores: TQ=1 × TK=1 × 8 floats = 32 bytes
- Misc (max_score, sum_score, factor arrays): ~120 bytes
- **Total: ~2200 bytes per thread**

**Per SIMD group (32 threads): ~69 KB**

**Threadgroup memory (BQ=8, BK=8):**
- Q shared memory: 8 × (512 + 8) × 2 = 8,320 bytes
- KV shared memory: max((8+8)×512, 8×(512+8)) × 2 = 16,384 bytes
- **Total: ~24 KB** (fits in 32 KB limit)

---

## 2. Per-Generation Analysis

### M1 Max (2021)

| Spec | Value |
|------|-------|
| GPU cores | 32 |
| Register file per core | ~208 KB |
| Threadgroup memory | 32 KB |
| L2 cache | 512 KB |
| SLC (L3) | 48 MB |
| Memory bandwidth | 400 GB/s |
| FP32 TFLOPS | 10.4 |

**BD=256 occupancy:** BQ=16, WM=2 → 64 threads/threadgroup. Per-SIMD ~20 KB. **~10 SIMDs/core = 21% occupancy.** Viable — moderate latency hiding.

**BD=512 occupancy:** BQ=8, WM=1 → 32 threads/threadgroup. Per-SIMD ~69 KB. **~3 SIMDs/core = 6% occupancy.** Poor — very limited latency hiding. The kernel will stall on memory reads frequently. However, it avoids materializing ~0.9 GB of score matrices for 7 full-attention layers, which may be a net win despite low occupancy. **Gated behind `MLX_SDPA_NO_BD512=1` env var** to allow fallback to matmul where that's faster.

### M5 Max (2026)

| Spec | Value |
|------|-------|
| GPU cores | 40 |
| Total register file | ~8.2 MiB (~210 KB/core) |
| Dedicated matrix unit | 1024 FMA ops/core/cycle |
| Matrix unit internal buffers | Register-adjacent high-speed memory |
| Memory bandwidth | 614 GB/s |
| Estimated FP32 TFLOPS | ~20 (general), ~57 (matrix unit) |

**The game changer: dedicated matrix multiplication units.** Each GPU core has hardware blocks specifically for matrix operations with their own internal high-speed buffers. The Metal compiler routes `simdgroup_matrix` operations to these units automatically.

**BD=512 with matrix unit offload:**
The Otile accumulator (2048 bytes/thread) is pure FMA accumulation — exactly what the matrix unit is designed for. When the compiler routes Otile to the matrix unit's internal buffers:

- General register pressure drops to ~152 bytes/thread (Stile + misc only)
- Per-SIMD: ~4.8 KB (from 69 KB)
- **~44 SIMDs/core = 92% occupancy** (from 6%)

This transforms BD=512 from "barely viable" to "excellent" on M5 Max. No code changes needed — our kernel already uses `MMATile`/`tile_matmad` which compile to `simdgroup_matrix` operations.

**BD=256 with matrix unit offload:**
Similarly, BD=256 occupancy would improve significantly on M5, though it's already viable on M1.

### Summary Table

| Config | M1 Max Occupancy | M5 Max Occupancy (est.) | Memory Saved (4k ctx) |
|--------|-----------------|------------------------|----------------------|
| BD=256 full SDPA | ~21% | ~90%+ | 3.7 GB (28 layers) |
| BD=512 full SDPA | ~6% | ~90%+ | 0.9 GB (7 layers) |
| BD=512 vector SDPA | N/A (decode=good) | N/A | negligible |
| Matmul fallback | 100% (GEMM optimal) | 100% | 0 (materializes all) |

---

## 3. Optimization Tiers

### Tier 1: Current Implementation (done)

- **BD=256 full SDPA enabled** in dispatch (was compiled but disabled)
- **BD=512 full SDPA added** with BQ=8, BK=8, WM=1, WN=1 tile config
- **BD=512 vector SDPA added** for decode path
- **Float32 BD=256 instantiation added** (was float16/bfloat16 only)
- **Runtime flag**: `MLX_SDPA_NO_BD512=1` disables BD=512 Steel for hardware where matmul is faster
- Files: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`
- Kernels: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal`

**Blocking issue**: The `swift test -c release` build overrides our custom metallib with SPM's default compilation. Need to fix build integration so SPM includes the updated `.metal` sources.

### Tier 2: M5-Specific Tile Sizes (future)

Compile a second BD=512 variant with larger tiles for M5+:

```
BQ=16, BK=8, WM=2, WN=1 (33 KB TGP — may need M5's larger TGP limit)
```

This processes 2x more query rows per threadgroup, doubling prefill throughput for head_dim=512 layers. Select at runtime based on GPU architecture generation:

```cpp
// In scaled_dot_product_attention.cpp tile selection:
if (bd == 512 && arch_gen_ >= 17) {  // M5 = generation 17
    wm = 2; bq = 16; bk = 8;  // Larger tiles for M5+
} else if (bd == 512) {
    wm = 1; bq = 8; bk = 8;   // Conservative for M1-M4
}
```

Estimated improvement: 2x prefill throughput for head_dim=512 layers on M5 Max vs current implementation.

### Tier 3: ANE (Apple Neural Engine) Offloading (future)

The M5 Max Neural Engine runs in parallel with the GPU — operations on ANE don't consume GPU cycles or bandwidth. Key opportunity areas:

#### 3a. LM Head on ANE

The vocabulary projection (`hidden_state × lm_head_weight → logits`) is:
- Gemma4: [1, 1, 1536] × [1536, 262144] = 402M multiply-accumulates
- Pure matmul, no KV cache dependency
- Runs once per token, after the transformer layers complete
- Currently one of the largest single ops per decode token

**Pipeline**: GPU finishes transformer layers → hidden state copied to ANE (IOSurface) → ANE computes logits + softmax → CPU samples token. GPU starts next token's layer 0 immediately, overlapping with ANE's LM head computation.

ANE advantages:
- Softmax is 33.8x faster than CPU on ANE (per Orion benchmarks)
- Int8 matmul is natively supported — LM head could be quantized to int8 for ANE
- Runs in parallel with GPU starting the next token's forward pass

**Risk**: ANE has a 32K channel limit. Gemma4's 262K vocab exceeds this — would need to split the projection into 262K/32K ≈ 9 tiles, each processed separately. The tiling overhead may negate the parallelism benefit.

#### 3b. PLE (Per-Layer Embeddings) on ANE

Gemma4 E2B's `perLayerModelProjection(h)` is a matmul ([B, T, 1536] × [1536, plDim]) that runs every decode step. It could execute on ANE while the GPU handles the current layer's attention:

**Pipeline**: GPU layer N attention starts → ANE computes PLE for layer N+1 → GPU layer N attention finishes → PLE result available for layer N+1's feed-forward.

This overlaps PLE compute with attention compute, hiding the PLE latency entirely.

#### 3c. ANE Integration Approach

Direct ANE access via Orion-style `_ANEClient`/`_ANECompiler` (private APIs):
- Bypasses CoreML overhead (~2.3ms dispatch → ~0.095ms with direct access)
- Requires IOSurface for data transfer (zero-copy with unified memory)
- Supported dtypes: fp32, bf16, fp16, int8
- 32 MB SRAM — working set must fit or performance drops 30%

**Decision criteria**: Worth pursuing if parallel ANE offloading achieves >5% per-token latency reduction with <1 week implementation effort.

**Risk assessment**:
- `_ANEClient` is a private API — may break between macOS versions
- 32K channel limit requires tiling for large vocab projections
- IOSurface round-trip adds ~2.3ms latency (reduced to ~0.1ms with Orion's direct path)
- Must verify ANE can pipeline with GPU without stalling either

---

## 4. References

- M1 Max architecture: `benchmarks/notes/inference-architecture-m1-max.md`
- Orion ANE framework: https://github.com/mechramc/Orion (170+ tok/s GPT-2 124M)
- Orion paper: https://arxiv.org/abs/2603.06728
- Ensue-network ANE benchmarks: 6.31x faster than CoreML on DistilBERT (M5 Max)
- ANE kernel example: `benchmarks/notes/ane-kernel-example-distilbert.md`
- Steel attention source: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`
- SDPA dispatch: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`
- Performance optimization plan: `benchmarks/notes/final-performance-optimization-plan-04-07-2026.md` (items 3i, 6a-6e)
