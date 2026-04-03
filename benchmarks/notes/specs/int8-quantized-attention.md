# Feature Spec: Int8 Quantized Attention (P8)

**Status**: Research / Design
**Priority**: Medium-High — 1.24-1.41x attention speedup potential
**Reference**: Draw Things v1.20260330.0, reaching ~110 TFLOPs on M5 Max

## Problem

During decode, the attention computation (Q×K scoring + softmax + Attn×V aggregation) operates on FP16 tensors. On Apple Silicon, Int8 matrix multiplication can be 1.6-1.9x faster than FP16 due to the Neural Engine / AMX supporting wider Int8 SIMD operations. The idea: quantize Q/K/V to Int8 on-the-fly before attention, compute in Int8, dequantize after.

## Reference Implementation: Draw Things

Draw Things implements "Metal Quantized Attention" achieving 1.24-1.41x speedup over their Metal Flash Attention:

### Quantization Strategy

- **Queries & Keys**: Row-group-wise scale quantization
  - For each row (or group of rows), compute `scale = max(abs(row)) / 127`
  - `q_int8 = round(q_fp16 / scale)`
  - Symmetric quantization (no zero-point)

- **Values**: Row-wise affine quantization
  - Per-row scale + zero-point: `v_int8 = round((v_fp16 - zero) / scale)`
  - Affine because V has less uniform distribution than Q/K

### Fused Shader

The shader handles everything in one dispatch:
1. Row-wise dynamic activation quantization (FP16 → Int8)
2. Int8-to-Int8 matrix multiplication
3. Post-accumulation dequantization (Int32 accumulator → FP16)

This achieves 1.61-1.87x faster than FP16 baseline for the Int8 matmul portion.

### Quality

Draw Things reports the quality impact as "comparable to, and often better than, our 6-bit palettized quantization scheme." The key insight: attention scores go through softmax which is numerically forgiving — small quantization errors in Q×K get washed out by the exponential.

## Technical Challenges for mlx-swift-lm

### Challenge 1: MLX Doesn't Support Int8 Attention

MLX's SDPA kernels are only instantiated for float16, bfloat16, and float32. The Steel Attention and SDPA Vector kernels have no Int8 path. We can't use MLXFast.scaledDotProductAttention with Int8 inputs.

### Challenge 2: MLXFast.metalKernel Has Limited Int8 Support

The `MLXFast.metalKernel` API (used by our turbo kernels) does support Int8 arrays as inputs/outputs, but:
- No `simdgroup_matrix` (AMX) access — the key to Int8 speedup on Apple Silicon
- No `simd_multiply_accumulate` for Int8×Int8→Int32
- Metal Shading Language does support `char` (int8) and `uchar` (uint8) types natively
- The `simd_sum` and basic arithmetic work with int8, but no hardware-accelerated GEMM

### Challenge 3: Accumulator Precision

Int8 × Int8 multiplication produces Int16, accumulated into Int32. For a head dimension of 128, that's 128 multiply-accumulates. The Int32 accumulator can overflow if values are large — need careful scale management.

## Proposed Implementation

### Approach A: Custom Metal Kernel via MLXFast.metalKernel (Feasible Now)

Write a custom Int8 attention kernel that:

1. **Online Q/K quantization**: Load FP16 Q/K, quantize to Int8 in-register
2. **Int8 dot product**: Manual SIMD lane accumulation (no AMX, but still faster due to 2x bandwidth)
3. **FP32 softmax**: Dequantize scores, apply softmax in FP32
4. **Int8 V aggregation**: Quantize V to Int8, accumulate weighted sum in Int32, dequantize output

```metal
// Pseudocode for Int8 attention score kernel
uint lane = thread_position_in_grid.x;
uint q_idx = thread_position_in_grid.y;
uint k_idx = thread_position_in_grid.z;

// Load and quantize Q to Int8
float q_scale = q_row_scales[q_idx];
char q_int8 = (char)round(q_fp16[q_idx * D + d] / q_scale);

// Load and quantize K to Int8
float k_scale = k_row_scales[k_idx];
char k_int8 = (char)round(k_fp16[k_idx * D + d] / k_scale);

// Int8 dot product (accumulated in int32)
int acc = 0;
for (uint d = lane; d < D; d += 32) {
    acc += (int)q_int8[d] * (int)k_int8[d];
}
acc = simd_sum(acc);

// Dequantize score
float score = (float)acc * q_scale * k_scale;
```

**Expected speedup**: 1.1-1.3x (bandwidth savings from Int8 reads, but no AMX acceleration)

### Approach B: C++ Integration with Metal Library (Higher Performance)

Bypass MLXFast.metalKernel and compile a custom `.metallib` with full access to:
- `simdgroup_matrix_storage<int8_t>` for AMX-accelerated Int8 GEMM
- `simdgroup_multiply_accumulate` for hardware Int8×Int8→Int32 matmul
- Explicit threadgroup memory management

This requires:
1. Writing a `.metal` file compiled separately (like SwiftLM's `build-metallib.sh`)
2. Loading the metallib at runtime
3. Dispatching via Metal compute pipeline (not MLXFast)

**Expected speedup**: 1.4-1.8x (matching Draw Things numbers with AMX)

### Approach C: Hybrid — Int8 Storage + FP16 Compute (Simplest)

Don't change the attention kernel. Instead:
1. Store K/V cache as Int8 (with per-row scales)
2. Dequantize to FP16 on-the-fly before SDPA
3. Use standard MLXFast.scaledDotProductAttention

This gives memory savings (Int8 = 2x compression vs FP16) but no compute speedup. Essentially a simpler version of turbo quantization without the rotation.

## Interaction with TurboQuant

Int8 attention is **orthogonal** to TurboQuant KV compression:

| Scenario | KV Storage | Attention Compute | Notes |
|----------|-----------|-------------------|-------|
| Current turbo | Packed b-bit + norms | Metal score/value kernels (FP32) | Our current approach |
| Int8 attention (no turbo) | FP16 or Int8 | Int8 matmul | Draw Things approach |
| **Combined** | Packed b-bit + norms | Int8 score kernel | Best of both worlds |

The combined approach would modify our existing Metal score/value kernels to use Int8 arithmetic internally:
- Score kernel: dequant K from packed → Int8 (skip FP16), Int8 dot product with Int8 Q
- Value kernel: dequant V from packed → Int8, Int8 weighted sum

This combines turbo's compression advantage with Int8's compute advantage.

## Implementation Plan

### Phase 1: Approach A (Quick Win)

1. Write `int8ScoreKernel` as a variant of existing `scoreKernelSource`
2. Add online Q quantization (per-row scale computation)
3. Modify K dequant path to output Int8 instead of FP32
4. Int8 dot product with Int32 accumulation + dequant
5. Benchmark vs FP32 score kernel

### Phase 2: Approach B (Full Performance)

1. Create `TurboQuantInt8Attention.metal` compiled via build script
2. Implement `simdgroup_matrix` Int8 GEMM for Q×K
3. Two-pass attention for long sequences (like SDPA vector 2-pass)
4. Integration with AttentionUtils.swift dispatch

### Phase 3: Combined with TurboQuant

1. Modify turbo score kernel to dequant packed K → Int8 (not FP32)
2. Pre-quantize queries to Int8 with per-row scales
3. Int8 dot product in the score kernel
4. Benchmark combined turbo + Int8 vs turbo alone

## Estimated Impact

| Approach | Speedup | Complexity | Timeline |
|----------|:-------:|:----------:|:--------:|
| A: MLXFast Int8 kernel | 1.1-1.3x | Medium | 1-2 weeks |
| B: Custom metallib + AMX | 1.4-1.8x | High | 3-4 weeks |
| C: Int8 storage only | 1.0x (memory only) | Low | 1 week |

## Hardware Considerations

- **M1/M2**: AMX supports Int8 but with lower throughput
- **M3/M4**: Improved AMX Int8 throughput
- **M5**: Best Int8 AMX performance, Neural Accelerator for attention

Draw Things achieves ~110 TFLOPs on M5 Max — this is well above theoretical FP16 throughput, confirming AMX Int8 is the key enabler.

## Open Questions

1. Can `MLXFast.metalKernel` access `simdgroup_matrix` intrinsics? Initial research says no — need to verify with actual compilation test.
2. Does M1 Max's AMX support Int8 GEMM at all? The ISA documentation is sparse.
3. What's the quality impact of Int8 attention on thinking models specifically? Thinking phase may be more sensitive to quantization noise in attention scores.
