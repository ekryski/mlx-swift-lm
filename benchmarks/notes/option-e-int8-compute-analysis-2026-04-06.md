# Option E: Int8 Quantized MatMul Path — Analysis

**Date**: 2026-04-06
**Status**: Not viable on Apple Silicon GPU

## Goal

Add Int8 compute path for quantized matmul kernels to achieve 2x throughput improvement by using Int8×Int8→Int32 accumulation instead of FP16×FP16→FP32.

## Analysis

### Metal GPU simdgroup_matrix: No Int8 Support

Apple's Metal GPU `simdgroup_matrix` instruction only supports:
- `float` (FP32)
- `half` (FP16)
- `bfloat` (BF16)

**Int8 is NOT supported by Metal's simdgroup_matrix.** This was verified by:
1. Searching the MLX-Swift Metal kernel codebase — no Int8 simdgroup_matrix instantiations
2. Web research confirming Apple GPU ALUs don't scale to Int8 multiply
3. Metal Feature Set Tables showing no Int8 matrix support

Sources:
- [Metal Benchmarks — Apple GPU microarchitecture](https://github.com/philipturner/metal-benchmarks)
- [AMX Benchmarks](https://github.com/philipturner/amx-benchmarks)

### AMX (Apple Matrix Accelerator): CPU-Only

Apple's AMX coprocessor DOES support Int8 operations, but:
- AMX operates on the **CPU**, not the GPU
- Not accessible from Metal shader code
- Would require running MoE on CPU instead of GPU — massive regression
- MLX uses AMX for CPU-side operations (separate execution path)

### Neural Engine: Not Viable for LLM Inference

Apple's Neural Engine supports Int8, but:
- Operates as an independent accelerator (not co-processor)
- Limited model size support
- Not integrated with MLX's execution model
- Context switching overhead makes it impractical for per-layer use

### Software Int8 in Metal Shader (Option E-A)

Even without hardware support, we could implement Int8 math in the shader:

```metal
// Instead of:
float accum = x_float * w_dequant_float;  // FP32 multiply

// Do:
int32_t accum = int8_t(x_quant) * int8_t(w_dequant);  // Int32 multiply
```

**Problem**: On Apple GPUs, integer multiply has the **same throughput** as FP32 multiply. There's no dedicated Int8 SIMD unit. The conversion overhead (FP16→Int8 quantization of x, Int8→FP32 dequant of result) would actually make it **slower**.

### What About M3/M4/M5?

No public evidence that newer Apple Silicon adds Int8 SIMD to the GPU. Apple's direction for low-precision compute is:
1. **GPU**: FP16/BF16 via simdgroup_matrix (sufficient for most ML workloads)
2. **Neural Engine**: Int8/Int4 for fixed-model inference (not flexible enough for LLM)
3. **AMX**: Int8 on CPU side (used by MLX for CPU dispatch)

## Conclusion

**Option E is not viable on Apple Silicon GPU.** The hardware lacks Int8 SIMD acceleration in Metal shaders. All three approaches (E-A software Int8, E-B AMX, E-C Neural Engine) either provide no benefit or are architecturally incompatible.

The fundamental bottleneck for MoE decode remains **memory bandwidth** (reading 4-bit quantized weights), not compute throughput. Even if Int8 compute were available, it wouldn't help because:
1. Weights are already 4-bit packed — Int8 compute doesn't reduce memory reads
2. Decode is M=1 (memory-bound, not compute-bound)
3. The ~43% CPU overhead (`.item()` sync) is irreducible

## Future Possibilities

If Apple adds Int8 simdgroup_matrix support in a future Metal version, the implementation path would be:
1. Add `BaseMMAFrag<int8_t, 8, 8>` specialization in `steel/gemm/mma.h`
2. Modify `QuantizedBlockLoader` to dequantize to Int8 instead of float
3. Modify `qmv_fast_impl` to use Int8 accumulation with Int32 accumulators
4. Add dispatch path in `quantized.cpp` for Int8 kernel variants

This would primarily benefit **prefill** (compute-bound at large batch sizes), not decode.
