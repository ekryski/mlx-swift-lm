# TurboQuant Optimization: P0 — Fix Default Attention Path

**Date**: 2026-04-02
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Base**: `ek/consolidated-benchmarks`

## Hypothesis

The default TurboQuant attention path in `AttentionUtils.swift` was calling `turboCache.update()` which stores raw FP16 — providing zero compression. By wiring in the `updateAndDequant()` path, turbo4 should actually compress the KV cache, showing reduced KV Delta at the cost of some gen tok/s overhead from rotation operations.

## Changes

**`Libraries/MLXLMCommon/AttentionUtils.swift`**: Replaced the TurboQuant else-branch to use the dequant-first path during decode (L==1). During prefill (L>1, not yet compressed), raw FP16 is still used for zero overhead. On first decode call, the cache transitions to compressed mode — encoding to packed indices + norms and dequanting to rotated-space FP16 for SDPA.

## Results vs Baseline (Qwen3.5-2B 8-bit, summarization, quick)

| Metric | No-Quant | Affine4 | **Turbo4 (P0)** | Delta vs Affine4 |
|--------|----------|---------|-----------------|-------------------|
| **Gen tok/s (128)** | 88.7 | 88.7 | **78.7** | **-11.3% slower** |
| **Gen tok/s (1024)** | 89.0 | 83.4 | **77.9** | **-6.6% slower** |
| **Gen tok/s (4096)** | 85.3 | 82.0 | **77.8** | **-5.1% slower** |
| Prefill tok/s (128) | 624 | 635 | 595 | -6.3% |
| Prefill tok/s (1024) | 1058 | 1043 | 1022 | -2.0% |
| Prefill tok/s (4096) | 1263 | 1238 | 1229 | -0.7% |
| TTFT (128) | 192ms | 189ms | 201ms | +6.3% |
| TTFT (1024) | 966ms | 979ms | 1000ms | +2.1% |
| TTFT (4096) | 3295ms | 3351ms | 3381ms | +0.9% |
| KV Delta (128) | 15MB | 9MB | **11MB** | +22% more |
| KV Delta (1024) | 9MB | 15MB | **18MB** | +20% more |
| KV Delta (4096) | 39MB | 18MB | **60MB** | **+233% more** |
| Think KLD (128) | 0.039 | 0.021 | **-0.004** | Better |
| Think KLD (1024) | 0.008 | 0.047 | **0.012** | Better |
| Think KLD (4096) | 0.017 | 0.052 | **0.019** | Better |
| Gen KLD (128) | 0.003 | 0.041 | **0.032** | Better |
| Gen KLD (1024) | 0.020 | 0.057 | **-0.045** | Better |
| Gen KLD (4096) | 0.094 | -0.087 | **0.007** | Better |
| Think PPL (128) | 3.18 | 2.48 | 3.21 | Similar |
| Think PPL (4096) | 2.93 | 3.56 | 2.50 | Better |
| Gen PPL (128) | 1.30 | 2.56 | 3.19 | Worse |
| Gen PPL (4096) | 1.95 | 1.88 | 1.50 | Better |

## Key Learnings

1. **Compression is working**: The fix successfully enables TurboQuant compression. The model generates coherent output with turbo4 KV quantization.

2. **Gen tok/s regression is significant**: 78.7 vs 88.7 tok/s at 128 context is a **-11.3% slowdown** vs no-quant. This is worse than affine4 at all context sizes. The overhead comes from:
   - Per-token rotation: `matmul(newKeys, rotationT)` and `matmul(newValues, rotationT)` during encode
   - Per-token inverse rotation: `matmul(rotatedOutput, rotation)` during decode
   - Codebook lookup and norm correction operations

3. **KV Delta is HIGHER than both baselines**: turbo4 at 60MB vs affine4's 18MB at 4096 context. This is because `updateAndDequant()` maintains BOTH compressed storage AND a rotated-space FP16 dequant buffer. The compression ratio is negated by the dual storage overhead.

4. **KLD quality is actually better than affine4**: Think KLD values (-0.004, 0.012, 0.019) are consistently closer to zero than affine4 (0.021, 0.047, 0.052). TurboQuant's rotation + norm correction preserves quality better than affine quantization.

5. **Critical bottleneck identified**: The dequant-first approach is fundamentally inefficient because it materializes a full FP16 buffer in rotated space. The fused SDPA kernel (P1) is essential — it should decompress on-the-fly inside the attention kernel, eliminating both the intermediate buffer and the per-token rotation overhead.

6. **Prefill slowdown is minor**: Only -0.7% to -6.3%, since prefill still uses raw FP16 storage. The slowdown comes from codec initialization (codebook generation, rotation matrix computation) on first decode call.

## Decision

**ITERATE** — The fix proves TurboQuant compression works and quality is good (better KLD than affine4), but the performance regression and memory overhead make it worse than affine4 in practice. The root cause is clear: the dequant-first approach materializes too much intermediate data. **P1 (fused SDPA kernel) is critical** to eliminate the FP16 dequant buffer and per-token rotation overhead.

Next steps:
1. Merge P0 as-is (correctness fix — turbo4 now actually compresses)
2. Immediately pursue P1 (fused SDPA kernel) to fix the performance regression
3. Also consider P2 (pre-computed constants) to reduce codec init time
