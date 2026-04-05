# MoE Prefill & Decode Bottleneck Analysis

**Date**: 2026-04-04
**Hardware**: Apple M1 Max (400 GB/s, 10.4 TFLOPS FP32), 64 GB RAM
**Branch**: `ek/tom-eric-moe-tuning`

## Summary

Deep analysis of inference bottlenecks across three MoE models: Qwen3.5-35B-A3B,
GPT-OSS-20B, and Nemotron Cascade 2-30B. Identifies root causes and quantifies
gaps between theoretical and actual throughput.

---

## 1. Qwen3.5-35B-A3B (GatedDeltaNet Hybrid MoE)

### Architecture
- 40 layers: 30 GatedDeltaNet (linear attention) + 10 full attention (every 4th)
- MoE: 256 experts, top-8 routing, moe_intermediate_size=512, + shared experts
- hidden=2048, vocab=248,320, head_dim=128, 16 attn heads, 2 KV heads
- 35B total / ~3B active parameters per token
- 4-bit model size: 18.16 GB

### Theoretical Maximums (4-bit, M1 Max)
- Active weight reads per decode step: ~1.0-1.5 GB
  - MoE experts (8/256 active, 40 layers): ~500 MB
  - Attention/GatedDeltaNet projections: ~200 MB
  - LM head (vocab=248,320 x hidden=2048): ~254 MB
  - Norms, embeddings, gates: ~50 MB
- Decode max: 400 / 1.0-1.5 = **267-400 tok/s** (bandwidth bound)
- Prefill max: ~1,800 tok/s (compute bound, 10.4 TFLOPS / (2 x 3B x 4/16))

### Actual Performance (4-bit, no-quant KV)

**Before optimization:**
| Context | Prefill tok/s | Decode tok/s | TTFT |
|---------|--------------|-------------|------|
| 128 | 210.8 | 52.5 | 556ms |
| 256 | **42.0** | 52.6 | 6,231ms |
| 512 | **39.1** | 51.9 | 13,286ms |
| 1024 | 84.4 | 51.6 | 12,498ms |
| 2048 | 146.2 | 51.6 | 14,381ms |
| 32768 | 429.2 | 40.0 | 76,719ms |

**After optimization (commit 2a695c0):**
| Context | Prefill tok/s | Decode tok/s | TTFT |
|---------|--------------|-------------|------|
| 128 | 191.9 | 52.9 | 611ms |
| 1024 | **478.2** | 52.9 | 2,390ms |
| 4096 | **497.7** | 51.2 | 8,731ms |

### Root Causes

**Prefill anomaly (42 tok/s at 256 context → 478 tok/s after fix):**

1. **GatedDeltaNet sequential Metal kernel** (`GatedDelta.swift`): 30/40 layers use
   a kernel that loops `for (int t = 0; t < T; ++t)` sequentially. Each step depends
   on previous state. This makes prefill O(T) per layer for 75% of the model.

2. **4-bit gatherQuantizedMM performance cliff at T=256-512**: The dequantization
   kernel has poor GPU utilization at these batch sizes. Evidence: 8-bit achieves
   204-470 tok/s at these sizes while 4-bit gets 42-39 tok/s.

3. **Default 512-token prefill chunks**: Creates many `eval(cache)` sync barriers
   and forces MoE layers to operate at the worst batch size.

**Fix**: Override `prepare()` in `Qwen35Model` with min 4096-token chunks. This
keeps MoE above the performance cliff and reduces sync barriers. 5.7x improvement.

**Decode gap (47 tok/s actual vs 267-400 tok/s theoretical):**

Metal System Trace profiling (Apr 5, via xctrace — zero overhead):
- **78 Metal encoder dispatches per decode token**
- **12.6ms actual GPU compute per token**
- **22.2ms wall clock per token** (45 tok/s)
- **57% GPU utilization — 43% is dispatch/scheduling/CPU overhead**
- ~3,500 encoders/sec during decode

The 43% non-GPU overhead includes:
- Metal command buffer creation and encoder dispatch scheduling
- CPU<->GPU synchronization gaps between command buffers
- MLX lazy graph compilation (amortized after first token)

The 12.6ms GPU time itself is explained by:
- `gatherQuantizedMM` sparse access pattern defeats GPU cache hierarchy
- LM head / tied embeddings (vocab=248,320 × hidden=2048) read every token
- 256 expert weights in memory even though only 8 active (cache pollution)
- Per-token `.item()` sync in convertToToken (fixed for non-thinking models)

**Remaining prefill ceiling**: At ~500 tok/s we're at ~27% of theoretical max.
The ceiling is the GatedDeltaNet sequential kernel. A parallel scan algorithm
(O(log T) via associative scan) would be transformative but is a multi-day effort.

---

## 2. GPT-OSS-20B (Pure Transformer MoE)

### Architecture
- 36 layers, ALL pure transformer (no SSM/GatedDeltaNet)
- MoE: 128 experts, top-4 routing, hidden=2,880, intermediate=2,880
- Alternating full/sliding-window attention (128-token window)
- Custom SwiGLU with clipping (alpha=1.702, limit=7.0), uses `compile()`
- vocab=201,088, head_dim=64, 64 attn heads, 8 KV heads (GQA)
- Default `prepare()` (512-token chunks)

### Performance (4-bit, no-quant KV)
| Context | Prefill tok/s | Decode tok/s |
|---------|--------------|-------------|
| 128 | 360 | 59.8 |
| 512 | 550 | 60.0 |
| 1024 | 609 | 59.1 |
| 2048 | 689 | 57.5 |
| 8192 | 666 | 51.5 |
| 32768 | 545 | 37.7 |

### Analysis
- **No prefill anomaly** — monotonic improvement. Pure transformer benefits from
  batch parallelism at all sizes.
- **Decode at ~9.4% of theoretical** — active reads ~0.5-0.6 GB, achievable BW
  320 GB/s → theoretical ~640 tok/s vs actual 60. Same `gatherQuantizedMM`
  bottleneck as Qwen3.5.
- The larger prefill chunk optimization would NOT help (no sequential layers).
- The `.item()` sync fix helps if used as non-thinking model.
- Primary improvement opportunity: fused MoE decode kernel.

---

## 3. Nemotron Cascade 2-30B (Hybrid 4-Type Architecture)

### Architecture
- Hybrid with 4 layer types via `hybridOverridePattern` string:
  - 'M' = Mamba2 SSM, '*' = Attention, '-' = Dense MLP, 'E' = Sparse MoE
- Mamba2 SSM layers with TWO paths in `SSM.swift`:
  - `seqLen == 1` (decode): Metal kernel `ssm_kernel` — fast, single-step
  - `seqLen > 1` (prefill): `ssmAttn()` — **O(T^2) surrogate attention matrix**
- MoE uses `NemotronHSwitchMLP` with **relu2** activation (not silu/GLU)
- Sigmoid routing with group-based expert selection + correction bias
- Optional shared experts
- Default `prepare()` (512-token chunks)
- **No benchmark data yet**

### Analysis
- **Different bottleneck from Qwen3.5**: Mamba2 prefill uses O(T^2) `ssmAttn()`
  which materializes a T x T surrogate attention matrix. Larger chunks make it WORSE.
- NemotronH needs the OPPOSITE strategy from Qwen3.5: smaller prefill chunks
  for Mamba layers, OR a chunked ssmAttn that processes C-token blocks.
- The decode path uses the Metal `ssm_kernel` (single-step, same as GatedDeltaNet)
  so it should be efficient for decode.
- Need to benchmark before optimizing.

---

## 4. Cross-Model Bottleneck: gatherQuantizedMM

All three MoE models show the same pattern: decode utilization is <20% of
theoretical bandwidth. The shared bottleneck is `gatherQuantizedMM` in
`SwitchLayers.swift`.

### Why gatherQuantizedMM is slow at decode

During decode (batch=1, seqLen=1), each MoE layer:
1. Routes to top-K experts (K=4-8)
2. For each expert, reads weight slices from [numExperts, outputDim, packedInputDim]
3. Unpacks 4-bit → float16
4. Multiplies by input vector
5. The K experts are typically NOT contiguous in the tensor

The memory access pattern is: read K non-contiguous slices from a large 3D tensor.
With 128-256 experts, these slices are scattered across ~0.5-18 GB of memory.
GPU cache lines (128 bytes on Apple Silicon) are loaded but only partially used.

### Potential solutions (from most to least impact)

1. **Fused MoE Metal kernel**: Single kernel does expert selection + dequant + GEMM
   without intermediate materialization. Similar architecture to TurboFlash.

2. **Expert weight reordering**: Cluster frequently co-selected experts in memory.
   Requires profiling expert selection patterns across a corpus.

3. **Upstream MLX improvement**: The `gatherQuantizedMM` implementation in MLX C++
   may have optimization opportunities (tiling, prefetching, etc.).

---

## 5. SSM Models: Sequential Bottleneck Taxonomy

| Model | SSM Type | Prefill Path | Decode Path | Sequential Bottleneck |
|-------|----------|-------------|-------------|----------------------|
| Qwen3.5 | GatedDeltaNet | Metal kernel O(T) loop | Metal kernel O(1) | **Severe** — 75% of layers |
| NemotronH | Mamba2 | ssmAttn O(T^2) matmul | Metal kernel O(1) | **Different** — O(T^2) not O(T) |
| Jamba | Mamba1 | Swift for-loop O(T) | Same loop O(1) | **Severe** — no Metal kernel |
| FalconH1 | Mamba2 | ssmAttn O(T^2) | Metal kernel O(1) | Same as NemotronH |

### Parallel scan opportunity — status and challenges

**Mamba2** (`ssmAttn` in SSM.swift): The Mamba recurrence `s_t = dA * s_{t-1} + dB * x_t`
IS a linear recurrence. `ssmAttn` already implements a parallelized O(T^2) surrogate
attention formulation for this. Parallel scan (Blelloch) is also viable. **Already solved.**

**GatedDeltaNet** (GatedDelta.swift): The recurrence `S_t = g*S_{t-1} + k*(v - k^T*S_{t-1})*beta`
is **non-linear** because `delta_t = (v_t - k_t^T * S_{t-1}) * beta_t` depends on the
current state. Expanding:

```
S_t = (g_t * I - beta_t * k_t * k_t^T) * S_{t-1} + beta_t * k_t * v_t^T
    = A_t * S_{t-1} + B_t
```

This IS linear in the matrix space, but `A_t = g_t * I - beta_t * k_t * k_t^T` is a
[Dk, Dk] = [128, 128] matrix. Parallel scan on 128x128 matrices requires:
- O(T × 128^2 × 128) = O(T × 2M) work per scan step — prohibitively expensive
- Storing T intermediate 128x128 matrices in GPU memory

**Tested (commit c9aa261):** Chunk-parallel approach (split T into 64-token chunks,
Metal kernel per chunk, propagate state between chunks) showed **no improvement** —
the same sequential work is just redistributed across kernel launches.

**Matrix-form parallel scan — estimated impractical (FLOP analysis, NOT measured):**
The matrix form IS linear (A_t = g*I - beta*k*k^T), so Blelloch scan works in theory.
But Dk=128 makes each combine cost O(128^3) = 2M FLOPs. FLOP estimate: scan ~6.0 TFLOPS
vs sequential kernel ~80.6G FLOPs → ~74x more compute. However, these are estimates.
Actual Metal matmul efficiency and kernel launch overhead could change the picture.

**Hypothesis (NEEDS MEASUREMENT): GatedDeltaNet kernel may not be the main bottleneck.**
FLOP estimates suggest the kernel is ~0.5ms out of 2,390ms TTFT (0.02%), with Linear
projections and MoE gatherQuantizedMM dominating. But FLOP counts can be misleading —
memory latency, kernel dispatch overhead, and GPU occupancy are not captured by FLOPs.
**Per-component profiling with eval() barriers is needed to verify where time actually goes.**

---

## 6. Implementation Priority (Updated Apr 5 — based on Metal System Trace data)

See `metal-trace-decode-profile-2026-04-05.md` for raw GPU profiling data.

**Key finding: 43% of decode time is dispatch overhead, not GPU compute.**

| # | Optimization | Models Affected | Expected Impact | Difficulty | Status |
|---|-------------|-----------------|-----------------|------------|--------|
| 1 | ~~Prefill chunk size~~ | Qwen3.5 | **5.7x prefill** | Low | ✅ Done |
| 2 | ~~.item() sync skip~~ | All non-thinking | **~10% decode** | Low | ✅ Done |
| 3 | Speculation tuning | MoE models | 5-15% decode | Low | Next |
| 4 | Lazy log-prob | All | 5-10% decode | Low | Next |
| 5 | MLX compile() for layers | Qwen3.5 | ~15% decode | Medium | Tier 2 |
| 6 | Fused RMSNorm+Linear | All | ~10% decode | Medium | Tier 2 |
| 7 | gatherQuantizedMM profiling | All MoE | Diagnostic | Medium | Tier 2 |
| 8 | Fused MoE dispatch kernel | All MoE | **2-5x decode** | Very High | Tier 3 |
| 9 | Quadratic attn for GDN | Qwen3.5 | **5-15x prefill** | Very High | Tier 3 |
| 10 | Jamba SSM fix | Jamba | Significant prefill | Low | Tier 4 |
| 11 | NemotronH benchmark | NemotronH | Unknown | Medium | Tier 4 |
