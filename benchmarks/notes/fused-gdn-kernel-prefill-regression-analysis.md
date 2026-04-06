# Fused GDN Kernel: Prefill Regression & Decode/Prefill Split

**Date**: 2026-04-05
**Model**: Qwen3.5-35B-A3B 4-bit, M1 Max

## Background

The GatedDeltaNet Metal kernel (`gated_delta_step`) was extended with a fused variant
(`gated_delta_step_fused`) that absorbs 4 operations into the kernel:

1. `rmsNorm(q)` with invScale² scaling — SIMD reduction across 32 lanes
2. `rmsNorm(k)` with invScale scaling — SIMD reduction across 32 lanes
3. `sigmoid(b)` → beta — per-head scalar
4. `computeGatedDeltaG(aLog, a, dtBias)` → g — exp + softplus per-head

These were previously 4-6 separate Metal encoder dispatches per GDN layer × 30 layers
= ~120-180 dispatches eliminated.

## The Tradeoff

| Metric | Original Kernel | Fused Kernel | Change |
|--------|----------------|-------------|--------|
| Decode @ 1024 | 48.7 tok/s | **51.3 tok/s** | **+5%** |
| Prefill @ 1024 | 481.8 tok/s | **422.7 tok/s** | **-12%** |
| Decode @ 32K | 34.9 tok/s | **37.2 tok/s** | **+7%** |
| Prefill @ 32K | 413.6 tok/s | **389.1 tok/s** | **-6%** |

**Decode improved +4-7%, but prefill regressed -6-12%.**

## Root Cause: Register Pressure vs Dispatch Overhead

The fused kernel adds per-thread register usage:

```
Original kernel per thread:
  float state[n_per_t];     // 4 floats (n_per_t = Dk/32 = 128/32 = 4)
  float kv_mem, out, delta; // 3 floats
  Total: ~7 floats + loop vars ≈ ~10 registers

Fused kernel per thread:
  float state[n_per_t];     // 4 floats
  float q_vals[n_per_t];    // 4 floats (NEW — stores raw q for norm computation)
  float k_vals[n_per_t];    // 4 floats (NEW — stores raw k for norm computation)
  float q_sum_sq, k_sum_sq; // 2 floats (NEW — for rmsNorm reduction)
  float q_rms, k_rms;       // 2 floats (NEW — norm values)
  float a_val, b_val, ...;  // 4 floats (NEW — g/beta intermediates)
  Total: ~20 floats + loop vars ≈ ~24 registers
```

The fused kernel uses **~2.4x more registers per thread**. On Apple GPU:
- Registers are shared across threadgroups occupying the same compute unit
- More registers per thread = fewer threadgroups can run concurrently = lower occupancy
- Lower occupancy = less latency hiding = worse throughput for parallel workloads

**Why this only hurts prefill, not decode:**

- **Decode (T=1)**: Only 1 token to process. The kernel runs once per (head, dv) pair.
  GPU occupancy doesn't matter much — the bottleneck is dispatch overhead between
  encoders. Fusing eliminates dispatches → net win.

- **Prefill (T=1024+)**: Many tokens processed in the sequential loop. The kernel runs
  for T timesteps per dispatch. Higher occupancy helps because the GPU can hide memory
  latency by switching between threadgroups while one waits for data. Fewer concurrent
  threadgroups = more stalls = lower throughput.

## Solution: Decode/Prefill Kernel Split

Use the fused kernel for decode (T=1) and the original kernel for prefill (T>1):

```swift
if S == 1 {
    // Decode: fused kernel (fewer dispatches, register pressure doesn't matter)
    (out, state) = fusedGatedDeltaUpdate(qRaw: q, kRaw: k, ...)
} else {
    // Prefill: original kernel (better occupancy for parallel processing)
    let qNormed = invScale² * rmsNorm(q, ...)
    let kNormed = invScale * rmsNorm(k, ...)
    (out, state) = gatedDeltaUpdate(q: qNormed, k: kNormed, ...)
}
```

## Results: Best of Both Worlds

| Metric | Original | Fused Only | **Split** |
|--------|----------|-----------|-----------|
| Decode @ 1024 (no-quant) | 48.7 | 51.3 | **51.7** |
| Prefill @ 1024 (no-quant) | 481.8 | 422.7 | **469.1** |
| Decode @ 1024 (turbo4v2) | 49.9 | 51.9 | **51.4** |
| Prefill @ 1024 (turbo4v2) | 489.0 | 420.7 | **473.8** |
| Decode @ 32K (turbo4v2) | 37.6 | 40.0 | **40.3** |
| Prefill @ 32K (turbo4v2) | 482.0 | 437.4 | **486.7** |

The split recovers prefill to near-original levels while keeping the decode improvement.

## General Principle

**Kernel fusion has different ROI for decode vs prefill:**

- **Decode** is dispatch-overhead-limited (43% of wall time is inter-encoder gaps).
  Fusing operations reduces dispatches → direct speedup. Register pressure is
  irrelevant because GPU occupancy doesn't help single-token processing.

- **Prefill** is compute/occupancy-limited. The GPU processes many tokens in parallel.
  Higher register usage reduces occupancy, limiting the GPU's ability to hide memory
  latency through threadgroup switching. Separate dispatches with lower register
  pressure per kernel can achieve higher throughput.

**Implication for future kernel fusion work:**
Always benchmark both decode AND prefill. A fused kernel that improves decode may
regress prefill. The solution is conditional dispatch: fused path for T=1, unfused
for T>1. This pattern applies to any custom Metal kernel in the inference pipeline.
