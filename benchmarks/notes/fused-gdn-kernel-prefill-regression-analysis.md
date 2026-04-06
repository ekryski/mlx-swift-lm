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

---

## Quadratic Attention for GatedDeltaNet (Experimental, GDN_QUADRATIC=1)

**Date**: 2026-04-05
**Commit**: 2a3dcab

### Background

Attempted to parallelize the sequential GatedDeltaNet recurrence using the
"quadratic attention" formulation from Yang et al. NeurIPS 2024 ("Parallelizing
Linear Transformers with the Delta Rule over Sequence Length").

The approach: express the GatedDeltaNet recurrence as a causal attention-like
operation with decay-weighted scores, computed via parallel matmul instead of
sequential kernel iteration.

### Implementation

Linearized approximation of the delta rule:
```
S_t = g_t * S_{t-1} + β_t * k_t * (v_t - k_t^T * S_{t-1})^T
```

Drops the `k_t^T * S_{t-1}` correction term to get a linear recurrence:
```
S_t ≈ g_t * S_{t-1} + β_t * k_t * v_t^T
```

This allows expressing outputs via a causal attention matrix:
```
y_t = Σ_{s≤t} (q_t^T * cumDecay(t,s) * β_s * k_s) * v_s + q_t^T * cumDecay(t,0) * S_prev
```

Processed in chunks of C=64 tokens. Within each chunk, the [C, C] attention
matrix is computed via Q @ K^T matmul (fully parallel). Between chunks, state
is propagated sequentially (T/C iterations).

### Results (Qwen3.5-35B 4-bit, M1 Max)

| Metric | Sequential Kernel | Quadratic Attention |
|--------|------------------|---------------------|
| Prefill @ 128 | 238.6 tok/s | **115.9 tok/s (-51%)** |
| Quality (PPL) | 1.3-1.7 | **2.0-2.2 (degraded)** |
| Decode | 51.6 tok/s | 51.3 tok/s (same) |

### Why It's Slower

For Qwen3.5's dimensions (Dk=128, Dv=128, Hv=32, C=64):

**Quadratic attention per chunk**: O(C² × D) = 64² × 128 = 524K FLOPs per head
Plus: log-decay computation, causal masking, inter-chunk state matmul

**Sequential kernel per chunk**: O(C × D) = 64 × 128 = 8.2K FLOPs per thread × 32 threads
= ~262K FLOPs per head, with SIMD-optimized memory access patterns

The quadratic approach is ~2x MORE compute per head, and the matmul operations
([32, 64, 128] @ [32, 128, 64] → [32, 64, 64]) create large intermediate tensors
that stress GPU memory bandwidth.

### Why Quality Degrades

The linearized approximation drops `k_t^T * S_{t-1}` from the delta formula.
This "error correction" term is what distinguishes DeltaNet from standard linear
attention — it allows the model to "forget" incorrect associations by subtracting
the current state's prediction before adding the new key-value pair.

Without it, the model accumulates stale associations, leading to higher perplexity
(2.0-2.2 vs 1.3-1.7 baseline).

The full WY representation (Yang et al. NeurIPS 2024) preserves this correction
by expressing the product of Householder-like transforms as low-rank matrices:

```
Π_{m=0}^{t} (g_m * I - β_m * k_m * k_m^T) = cumG * (I - W * Y^T)
```

But the WY construction is still sequential (O(C) steps of O(Dk²) per head),
adding more compute than it saves for Dk=128.

### When Quadratic Attention WOULD Help

The crossover point depends on the ratio of Dk² (sequential kernel per-step cost)
to C (chunk size for quadratic attention):

| Head Dim (Dk) | Sequential O(C × Dk) | Quadratic O(C² × Dk) | Winner |
|---------------|---------------------|---------------------|--------|
| 32 | C × 32 | C² × 32 | **Quadratic** for C < Dk |
| 64 | C × 64 | C² × 64 | Quadratic for C < 64 |
| 128 | C × 128 | C² × 128 | **Sequential** for typical C |
| 256 | C × 256 | C² × 256 | Sequential (large Dk dominates) |

**Models that would benefit:**
- **Gemma 4 26B A4B**: Has GatedDeltaNet — need to check head dimensions though
- Models with Dk ≤ 64 where the quadratic matmul is cheaper
- Models with longer sequences (T > 8192) where chunk-level parallelism matters more

### Open Questions

1. **Fused quadratic kernel**: Would implementing the quadratic attention as a
   custom Metal kernel (instead of MLX matmul ops) reduce memory pressure enough
   to change the tradeoff? The current implementation materializes [C, C] attention
   matrices in GPU memory. A fused kernel could compute scores + weighted sum
   in-register without materializing the full matrix.

2. **Alternative algorithms**:
   - **Chunkwise WY representation**: Full delta correction via WY matrices, O(C × Dk)
     sequential WY construction + O(C × Dk) parallel output. More compute but exact.
   - **Semi-parallel hybrid**: Use quadratic attention for the first few positions in
     each chunk (where the sequential kernel wastes GPU occupancy) and sequential
     kernel for the rest (where it's more efficient).
   - **Hierarchical chunking**: Nested chunks of different sizes (e.g., C=16 inner,
     C=256 outer) to balance parallelism vs compute overhead.

### Fused Quadratic Kernel (Potential Improvement)

The current quadratic attention uses MLX matmul ops that materialize large
intermediate tensors ([B, Hv, C, C] attention matrix = 8MB+). A fused Metal
kernel could compute scores + weighted sum in-register:

```metal
// Pseudocode for fused quadratic GDN kernel
// Grid: (32, Dv, B * Hv * C)  — one output per (batch, head, query_pos, value_dim)

for (uint s = 0; s <= query_pos; s++) {
    float score = dot(q[query_pos], k[s]) * cumDecay[query_pos, s] * beta[s];
    output[dv_idx] += score * v[s][dv_idx];  // accumulate in register
}
```

This is structurally identical to the TurboFlash attention kernel (pass 1)
which computes Q·K scores from quantized keys and accumulates V·attn_weights
without materializing the score matrix. The key difference: decay weighting
instead of softmax normalization.

Expected improvement: eliminating the [C, C] intermediate would reduce memory
pressure from ~16MB to ~0 (all in-register). Whether this changes the speed
tradeoff vs the sequential kernel depends on:
- C=64: the scoring loop is 64 iterations × 128 dims = 8K FLOPs per output element
- Sequential kernel: 1 iteration × ~20 FLOPs per thread
- Fused quadratic is still ~400x more FLOPs per output element

So even with perfect memory efficiency, the quadratic approach does fundamentally
more work than the sequential kernel. The win would only come from better GPU
utilization (parallel across all C query positions vs sequential through T steps).

### Alternative Algorithms

1. **Chunkwise WY representation** (Yang et al. NeurIPS 2024):
   Full delta correction via WY matrices. Sequential WY construction is O(C × Dk²)
   but the output computation is O(C × Dk) parallel. Exact (no quality loss).
   However, at Dk=128 the WY construction is expensive: C × 128² = 1M FLOPs per head.

2. **Tiled sequential with overlap**:
   Instead of one big sequential kernel dispatch, tile into small tiles (e.g., 8 steps)
   with state checkpointing. Multiple tiles can execute on different GPU compute units.
   This doesn't reduce total compute but improves GPU utilization by distributing
   the sequential work across more SIMD groups.

3. **Approximate parallel scan with correction**:
   Use the linearized approximation (quadratic attention) as a FIRST PASS, then run
   a correction pass using the sequential kernel with the approximate state as starting
   point. If the approximation is close, the correction converges in fewer iterations.
   This is related to iterative refinement in numerical linear algebra.

4. **Hardware-specific: Apple Neural Engine (ANE)**:
   The ANE can run matmul operations in parallel with the GPU. If the quadratic
   attention matmuls could be dispatched to ANE while the GPU handles other layers,
   the latency could be hidden. Requires MLX ANE support (not currently available).

5. **Reduced-rank delta rule**:
   Approximate the delta correction with a low-rank update instead of full k*k^T.
   This reduces the sequential kernel's per-step cost and might make the quadratic
   formulation competitive. However, this changes the model's mathematics.

### Applicability to Other Models

**Gemma 4 26B A4B IT**: If it uses GatedDeltaNet-style layers, check:
- Head dimension (Dk): if ≤ 64, quadratic attention becomes viable
- Number of GDN layers: more layers = more dispatch savings from parallelization
- Whether the model uses the delta rule or simpler linear attention

**General rule**: The sequential kernel wins when Dk is large (128+) because
per-step compute is proportional to Dk. The quadratic approach wins when Dk is
small (32-64) because the [C, C] attention matrix compute is independent of Dk
(only the dot product dimension depends on Dk).

### Code Location

`Libraries/MLXLLM/Models/GatedDelta.swift`:
- `gatedDeltaAttn()` — quadratic attention implementation
- Enabled via `GDN_QUADRATIC=1` environment variable
- NOT default — the sequential Metal kernel is faster for Qwen3.5's dimensions
