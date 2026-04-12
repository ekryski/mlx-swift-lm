# Option B: Fused MoE Kernel — Detailed Analysis

**Date**: 2026-04-06
**Status**: Not viable — architectural mismatch with Metal execution model

## Goal

Fuse gate_up + activation + down_proj into a single Metal kernel dispatch to eliminate one dispatch gap (~23us × 40 blocks = 0.92ms/token, ~5% decode improvement).

## Architecture Analysis

### Current Flow (FusedGateUpSwitchGLU)

```
Dispatch 1: gate_up_proj = gatherQuantizedMM(x[2048], W[1024, 2048]) → [1024]
CPU: split + silu(gate) * up → activated[512]
Dispatch 2: down_proj = gatherQuantizedMM(activated[512], W[2048, 512]) → [2048]
```

### Why True Fusion Is Not Viable

The gate_up and down projections have different dimensions:
- gate_up: input=2048, output=1024
- down: input=512, output=2048

The `qmv_fast_impl` kernel parallelizes along the OUTPUT dimension. Grid = (M, ceil(N/8), B):
- gate_up: 128 threadgroups per (token, expert)
- down: 256 threadgroups per (token, expert)

**Metal does not support inter-threadgroup barriers within a single dispatch.** To fuse both stages, ALL gate_up work must complete before down_proj starts. The only way to guarantee this is to compute BOTH stages within a single threadgroup.

### Single-Threadgroup Cost

Each threadgroup computes 8 output elements per iteration (results_per_simdgroup=4, num_simdgroups=2).

- gate_up: 1024 outputs / 8 = 128 iterations
- down: 2048 outputs / 8 = 256 iterations
- Total: 384 sequential iterations in one threadgroup

vs. current: 384 threadgroups × 1 iteration each, running **in parallel**.

**Result**: Fused kernel is ~384x slower per dispatch, saving only ~23us of dispatch gap. Net impact: massive regression.

### Alternative: Activation-Only Fusion

Fuse just the activation (silu + multiply + split) into the gate_up kernel's output write:

```metal
// In qmv_fast_impl, at the output write (line 811):
// Instead of: y[row] = result[row]
// Do: y[row] = (row < hidden/2) ? silu(result[row]) * y[row + hidden/2] : result[row]
```

This doesn't work either because:
1. Each threadgroup computes 8 independent output rows
2. Rows 0-511 (gate) and 512-1023 (up) are computed by DIFFERENT threadgroups
3. The silu(gate[i]) * up[i] operation requires both values, which live in different threadgroups

### What Would Work

A completely different kernel architecture:
1. **One thread per output element of activated[512]**
2. Each thread computes the FULL gate value and FULL up value for its index
3. This means each thread does TWO full inner products of dimension 2048
4. Then applies activation in-register
5. Then... we still can't do down_proj because activated[512] is distributed across 512 threads in different threadgroups

**Conclusion**: The dimensional asymmetry between gate_up (output=1024) and down (input=512, output=2048) makes true fusion impossible with Metal's non-cooperative threadgroup model.

## Recommendation

Option B provides ~5% gain at prohibitive complexity. The architectural analysis shows it's not just difficult — it's fundamentally incompatible with Metal's parallel execution model for these dimensions.

The existing FusedGateUpSwitchGLU (fusing gate + up into one dispatch) already captures the easy win. Further fusion requires architectural changes to either:
1. Metal's execution model (cooperative threadgroups, not available)
2. The MoE architecture itself (equal input/output dims, not realistic)

**Skip Option B entirely.**
