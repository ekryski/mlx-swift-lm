# Option D: Expert-Aware L2 Weight Prefetching — Analysis

**Date**: 2026-04-06
**Status**: Already captured by existing sort mechanism

## Goal

Optimize memory access patterns in the gather kernel so consecutively-dispatched experts share L2 cache lines.

## Analysis

### L2 Cache Sizing

| Chip | L2 Cache | Active Expert Weights (Qwen3.5, 8×1.5MB) |
|------|----------|------------------------------------------|
| M1 Max | 48MB | 12MB — fits with room to spare |
| M1/M2 base | 12-16MB | 12MB — tight, may thrash |

### What Sort Already Does

The `gatherSort` function in SwitchLayers.swift reorders tokens by expert index. This means:
1. Consecutive elements in `rhs_indices` (expert IDs) are grouped by expert
2. Consecutive threadgroups (z-dimension) process the same expert
3. GPU cores sharing L2 naturally cache the expert weights for consecutive threadgroups

### Why Explicit Prefetching Doesn't Help

1. **Metal has no `prefetch()` intrinsic** — we can only simulate via volatile reads
2. **Threadgroup scheduling is not ordered** — Metal may execute threadgroups in any order
3. **Simulated prefetch (volatile read)** adds a memory read with no compute to hide behind
4. **The sort already ensures the BEST possible access pattern** for L2 locality

### What Could Help (but isn't L2 prefetching)

- **Weight layout reordering** (Morton/Z-order): Reorder expert weight storage so frequently co-selected experts are adjacent in memory. This is the profile-guided expert reordering from the morton-order spec. It's a higher-level optimization than L2 prefetching.
- **Larger threadgroups**: Processing more output rows per threadgroup means more weight data reuse within a threadgroup. But the current 8 rows per threadgroup is already a good balance.

### Hardware-Dependent Gating

For the record, if explicit prefetching were viable:
```swift
// Query L2 at model load time
let l2Size = sysctlValue("hw.l2cachesize") ?? 12_582_912
let activeExpertBytes = topK * perExpertWeightBytes
let enablePrefetch = activeExpertBytes < Int(Double(l2Size) * 0.6)
```

## Conclusion

**No implementation needed.** The existing sort mechanism already provides the L2 locality benefit. Explicit prefetching is not feasible with Metal's execution model. The theoretical next step (profile-guided expert reordering) is a separate optimization tracked in `morton-order-expert-reorder-spec.md`.
