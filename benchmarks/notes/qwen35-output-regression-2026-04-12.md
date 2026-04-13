# Qwen3.5-35B Output Quality Regression — Analysis & Fix

**Date**: 2026-04-12
**Status**: Partially fixed. No-quant works. TurboQuant still broken (separate issue).

## Root Cause: Non-Fused GDN Framework Kernel Bug

The framework `gated_delta_step` (non-fused) Metal kernel produces incorrect output.
Unit test (`GatedDeltaKernelTests`) confirmed:

| Config | Max Diff (vs ops) | Status |
|--------|:-----------------:|--------|
| Non-fused T=1 (Dk=128, Dv=128, Hk=8, Hv=8) | 0.25 | **BROKEN** |
| Non-fused T=1 (Dk=192, Dv=128, Hk=4, Hv=4) | 0.375 | **BROKEN** |
| Non-fused T=4 (Dk=128, Dv=128, Hk=16, Hv=32) | 48.0 | **CATASTROPHIC** |
| Fused T=1 (Dk=128, Dv=128, Hk=16, Hv=32) | 0.00024 | PASSES |

The non-fused variant is broken for ALL configs (not just GQA). The fused variant is correct.

## Fix Applied

The non-fused kernel is used during prefill (T>1). The fused kernel is used for decode (T=1).
**Workaround**: bypass the non-fused framework kernel and use the pure ops fallback
(`gatedDeltaOps`) for prefill. The ops fallback is slower but correct.

```swift
// In GatedDelta.swift gatedDeltaUpdate():
// Was: gatedDeltaKernel(q, k, v, g, beta, state, mask)  // framework non-fused
// Now: gatedDeltaOps(q, k, v, g, beta, state, mask)     // pure MLX ops
```

This restores correct output for no-quant (62 tok/s decode, coherent text).

## Remaining Issue: TurboQuant + Qwen3.5

With `--kv turbo4v2`, Qwen3.5 still produces garbage ("Thinking!!!!!...") even after
the GDN prefill fix. The TurboQuant codec's rotation matrices or quantization introduces
errors that corrupt the attention layer output, which then cascades into GDN layers.

This is a separate issue from the GDN kernel bug. With `--kv none` and `--kv affine4`,
output is correct.

### Sequential Context Crash (TurboQuant)

Running sequential contexts (128 → 4096) with turbo4v2 crashes with `Invalid Resource`.
This happens because Metal buffer references from the previous run's TurboQuantKVCache
persist after the cache is freed. Individual contexts work fine in isolation.
Pre-existing issue — was masked by the dtype crash before our fixes.

## Next Steps

1. **Fix the non-fused GDN Metal kernel** — the bug is in `gated_delta.metal` (non-fused
   variant). Compare carefully with the fused variant which is correct. The state update
   loop logic looks identical but produces different results — suspect a buffer pointer
   offset or stride mismatch.
2. **Fix TurboQuant + GDN interaction** — investigate why turbo4v2 corrupts attention
   output for Qwen3.5 (dtype promotion during lazy eval, invisible to Swift `.dtype`).
3. **Fix TurboQuant sequential crash** — Metal buffer lifetime management in
   `maybeQuantizeKVCache` when model is reused across contexts.
