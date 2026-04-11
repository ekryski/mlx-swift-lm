# Optimization Change Log

Track each optimization attempt, benchmark results, and expert analysis.

---

## Entry 1: Dtype Leak Fixes (Step 2)
**Date**: 2026-04-11
**Branch**: `ek/tom-eric-moe-tuning`
**Baseline**: April 10 benchmarks (post-PR14)

### Changes Made
1. **KVCache.swift** (lines 1718, 1722, 1731): `MLXArray(Float.leastNormalMagnitude)` → `MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype)` — prevents float32 promotion of attention scores on every decode token
2. **TurboQuantKVCache.swift** (lines 1235, 1242, 1322): Same mask fill fix
3. **TurboQuantKVCache.swift** (lines 1228, 1272): `MLXArray(scale)` → bare `scale` (Float) — prevents float32 promotion of attention scores in separated path
4. **TurboQuantKVCache.swift** (lines 300, 340, 349, 478, 521, 538): Various `MLXArray(Float(...))` → dtype-aware scalars — prevents float32 promotion in encoding/init paths

### Benchmark Results (Gemma4 E2B 4bit, M1 Max, --ppl --kld --think --quick)

**no-quant** (apples-to-apples comparison, same flags):

| Context | Apr 10 Prefill | Apr 11 Prefill | Delta | Apr 10 Decode | Apr 11 Decode | Delta |
|---------|---------------|---------------|-------|--------------|--------------|-------|
| 128     | 669.3         | 730.2         | **+9.1%** | 78.3 | 78.9 | +0.8% |
| 1024    | 1888.9        | 1827.4        | -3.3% | 76.7 | 75.9 | -1.0% |
| 4096    | 1492.0        | 1496.9        | +0.3% | 75.1 | 74.8 | -0.4% |

**turbo4v2** (Apr 10 baseline ran without PPL/KLD/think flags — not directly comparable):

| Context | Apr 11 Prefill | Apr 11 Decode | Apr 10 Prefill* | Apr 10 Decode* |
|---------|---------------|--------------|----------------|---------------|
| 128     | 608.9         | 77.2         | 712.2*         | 104.1*        |
| 1024    | 1752.7        | 75.2         | 2118.7*        | 100.4*        |
| 4096    | 1418.4        | 73.9         | 1481.3*        | 97.3*         |
| 32768   | 392.1         | 59.6         | 419.5*         | 75.2*         |

*Apr 10 turbo4v2 ran WITHOUT --ppl --kld --think. The decode difference (~25%) is largely PPL/KLD overhead, not a regression from dtype fixes.

**Assessment**: no-quant results are roughly neutral (within noise). turbo4v2 can't be compared directly due to different benchmark flags. The dtype fixes are correctness improvements — they prevent float32 promotion which wastes memory and compute. Need to run turbo4v2 without PPL flags for a clean comparison, or wait for the combined build (dtype + PPL fix + compiledNormResidual) to evaluate everything together.

---

## Entry 2: Combined Build (dtype + PPL deferred sync + compiledNormResidual)
**Date**: 2026-04-11
**Branch**: `ek/tom-eric-moe-tuning`

### Changes Made (in addition to Entry 1)
1. **Evaluate.swift**: Deferred phase classification from `convertToToken` to `next()` — eliminates extra GPU→CPU sync (~4ms/token) when PPL+thinking is enabled. Phase tracking piggybacked on existing `.item()` call in `next()`.
2. **Evaluate.swift**: Added `finalizePerplexity()` for end-of-generation cleanup of deferred logprobs.
3. **Evaluate.swift**: Batch extraction of per-token logprobs for KLD (deferred `.item()` calls).
4. **Gemma4.swift (LLM)**: Re-enabled `compiledNormResidual` at 3 call sites (postAttention, postFFN-MoE, postFFN-dense). Fixed `1.0 + weight` Gemma bias offset in compiled closure.

### Benchmark Results
*(Pending)*

### Expert Analysis
*(Fill in if neutral/regression)*
