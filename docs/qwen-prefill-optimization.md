# Qwen Prefill Optimization Log

## Summary

The native prefill bridge (generic_prefill.cpp) for Qwen models matches Swift-only
prefill performance within ~3% across all context sizes. The bridge overhead from
eval barriers, embedding workarounds, and KV cache management is minimal compared
to the raw compute (quantized matmuls, SDPA) which dominates wall time.

Two clean-up optimizations were applied:
1. **Embedding simplification**: Removed unnecessary `add-zero + eval` copies in
   `make_embedding()`. Swift arrays already have pinned GPU buffers, so direct
   reference is sufficient (eval still called to ensure pinning).
2. **MoE routing fix**: Changed expert selection to use raw gate logits
   (`argpartition(-gates)`) instead of post-softmax scores, matching the Swift
   implementation. Functionally equivalent (softmax is monotonic) but avoids
   unnecessary `negative(softmax + 0)` graph nodes.

## Baseline Results

All measurements on Apple Silicon, release build, no background processes.
Numbers from back-to-back runs (second benefits from warm cache).

### Qwen2.5-3B-Instruct-4bit (dense, 36 layers, qwen2)

| Tokens | Bridge (t/s) | Swift (t/s) | Delta |
|--------|-------------|-------------|-------|
| 128    | 1861        | 1843        | +1.0% |
| 256    | 2158        | 2110        | +2.3% |
| 512    | 2290        | 2353        | -2.7% |
| 1024   | 2395        | 2473        | -3.2% |
| 2048   | 2412        | 2494        | -3.3% |
| 4096   | 2315        | 2409        | -3.9% |

### Qwen3-4B-4bit (dense, 36 layers, qwen3)

| Tokens | Bridge (t/s) | Swift (t/s) | Delta |
|--------|-------------|-------------|-------|
| 128    | 1314        | 1295        | +1.5% |
| 256    | 1476        | 1528        | -3.4% |
| 512    | 1593        | 1635        | -2.6% |
| 1024   | 1641        | 1684        | -2.6% |
| 2048   | 1628        | 1682        | -3.2% |
| 4096   | 1532        | 1592        | -3.8% |

### Qwen3-Coder-30B-A3B MoE (48 layers, 128 experts, 6-bit, qwen3_moe)

| Tokens | Bridge (t/s) | Swift (t/s) | Delta |
|--------|-------------|-------------|-------|
| 128    | 1054        | (*)         |       |
| 256    | 1386        | (*)         |       |
| 512    | 1543        | (*)         |       |
| 1024   | 1690        | (*)         |       |

(*) Swift-only MoE baseline not obtained due to extreme model loading time
(>30min for 18GB model with 48 x 128 expert weights). Expected to match bridge
within ~3% since both use the same underlying MLX ops (gather_qmm, SDPA).

## Architecture Analysis

### Generic bridge overhead sources (vs Gemma v2 specialized bridge):
1. **Eval barriers every 8 layers** -- required to prevent Metal allocator from
   reclaiming weight buffers. Removing them degrades correctness.
2. **Per-layer weight eval during build** -- pins GPU buffers for 36-48 layer
   models. Required for correctness.
3. **Post-forward KV eval** -- syncs all KV caches after forward. May overlap
   with chunk-end eval but removing it doesn't help (< 1ms).
4. **Embedding ownership workaround** -- `add-zero + eval` was unnecessary.
   Simplified to direct reference + eval (neutral perf impact).

### Why the bridge can't beat Swift for Qwen:
- Swift's `prepare()` already processes tokens in a single chunk (up to 4096 tokens)
  with one `asyncEval` at the end. This is effectively the same strategy as the bridge.
- The compute is dominated by `quantized_matmul` (dense layers) and `gather_qmm`
  (MoE layers), which are the same Metal kernels in both paths.
- The bridge adds ~3% overhead from eval barrier synchronization (4 eval points
  per 36-layer model), but this prevents memory corruption.

### MoE compute profile (Qwen3-Coder-30B):
- Attention: 19M params/layer (7% of compute)
- MoE: 269M active params/layer (93% of compute, top-8 of 128 experts)
- Theoretical roofline at 150 GB/s: ~14.8K t/s at 1024 tokens
- Measured: 1.69K t/s = 11% of roofline (gather_qmm cache locality limits)

## Optimizations Tested

| Optimization | Impact | Status |
|-------------|--------|--------|
| Remove embedding add-zero copy | Neutral | Applied (cleaner code) |
| MoE routing: use raw gates for argpartition | Neutral | Applied (matches Swift) |
| Remove mid-forward eval barriers | **Regression** (allocator corruption) | Reverted |
| Remove per-layer build eval | **Regression** (weight buffer reclaimed) | Reverted |
| Remove post-forward KV eval | **Regression** (timing measurement artifact) | Reverted |
| Increase EVAL_CADENCE (8 -> 36) | Neutral | Reverted (no benefit) |
| Remove eval barriers entirely | **-20% decode regression** | Reverted |

## Conclusion

The generic prefill bridge achieves near-parity with Swift for Qwen models.
Further optimization requires either:
1. **MLX kernel improvements** (faster gather_qmm, better MoE dispatch)
2. **Model-specific tricks** (fused gate+up projections, which require weight
   restructuring at load time)
3. **Hardware-specific tuning** (SDPA tiling for specific Apple Silicon variants)

None of these are bridge-side optimizations -- they require changes to the
underlying MLX framework.
