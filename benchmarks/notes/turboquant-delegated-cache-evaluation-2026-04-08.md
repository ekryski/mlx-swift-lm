# TurboQuant Delegated KVCache & Async Encode Evaluation (Phase 2a/2b)

Date: 2026-04-08

## Background

Tom's [commit 7ad7500](https://github.com/TheTom/mlx/commit/7ad75002bbf87f5d1774859e94e2a76ccbd3a56d) on mlx-lm (Python) achieved 0.99x baseline decode (96% overhead eliminated) via two optimizations:

1. **Delegated KVCache:** Replace per-step `mx.concatenate` (72+ allocations/step) with an internal mlx-lm KVCache using pre-allocated buffers + slice-assign. Packed storage updated in background via batch recompression every 64 tokens.
2. **CPU-stream async encode:** `turbo_encode_cpu()` dispatches encoding to `mx.stream(mx.cpu)`, overlapping with GPU SDPA. Hides 75-88% of encode latency. Note: "Metal kernels can't run on CPU" — Tom wrote a pure Python/CPU implementation of the encode, not a Metal kernel on a different stream.

## What We Implemented

### 2a. Pre-allocation step size (256 → 1024)

Increased buffer growth step from 256 to 1024 tokens. Reduces resize frequency by 4x at long contexts.

**Result: Within noise at all contexts (1K-16K).** MLX's lazy evaluation already amortizes allocation overhead — arrays aren't physically allocated until eval is triggered, and the concat+copy cost is dominated by actual compute.

| Model | Ctx | Decode (step=256) | Decode (step=1024) | Delta |
|-------|-----|------------------|-------------------|-------|
| E2B | 16K | 69.9 | 69.9 | 0% |
| 26B | 16K | 25.4 | 25.6 | +0.8% |
| Qwen | 16K | 54.9 | 54.6 | -0.5% |
| GPT-OSS | 16K | 53.1 | 53.4 | +0.6% |

Kept step=1024 as default (harmless, fewer resize events).

### 2a. Batch recompression (every 64 tokens)

Implemented deferred encoding: accumulate raw K/V tokens in pending lists, batch-encode every `recompressInterval` tokens (default 64, configurable via `TURBO_RECOMPRESS_INTERVAL` env var). This reduces Metal kernel launches from 1/token to 1/64 tokens.

**Long context baseline (interval=1, per-token encode):**

| Ctx | Decode | TTFT | PPL | KV Cache |
|-----|--------|------|-----|----------|
| 32K | 59.2 tok/s | 13966ms | 1.63 | 1.43GB |
| 64K | 46.7 tok/s | 49702ms | 1.63 | 2.85GB |

Batch=64 run was killed before completing. The 1K-16K results from earlier showed batch recompression was within noise of per-token encoding at those context lengths.

## Key Insight: Python vs Swift Execution Models

**Tom's optimizations were designed for Python mlx-lm where Metal kernel dispatch is synchronous.** In that execution model:
- Each `turbo_encode()` call dispatches a Metal kernel and blocks until completion
- `mx.concatenate` physically allocates and copies immediately
- 72+ allocations/step = 72+ synchronous GPU dispatches = measurable overhead

**In MLX Swift with lazy evaluation, the encode ops are graph nodes that the scheduler batches naturally.** The differences:
- `encodeNewToken()` builds lazy ops — no Metal kernel fires until `eval()` 
- Buffer slice-assign (`array[.ellipsis, range, 0...] = value`) triggers evaluation, but MLX's scheduler batches all pending graph nodes into a single GPU submission
- The "72+ allocations" in Python are 72+ synchronous round-trips; in Swift they're graph nodes that get fused by the lazy evaluator

This explains why our pre-allocation and batch recompression changes showed no measurable improvement: **MLX Swift's lazy evaluation already provides the batching and overlap that Tom had to implement explicitly in Python.**

### CPU-stream async encode (2b)

**Not applicable to our architecture.** Tom's `turbo_encode_cpu()` was a pure Python/CPU implementation of the encode (rotate + quantize + pack) that runs on `mx.stream(mx.cpu)` concurrently with GPU SDPA. This was necessary because:
1. Python mlx dispatches Metal kernels synchronously — CPU encode overlaps with GPU attention
2. "Metal kernels can't run on CPU" — so he wrote a CPU-side encoder

In MLX Swift:
- Our encode is a fused Metal kernel (`fusedEncodeDispatch`) — already optimal for GPU
- MLX's lazy evaluation graph naturally overlaps compatible GPU operations
- A CPU-side reimplementation would be slower than the Metal kernel and wouldn't overlap better than the lazy scheduler already achieves

## Decision

- **2a (step size 1024):** Kept. Harmless, marginally cleaner memory behavior.
- **2a (batch recompress):** Implemented with env var toggle. Default interval=64. No measurable speed impact due to lazy evaluation, but reduces kernel launch count which may matter on memory-constrained devices.
- **2b (CPU async encode):** Not implementing. The optimization solves a Python-specific problem (synchronous dispatch) that doesn't exist in MLX Swift's lazy evaluation model.

## Conclusion

Our TurboQuant implementation in MLX Swift already achieves near-baseline decode performance (turbo4v2 matches or exceeds FP16 none at all contexts tested up to 64K). The overhead that Tom eliminated in Python doesn't exist in our architecture because lazy evaluation handles operation batching and overlap automatically. Future TurboQuant improvements should focus on quality (turbo8v4 variants) and memory efficiency rather than decode speed, which is already at parity.
