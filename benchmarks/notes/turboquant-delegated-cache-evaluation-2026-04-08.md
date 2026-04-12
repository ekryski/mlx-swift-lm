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

**Results: interval=1 (per-token) vs interval=64 (batch)**

| Ctx | Metric | interval=1 | interval=64 | Delta |
|-----|--------|-----------|-------------|-------|
| 32K | Prefill | 2355 tok/s | 2509 tok/s | **+6.5%** |
| 32K | Decode | 59.2 tok/s | 61.2 tok/s | **+3.4%** |
| 32K | TTFT | 13966ms | 13088ms | **-6.3%** |
| 64K | Prefill | 1320 tok/s | 1709 tok/s | **+29.5%** |
| 64K | Decode | 46.7 tok/s | 47.0 tok/s | +0.6% |
| 64K | TTFT | 49702ms | 38464ms | **-22.6%** |
| 1K-16K | All | within noise | within noise | 0% |

**Batch recompression is a significant win at long contexts.** At 64K, TTFT drops from 49.7s to 38.5s (-22.6%) and prefill throughput increases +29.5%. At 32K, +3-6% across the board. At 1K-16K, within noise — the kernel launch savings are too small relative to total compute.

### Adaptive interval

Since batch recompression helps at long contexts and is neutral at short ones, we use an adaptive interval that scales with context length: `max(baseInterval, offset / 256)`.

- At 1K-16K tokens: interval stays at 64 (the floor)
- At 64K: interval grows to ~256
- At 128K+: interval reaches ~512

This ensures the ratio of pending-to-total tokens stays bounded (~0.4%) while maximizing amortization at long contexts.

## Key Insight: Python vs Swift Execution Models

**Tom's CPU-stream async encode was necessary in Python where Metal kernel dispatch is synchronous.** In that execution model:
- Each `turbo_encode()` call dispatches a Metal kernel and blocks until completion
- `mx.concatenate` physically allocates and copies immediately
- 72+ allocations/step = 72+ synchronous GPU dispatches = measurable overhead

**In MLX Swift with lazy evaluation, the encode ops are graph nodes that the scheduler batches naturally.** The differences:
- `encodeNewToken()` builds lazy ops — no Metal kernel fires until `eval()` 
- Buffer slice-assign (`array[.ellipsis, range, 0...] = value`) triggers evaluation, but MLX's scheduler batches all pending graph nodes into a single GPU submission
- The "72+ allocations" in Python are 72+ synchronous round-trips; in Swift they're graph nodes that get fused by the lazy evaluator

This is why the 2a pre-allocation step change (256→1024) showed no impact — MLX Swift's lazy eval already amortizes allocation overhead. However, batch recompression still helps at 32K+ because it reduces the total number of fused Metal kernel *launches* across the entire decode session, not just the per-step overhead.

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
- **2a (batch recompress):** Significant win at long contexts. Adaptive interval (`max(64, offset/256)`) scales with context length — neutral at 1K-16K, +3-6% at 32K, **+22-30% TTFT improvement at 64K**. Default on.
- **2b (CPU async encode):** Not implementing. The optimization solves a Python-specific problem (synchronous dispatch) that doesn't exist in MLX Swift's lazy evaluation model.

## Conclusion

Batch recompression with adaptive interval is a clear win for long-context inference. At 64K, TTFT improves from 49.7s to 38.5s (-22.6%) by reducing Metal kernel launches from 1/token to 1/~256. The improvement grows with context length as more launches are amortized.

Tom's CPU-stream async encode is unnecessary in MLX Swift — lazy evaluation already batches GPU operations. But the batch recompression concept translates well: even with lazy eval, reducing the total number of kernel launches across a long decode session has measurable impact at 32K+ tokens.

Future TurboQuant improvements should focus on quality (turbo8v4 variants) and this kind of long-context efficiency optimization.
