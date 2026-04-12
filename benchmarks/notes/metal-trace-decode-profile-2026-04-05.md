# Metal System Trace: Qwen3.5-35B-A3B Decode Profile

**Date**: 2026-04-05
**Hardware**: Apple M1 Max (applegpu_g13s), 64GB RAM
**Model**: Qwen3.5-35B-A3B 4-bit, no-quant KV
**Context**: 4096 tokens prefill, ~70 decode tokens captured in trace window
**Tool**: `xcrun xctrace record --template 'Metal System Trace' --time-limit 15s --attach <PID>` (zero overhead)
**Note**: Trace attached after model loading. The 10.5s window covers prefill (~8.6s at
502 tok/s) + early decode (~1.5s at ~45 tok/s). Encoder durations are uniform across
both phases (~160us avg), confirming MLX dispatches prefill and decode operations
with the same per-operation granularity.

## Raw Trace Statistics

- Total Metal encoder events: 31272
- Trace window: 10.5s
- Total GPU time: 5045.5ms
- GPU utilization: 48%

## Events Per Second

| Second | Events | GPU Time (ms) | Notes |
|--------|--------|---------------|-------|
| t=0s | 6 | 9.7 | Model loading / warmup |
| t=1s | 78 | 20.0 | Model loading / warmup |
| t=2s | 3463 | 574.8 | Prefill start |
| t=3s | 3562 | 562.1 |  |
| t=4s | 3588 | 573.3 |  |
| t=5s | 3588 | 548.0 |  |
| t=6s | 3510 | 577.3 |  |
| t=7s | 3462 | 542.2 |  |
| t=8s | 3407 | 591.1 |  |
| t=9s | 3624 | 540.0 |  |
| t=10s | 2984 | 507.1 |  |

## Decode Phase Analysis

Decode phase identified as the last ~1.5s of the trace (t=9-10.5s), after the 4096-token
prefill completes at ~t=8.6s. The decode window contains 5,399 events across ~69 estimated
tokens (~45 tok/s × 1.5s).

**Important**: Encoder durations are uniform across prefill and decode phases (~160us avg),
so the per-token metrics below also apply to individual prefill operations through 40 layers.

| Metric | Value |
|--------|-------|
| Decode window | ~1.5s (last portion of 10.5s trace) |
| Est. decode tokens | ~70 |
| Metal encoders per token | 78 |
| GPU compute per token | 12.6ms |
| Wall clock per token | 22.2ms |
| **GPU utilization** | **57%** |
| Dispatch overhead per token | 9.6ms |
| Avg encoder duration | 160.5us |
| Avg inter-encoder gap | ~123us |

## Encoder Duration Distribution

| Duration Bucket | Count | % |
|----------------|-------|---|
| 10-50us | 8821 | 28.2% |
| 50-100us | 4807 | 15.4% |
| 100-500us | 17423 | 55.7% |
| 500us-1ms | 204 | 0.7% |
| 1-5ms | 17 | 0.1% |

## Top 10 Longest GPU Operations

| Duration | Time in Trace | Notes |
|----------|--------------|-------|
| 4363us (4.36ms) | t=0.388s | Likely shader JIT compilation |
| 4250us (4.25ms) | t=0.384s | Likely shader JIT compilation |
| 2744us (2.74ms) | t=1.886s |  |
| 1344us (1.34ms) | t=1.943s |  |
| 1270us (1.27ms) | t=1.888s |  |
| 1169us (1.17ms) | t=10.259s |  |
| 1156us (1.16ms) | t=1.890s |  |
| 1117us (1.12ms) | t=2.413s |  |
| 1112us (1.11ms) | t=6.888s |  |
| 1103us (1.10ms) | t=2.390s |  |

## Key Findings

1. **57% GPU utilization** — 43% of decode time is Metal dispatch/scheduling overhead
2. **78 encoders per token** — approximately 2 per layer (40 layers)
3. **160us avg encoder** — individual GPU operations are fast; overhead is between them
4. **~123us avg gap** — CPU-side command buffer prep between dispatches dominates overhead
5. Reducing encoder count from 78 to ~50 (via operation fusion) could save ~3.5ms/token → ~15% decode speedup

## Raw Data

- **CSV export**: `benchmarks/notes/metal-trace-raw-data-2026-04-05.csv` (31,272 rows, 967KB)
  - Columns: `start_ns, duration_ns, start_fmt`
  - Each row = one Metal encoder dispatch
- **Trace file**: Captured via `xcrun xctrace record --template 'Metal System Trace'`
  - Open in Xcode Instruments for visual GPU timeline

### Reproducing

```bash
# Start benchmark in background
MLX_BENCH_MODEL=qwen35-35b-a3b MLX_BENCH_METHOD=summarization \
  MLX_BENCH_KV=none MLX_BENCH_QUANT=4bit MLX_BENCH_CONTEXT=4096 \
  swift test -c release --skip-build --filter "benchmark" &

# Wait for model to load, find the test helper PID
sleep 15
PID=$(ps aux | grep "swiftpm-testing-helper" | grep -v grep | awk '{print $2}')

# Capture 15s Metal trace
xcrun xctrace record --template "Metal System Trace" \
  --output /tmp/trace.trace --time-limit 15s --attach $PID

# Export encoder data to XML
xcrun xctrace export --input /tmp/trace.trace \
  --xpath '/trace-toc/run/data/table[@schema="metal-application-encoders-list"]' \
  > /tmp/metal-encoders.xml
```
