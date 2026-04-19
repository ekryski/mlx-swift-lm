# ICB Tok/s Projection + Gemma 4 E2B Baseline — 2026-04-17

Follow-up to [icb-first-measurement-2026-04-17.md](icb-first-measurement-2026-04-17.md).
Captures the baseline decode throughput we'd be comparing against once the
ICB decode-loop integration lands, and enumerates what's still missing.

## Baseline: 1024-ctx summarization (no ICB, `ek/metal-icb-prototype`)

Hardware: M1 Max, 64 GB, macOS 15.7.4. Method: `summarization`, context 1024,
1008–1024 prompt tokens, 4bit weights. Rows live in
[m1-max-64gb-2026-04-17.md](../m1-max-64gb-2026-04-17.md).

| Model | KV | Prefill tok/s | Decode tok/s | TTFT | KV cache |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | none | **2888.2** | **101.8** | 350 ms | 308 MB |
| Gemma 4 E2B | turbo4v2 | 2728.0 | 93.9 | 371 ms | 63 MB |
| GPT-OSS 20B | none | 580.8 | **64.0** | 1764 ms | 282 MB |
| GPT-OSS 20B | turbo4v2 | 589.0 | 46.5 | 1739 ms | 56 MB |

Output was coherent on all four runs.

**Surprise finding:** on this branch, **no-quant KV is faster than turbo4v2**
on both models (Gemma4: +8% decode, GPT-OSS: +38% decode). Historical
summarization peaks were 100.2 tok/s no-quant / 101.9 tok/s turbo4v2 on
`ek/tom-eric-moe-tuning`; we match the no-quant number but turbo4v2 has
regressed — worth investigating separately from ICB.

For ICB targeting: Gemma 4 E2B no-quant at **101.8 tok/s** and GPT-OSS 20B
no-quant at **64.0 tok/s** are the real comparison points.

Earlier 89.8 tok/s / 143 ms TTFT baseline was from a 38-token-prompt `simple`
run and is kept in the results table for reference but is not the relevant
floor for ICB evaluation.

## Projection for ICB-enabled decode

From the Phase 4d encoding-cost measurement:

| Model | Live (µs/step) | Replay (µs/step) | Savings |
|---|---:|---:|---:|
| Gemma 4 E2B | 8,573 | 6,779 | **1.79 ms** |
| GPT-OSS 20B | 18,939 | 12,165 | **6.77 ms** |

Naive projection applied to 1024-ctx decode floors:

```
Gemma 4 E2B:   9.82 ms/tok (101.8)  - 1.79 ms = 8.03 ms/tok → ~125 tok/s  (+23%)
GPT-OSS 20B:  15.63 ms/tok (64.0)   - 6.77 ms = 8.86 ms/tok → ~113 tok/s  (+76%)
```

Caveats on GPT-OSS: the 6.77 ms savings is measured against 85% dispatch
capture — 134 of 874 dispatches leaked to secondary streams during record.
Real tok/s delta is bounded by how many of those leaks persist after
multi-stream capture lands. The 76% projection is therefore the **upper
bound** assuming full capture; the lower bound assumes the leaked dispatches
are fixed cost and the win is proportional to the 740 captured ones
(~+60%). Either way GPT-OSS has much more ICB headroom than Gemma.

Caveats on the projection:

- Assumes the full 1.79 ms/step savings survive the replay-with-overrides
  rebinding cost. Rebinding N slots via `IndirectComputeCommand::setKernelBuffer`
  is an API call per slot; the overhead is small but non-zero.
- Assumes GPU execution does NOT become the bottleneck at the new rate.
  At 9.35 ms/step on M1 Max decoding a 30-layer 4-bit model, GPU utilization
  is probably still <100%, so CPU-encoding savings should translate.
- Ignores any interaction between ICB segments and mlx's async eval pipeline.

## What's needed to measure ICB-enabled tok/s

1. **Per-replay binding overrides.** Add `replay(overrides: [Int: MLXArray])`
   on `IndirectCommandBuffer`. Plumbs through mlx-c
   (`mlx_metal_icb_replay_with_overrides`) and down to
   `IndirectCommandRecorder::replay_with_overrides` which walks each
   segment's IndirectComputeCommand and calls `setKernelBuffer` before
   `executeCommandsInBuffer`. Metal's ICB commands are mutable, so this is
   a supported pattern. Est. 3-4 hours.

2. **Binding provenance.** To know which command + slot to override with
   the new token buffer / cache offset / output logits, the recorder needs
   to track where each binding came from. Options:
   - Tag each `set_kernel_buffer` call with a symbolic name (lightweight
     API addition on the recorder)
   - Diagnostic recording mode that logs every (command_idx, slot, array_id)
     tuple, then a separate Swift-side classifier identifies which
     correspond to the model's mutable inputs
   Est. 2-3 hours.

3. **TokenIterator integration.** After step 1 of generation (which runs
   live and captures the ICB), subsequent steps call
   `icb.replay(overrides: {inputSlot: newToken, outputSlot: freshLogits, ...})`.
   Add a GenerateParameters flag `icbEnabled: Bool` so we can A/B benchmark.
   Est. 1-2 hours including the `--icb` flag on the benchmark script.

Total additional scope for a real tok/s measurement: **~6-9 hours of focused
engineering across mlx / mlx-c / mlx-swift / mlx-swift-lm**.

## Alternative path: multi-stream capture (close GPT-OSS leak)

Instead of building tok/s for Gemma 4 (where ICB is 100% capture and the
win is projected 19%), it may be more valuable to first close the GPT-OSS
leak (134 of 874 dispatches bypass the recorder) since GPT-OSS is the
strategy-doc's primary target (1542 dispatches/token, +15-25% projected).

Multi-stream capture would either:
- Record ALL active `CommandEncoder`s simultaneously (N recorders, one per
  stream), replayed in submission order; or
- Add a `Stream` scope that pins all primitives in the record closure to
  a single stream.

Option 1 is more invariant but requires coordinating N recorders. Option 2
risks perturbing the execution pattern enough to change what gets dispatched
(e.g., forcing MoE expert scheduling onto the main stream).

## Status today (captured on `ek/metal-icb-prototype`)

- ✅ Feasibility validated (17x isolated dispatch encoding speedup)
- ✅ Real-model end-to-end encoding win on two architectures
- ✅ Gemma 4 E2B: 1.21-1.33x speedup, 100% dispatch capture, 0 leaks
- ✅ GPT-OSS 20B: 1.40-1.74x speedup, 85% dispatch capture, 134 leaks
- ✅ All fixes pushed to the four-repo stack
- ✅ Gemma 4 E2B baseline tok/s captured (89.8 simple, **101.8** summarization-1024)
- ✅ GPT-OSS 20B baseline tok/s captured (**64.0** no-quant / 46.5 turbo4v2 @ 1024-ctx)
- ⚠️ Turbo4v2 regression observed on `ek/metal-icb-prototype`: slower than no-quant on both models — separate investigation from ICB
- ✅ Multi-stream capture closed the 134-dispatch GPT-OSS leak (thread-local encoder steering). 874/874 captured, 0 leaked; encoding speedup 1.45x (down from 1.54x @ 85% capture — the 134 newly-captured MoE dispatches add replay cost but trade for numerical-parity viability)
- ⏳ ICB-enabled tok/s: requires replay-with-overrides + decode-loop
  integration (enumerated above, not yet implemented)
- ⏳ GPT-OSS 100% capture: requires multi-stream recording (enumerated
  above, not yet implemented)
