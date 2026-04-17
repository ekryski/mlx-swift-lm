# ICB Tok/s Projection + Gemma 4 E2B Baseline — 2026-04-17

Follow-up to [icb-first-measurement-2026-04-17.md](icb-first-measurement-2026-04-17.md).
Captures the baseline decode throughput we'd be comparing against once the
ICB decode-loop integration lands, and enumerates what's still missing.

## Baseline: Gemma 4 E2B (no ICB, alpha-branch path)

- Hardware: M1 Max, 64 GB, macOS 15.7.4
- Model: Gemma 4 E2B 4-bit (`mlx-community/gemma-4-e2b-it-4bit`)
- Command: `./scripts/benchmark.sh --model gemma4-e2b --method simple`
- Prompt: 38 tokens; context limit: 4096; output: 36 tokens

| Metric | Value |
|---|---|
| Prefill | **270.1 tok/s** |
| Generation (decode) | **89.8 tok/s** |
| TTFT | **143 ms** |
| Total time | 0.5 s |
| GPU baseline | 2.45 GB |
| GPU peak | 2.55 GB |
| KV cache | 16 MB |

Output was coherent (normal assistant response), confirming the ICB-enabling
patches on `ek/metal-icb-prototype` have not regressed the non-ICB path.

## Projection for ICB-enabled decode

From the Phase 4d encoding-cost measurement:

- Live decode step (step-2+): 8.57 ms
- ICB-replay step: 6.78 ms
- Per-step savings: ~1.79 ms

Naive projection at 89.8 tok/s (11.14 ms/token):

```
11.14 ms - 1.79 ms = 9.35 ms/token → ~107 tok/s  (+19%)
```

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
- ✅ Gemma 4 E2B baseline tok/s captured (89.8 tok/s / 143ms TTFT)
- ⏳ ICB-enabled tok/s: requires replay-with-overrides + decode-loop
  integration (enumerated above, not yet implemented)
- ⏳ GPT-OSS 100% capture: requires multi-stream recording (enumerated
  above, not yet implemented)
