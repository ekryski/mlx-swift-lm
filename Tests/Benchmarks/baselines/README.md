# Benchmark Baselines

This directory holds periodic full-matrix benchmark runs for the inference
benchmark suite in `Tests/Benchmarks/InferenceBenchmark.swift`. Each file is
a snapshot of prefill / decode tokens-per-second across the supported model
range on a specific piece of hardware at a specific point in time.

Use these as a reference when:

- Diagnosing a perf regression — compare current numbers against the most
  recent baseline on matching hardware.
- Landing a kernel or framework change — re-run the affected rows and update
  the baseline if the delta is material.
- Picking a model for a target device — the TL;DR table shows prefill/decode
  at 1k context and whether 8k coherency holds.

## File naming

`{hardware}-{ram}-{YYYY-MM-DD}.md`

Examples:
- `m5-max-128gb-2026-04-16.md`
- `m3-ultra-192gb-2026-04-30.md`

## Methodology

All baselines should record:

- Hardware, OS version, and the alpha branch / PRs applied
- Whether NAX is enabled
- `MLX_BENCH_METHOD` (summarization is the default for multi-ctx sweeps)
- Both swift and bridge paths (bridge only differs for Gemma 4)
- Prefill and decode tok/s at 128, 512, 1k, 2k, 4k, 8k, 16k, 32k
- An 8k coherency sample per model (short output snippet proving it generates
  sensible text, not garbage)
