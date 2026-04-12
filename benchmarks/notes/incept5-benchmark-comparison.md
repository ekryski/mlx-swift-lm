# Incept5 (mlx-vlm) vs mlx-swift-lm Benchmark Comparison

**Date**: 2026-04-08
**Repo**: https://github.com/Incept5/gemma4-benchmark
**Model**: Gemma 4 E2B 4bit (`mlx-community/gemma-4-e2b-it-4bit`)

## Key Differences

| Factor | Incept5 (mlx-vlm) | mlx-swift-lm |
|--------|-------------------|--------------|
| Framework | Python mlx-vlm 0.4.4 | MLX Swift |
| Prompt content | "The quick brown fox..." repeated | The Great Gatsby (real prose) |
| Max output tokens | 32 | 200 |
| Thinking mode | Off (`enable_thinking: false`) | Off (behind `--think` flag, disabled by default) |
| Perplexity tracking | No | Yes (optional, adds overhead) |
| KV quantization | 2.5-bit TurboQuant | turbo4v2 / various configs |
| Warmup | Single 5-token warmup | Warmup pass + GPU sync |

## Results: Incept5 on M1 Max 64GB (mlx-vlm)

| Context | Prefill tok/s | Decode tok/s | TQ Prefill | TQ Decode |
|---------|--------------|-------------|------------|-----------|
| 4096 | 2,225 | 96.1 | 3,067 | 96.3 |
| 8192 | 3,048 | 96.0 | 1,956 | 85.1 |
| 16384 | 2,879 | 86.4 | 2,826 | 82.9 |
| 32768 | 2,598 | 76.6 | 2,481 | 72.3 |

## Results: mlx-swift-lm on M1 Max 64GB (no KV quant)

| Context | Prefill tok/s | Decode tok/s |
|---------|--------------|-------------|
| 1024 | 1,013 | 80.1 |
| 4096 | 1,619 | 78.9 |
| 16384 | 2,576 | 68.6 |

## Results: Tom's M5 Max 128GB (mlx-swift-lm, turbo4v2, no PPL)

| Context | Prefill tok/s | Decode tok/s |
|---------|--------------|-------------|
| 4096 | 7,773 | 150.0 |
| 8192 | 10,252 | 147.0 |
| 16384 | 12,021 | 132.9 |
| 32768 | 11,203 | 115.1 |

## Results: Incept5 Published M5 Max 128GB (from README)

| Context | Decode tok/s |
|---------|-------------|
| 4096 | 205 |
| 8192 | 182 |
| 16384 | 148 |
| 32768 | 121 |

## Controlled Comparison: Repetitive vs Gatsby, 32 vs 200 Tokens

To isolate the effect of prompt content and generation length, we ran the Incept5 mlx-vlm
framework on the same M1 Max 64GB with four configurations per context length:
- **rep-32**: Repetitive "quick brown fox" prompt, 32 max tokens (original Incept5 config)
- **rep-200**: Repetitive prompt, 200 max tokens (matching our generation length)
- **gatsby-32**: Great Gatsby prompt, 32 max tokens
- **gatsby-200**: Great Gatsby prompt, 200 max tokens (matching our full config)

### 4096 context

| Config | Prefill tok/s | Decode tok/s |
|--------|--------------|-------------|
| rep-32 | 3,154 | 99.1 |
| rep-200 | 3,103 | 99.3 |
| gatsby-32 | 3,198 | 98.9 |
| gatsby-200 | 3,127 | 96.0 |

### 8192 context

| Config | Prefill tok/s | Decode tok/s |
|--------|--------------|-------------|
| rep-32 | 3,028 | 95.2 |
| rep-200 | 3,012 | 94.3 |
| gatsby-32 | 3,039 | 95.3 |
| gatsby-200 | 3,114 | 92.2 |

### 16384 context

| Config | Prefill tok/s | Decode tok/s |
|--------|--------------|-------------|
| rep-32 | 2,889 | 88.8 |
| rep-200 | 2,909 | 88.6 |
| gatsby-32 | 2,910 | 88.6 |
| gatsby-200 | 2,922 | 85.2 |

### 32768 context

| Config | Prefill tok/s | Decode tok/s |
|--------|--------------|-------------|
| rep-32 | 2,607 | 78.2 |
| rep-200 | 2,543 | 76.6 |
| gatsby-32 | 2,596 | 77.7 |
| gatsby-200 | 2,587 | 75.5 |

### Decode impact summary

| Context | rep-32 → rep-200 | gatsby-32 → gatsby-200 |
|---------|------------------|------------------------|
| 4k | 99.1 → 99.3 (noise) | 98.9 → 96.0 (-3%) |
| 8k | 95.2 → 94.3 (-1%) | 95.3 → 92.2 (-3%) |
| 16k | 88.8 → 88.6 (noise) | 88.6 → 85.2 (-4%) |
| 32k | 78.2 → 76.6 (-2%) | 77.7 → 75.5 (-3%) |

## Analysis

### Prompt content has zero effect on prefill or decode speed

Contrary to the initial hypothesis, repetitive vs diverse text makes no measurable difference
at the MLX kernel level. Attention computation cost is the same regardless of token diversity —
the matrix multiplications don't care about semantic content. Prefill and decode numbers are
within noise across all context lengths when controlling for max tokens.

### Max tokens has a small but consistent effect on decode

Generating 200 tokens instead of 32 degrades Gatsby decode by ~3-4% consistently. The
repetitive prompt shows almost no penalty at 200 tokens, suggesting the repetitive KV cache
may be slightly easier to attend over during extended generation — but the effect is small.

### The real decode gap is framework + hardware, not methodology

On the same M1 Max hardware with the same mlx-vlm framework, our Gatsby-200 config
produces decode numbers (96.0 @ 4k, 75.5 @ 32k) that are only ~3% slower than his
rep-32 config (99.1 @ 4k, 78.2 @ 32k). The methodology difference is negligible.

Comparing Incept5's published M5 Max results (205 tok/s decode @ 4k) vs Tom's M5 Max
results on our MLX Swift code (150 tok/s @ 4k with turbo4v2), the ~35% gap is therefore
mostly attributable to:
1. **Framework difference** — mlx-vlm Python vs MLX Swift runtime
2. **PPL tracking overhead** in Tom's with-PPL run
3. **Max tokens** accounts for only ~3% of the gap

### Thinking mode

Both benchmarks have thinking disabled. Our `--think` flag flows all the way through inference
(not just CLI/benchmark surface): `BenchEnv.thinkEnabled` gates it in
`InferenceBenchmark.swift:544`, injecting `enable_thinking: true` into the Gemma 4 chat
template and adding a 200-token thinking budget via `ThinkingBudgetProcessor`.

### Conclusion

The Incept5 benchmark methodology (repetitive prompt, 32 tokens) is not meaningfully gaming
the numbers vs our approach (Gatsby prose, 200 tokens) — the difference is only ~3%.
Our benchmark is still more representative of real-world usage, but the performance gap
between published numbers is primarily a framework difference (mlx-vlm vs MLX Swift),
not a benchmarking methodology issue.

### Update: Forward pass fixes (2026-04-08)

Deep-dive found 5 concrete differences in our Gemma4 Swift model vs Python's:
1. Norm applied after transpose instead of before (wrong memory layout for RMSNorm)
2. Missing kernel fusion for logit softcapping (3 kernels -> 1)
3. Missing kernel fusion for GEGLU in MLP (2 kernels -> 1)
4. Unnecessary dtype conversion in embedding scale
5. Per-layer inputs re-sliced inside loop instead of pre-sliced

After fixing: decode improved from 78.9 to 80.7 tok/s at 4k (+2.3%), and from 68.6 to
72.1 tok/s at 16k (+5.1%). See `generation-loop-perf-fixes.md` for full details.
