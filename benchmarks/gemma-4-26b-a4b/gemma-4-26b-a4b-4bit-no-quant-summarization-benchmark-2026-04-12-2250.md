# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-12 22:50
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `caddf7a perf: dtype fixes, GDN ops fallback, SSM reshape, warp MoE (disabled)`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M1 Max (applegpu_g13s) |
| System RAM | 64GB |
| GPU Memory Limit | 48GB |
| macOS | 15.7.4 |

## Parameters

| Parameter | Value |
|-----------|-------|
| KV cache strategy | None (FP16) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 2048 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Repetition penalty | nil |
| Repetition context size | 20 |
| Presence penalty | nil |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Collect per-token data | false |
| Track perplexity | false |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 0 |
| Thinking (effective) | No |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Speculative decoding | none |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 100 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 114 | no-quant | 247.5 | 26.0 | 400 | 462ms | — | — | — | — | 13.48GB | 14.16GB | 37MB | 112MB | The provided ability  *thought <channel\|>The in-thought <cha |
| summarization | 512 | 500 | no-quant | 537.8 | 25.9 | 400 | 931ms | — | — | — | — | 13.48GB | 14.74GB | 191MB | 197MB | The provided ability-e own ownly own-inly own-day-person- mu |
| summarization | 1024 | 1012 | no-quant | 559.9 | 25.6 | 376 | 1808ms | — | — | — | — | 13.48GB | 15.05GB | 348MB | 304MB | The providedingness-outly ownly own-nessly  The provided tex |
| summarization | 4096 | 4092 | no-quant | 542.7 | 25.0 | 375 | 7541ms | — | — | — | — | 13.48GB | 18.04GB | 424MB | 977MB | This text is the opening chapter of F. Scott Fitzgerald’s no |
