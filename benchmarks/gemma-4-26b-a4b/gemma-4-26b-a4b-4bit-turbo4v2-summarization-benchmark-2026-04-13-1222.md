# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-13 12:22
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
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
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
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
| Collect per-token data | true |
| Track perplexity | true |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 0 |
| Thinking (effective) | No |
| Perplexity tracking (MLX_BENCH_PPL) | Yes |
| KL divergence (MLX_BENCH_KLD) | Yes |
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
| summarization | 128 | 114 | turbo4v2 | 260.2 | 28.6 | 400 | 439ms | — | 1.3723 | — | 1.2410 | 13.48GB | 14.51GB | 0MB | 23MB | Thely-thought 닫st-สุด a little-by-by wayki-ey way. way-wise-e |
| summarization | 512 | 500 | turbo4v2 | 66.4 | 27.9 | 400 | 7527ms | — | 1.1825 | — | 4.3703 | 38.45GB | 38.45GB | 0MB | 40MB | The providedly-แล้วful-สุด own-nya- uniquevel-ness-way-way own |
| summarization | 1024 | 1012 | turbo4v2 | 153.2 | 27.0 | 400 | 6606ms | — | 1.3577 | — | 0.6258 | 38.45GB | 38.45GB | 0MB | 63MB | The provided-us://--- <\|channel>quély-ness-the-end-of-the-pa |
| summarization | 4096 | 4092 | turbo4v2 | 272.1 | 25.7 | 400 | 15038ms | — | 1.2900 | — | 0.3881 | 38.45GB | 38.45GB | 0MB | 200MB | The provided text is the opening chapter of F. Scott Fitzger |
