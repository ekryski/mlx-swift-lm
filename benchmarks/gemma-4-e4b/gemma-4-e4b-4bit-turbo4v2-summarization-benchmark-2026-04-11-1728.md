# Inference Benchmark - Gemma 4 E4B

- **Date**: 2026-04-11 17:28
- **Branch**: `ek/fuse-all-the-kernels`
- **Commit**: `85ead8b adding more gemma4 benchmarks.`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e4b-it-4bit`

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
| summarization | 128 | 110 | turbo4v2 | 438.7 | 68.6 | 169 | 252ms | — | — | — | — | 3.98GB | 4.33GB | 14MB | 12MB | The provided text contains two distinct parts:  1. **The tit |
| summarization | 1024 | 1008 | turbo4v2 | 659.7 | 65.9 | 400 | 1530ms | — | — | — | — | 3.98GB | 4.67GB | 35MB | 63MB | This text is an excerpt from the beginning of F. Scott Fitzg |
| summarization | 4096 | 4088 | turbo4v2 | 590.9 | 63.6 | 400 | 6921ms | — | — | — | — | 3.98GB | 4.85GB | 62MB | 199MB | The provided text is the beginning of John Fitzgerald Kenned |
| summarization | 32768 | 32768 | turbo4v2 | 232.5 | 49.8 | 400 | 140947ms | — | — | — | — | 3.98GB | 5.33GB | 528MB | 1.44GB | Here are several questions you could ask about the provided  |
