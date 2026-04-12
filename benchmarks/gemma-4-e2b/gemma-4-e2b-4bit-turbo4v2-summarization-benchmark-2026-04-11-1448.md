# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 14:48
- **Branch**: `ek/more-c-kernels`
- **Commit**: `65ad9e3 adding more historical gemma4-e2b benches`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| Think start token id | 100 |
| Think end token id | 101 |
| Thinking phase prefilled | false |
| Collect per-token data | true |
| Track perplexity | true |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 1 |
| Thinking token budget (processor) | 200 |
| Thinking (effective) | Yes |
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
| summarization | 128 | 116 | turbo4v2 | 624.5 | 80.3 | 400 | 187ms | — | 1.1623 | 0.9277 | — | 2.45GB | 2.87GB | 8MB | 23MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | turbo4v2 | 1699.8 | 76.8 | 400 | 598ms | — | 1.3789 | 0.9081 | — | 2.45GB | 3.29GB | 14MB | 63MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | turbo4v2 | 1432.4 | 76.1 | 400 | 2860ms | — | 1.4080 | 1.0757 | — | 2.45GB | 3.33GB | 39MB | 200MB | <\|channel>thought Here's a thinking process to arrive at the |
