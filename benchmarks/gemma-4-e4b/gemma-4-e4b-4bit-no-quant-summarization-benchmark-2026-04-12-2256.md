# Inference Benchmark - Gemma 4 E4B

- **Date**: 2026-04-12 22:56
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `caddf7a perf: dtype fixes, GDN ops fallback, SSM reshape, warp MoE (disabled)`
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
| summarization | 128 | 110 | no-quant | 429.2 | 68.4 | 88 | 258ms | — | — | — | — | 3.98GB | 4.33GB | 2MB | 43MB | The provided here is a summary:  The provided text appears t |
| summarization | 512 | 496 | no-quant | 717.0 | 62.5 | 274 | 693ms | — | — | — | — | 3.98GB | 4.63GB | 23MB | 168MB | This excerpt provided text provided text passage is a passag |
| summarization | 1024 | 1008 | no-quant | 737.0 | 59.9 | 400 | 1370ms | — | — | — | — | 3.98GB | 4.67GB | 32MB | 308MB | The provided text contains two excerpts, which appear to be  |
| summarization | 4096 | 4088 | no-quant | 620.5 | 62.8 | 400 | 6591ms | — | — | — | — | 3.98GB | 5.00GB | 79MB | 982MB | The provided text is the beginning of *The Great Gatsby*, co |
