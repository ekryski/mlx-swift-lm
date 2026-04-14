# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-12 12:39
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `3c197a7 improvements to our prefill bridge to use memory pointers to reduce CPU -> GPU round trips`
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
| summarization | 128 | 110 | no-quant | 975.2 | 105.2 | 224 | 114ms | — | — | — | — | 2.45GB | 2.49GB | 1MB | 73MB | The provided text is a highly fragmented and incomplete exce |
| summarization | 1024 | 1008 | no-quant | 2474.5 | 99.1 | 400 | 409ms | — | — | — | — | 2.45GB | 2.50GB | 9MB | 308MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | no-quant | 2557.0 | 95.3 | 400 | 1600ms | — | — | — | — | 2.45GB | 2.52GB | 20MB | 982MB | The provided text is an excerpt from **The Great Gatsby** by |
