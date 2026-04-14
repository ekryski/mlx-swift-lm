# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-13 10:18
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `caddf7a perf: dtype fixes, GDN ops fallback, SSM reshape, warp MoE (disabled)`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-0.8B-4bit`

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
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
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
| summarization | 128 | 109 | turbo4v2 | 280.1 | 210.5 | 400 | 390ms | — | — | — | — | 404MB | 621MB | 0MB | 23MB | # Summary of the Provided Text: The Song "Once Again to Zeld |
| summarization | 512 | 496 | turbo4v2 | 290.5 | 198.3 | 333 | 1708ms | — | — | — | — | 404MB | 646MB | 0MB | 37MB | Based on the text provided, here is a summary of the content |
| summarization | 1024 | 1011 | turbo4v2 | 292.9 | 192.6 | 400 | 3452ms | — | — | — | — | 404MB | 677MB | 0MB | 63MB | Based on the text provided below, here is a summary of the c |
| summarization | 4096 | 4077 | turbo4v2 | 283.8 | 185.1 | 400 | 14367ms | — | — | — | — | 404MB | 752MB | 0MB | 199MB | Based on the provided text from *The Great Gatsby* by F. Sco |
