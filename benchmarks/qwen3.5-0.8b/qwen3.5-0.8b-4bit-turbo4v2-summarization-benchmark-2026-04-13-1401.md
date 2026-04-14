# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-13 14:01
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
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
| Think start token id | 248068 |
| Think end token id | 248069 |
| Thinking phase prefilled | true |
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
| summarization | 128 | 117 | turbo4v2 | 244.2 | 168.6 | 400 | 480ms | 3.8579 | 1.2101 | 0.4777 | 0.1677 | 404MB | 774MB | 0MB | 23MB | The user has provided the Table of Contents for "Once again  |
| summarization | 1024 | 1019 | turbo4v2 | 311.9 | 169.0 | 400 | 3267ms | 2.5687 | 2.4941 | 0.5773 | 0.3619 | 404MB | 784MB | 0MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   **So |
| summarization | 4096 | 4085 | turbo4v2 | 315.6 | 174.1 | 400 | 12943ms | 2.8038 | 2.2871 | 0.5581 | 0.2006 | 404MB | 820MB | 0MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Task |
| summarization | 32768 | 32700 | turbo4v2 | 273.2 | 117.8 | 400 | 119714ms | 2.9560 | 2.6790 | 0.6658 | 0.2926 | 404MB | 1.13GB | 0MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   The  |
