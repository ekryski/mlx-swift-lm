# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-13 14:12
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-2B-4bit`

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
| summarization | 128 | 117 | no-quant | 237.9 | 128.9 | 400 | 492ms | 2.4717 | 2.7085 | 0.4825 | 0.2602 | 1010MB | 1.35GB | 0MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 272.5 | 126.5 | 400 | 3740ms | 2.2140 | 2.8465 | 0.3252 | 0.2781 | 1010MB | 1.36GB | 0MB | 310MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 4096 | 4085 | no-quant | 288.6 | 129.5 | 400 | 14156ms | 2.4656 | 3.4989 | 0.3957 | 0.2452 | 1010MB | 1.45GB | 0MB | 981MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 32768 | 32700 | no-quant | 258.1 | 93.6 | 400 | 126712ms | 2.4812 | 1.2405 | 0.4974 | 0.1255 | 1010MB | 1.76GB | 0MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
