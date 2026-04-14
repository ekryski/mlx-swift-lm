# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-13 13:07
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
| summarization | 128 | 116 | turbo4v2 | 236.1 | 23.9 | 400 | 492ms | 1.6658 | 1.8524 | 2.6365 | 5.2737 | 13.48GB | 14.51GB | 0MB | 23MB | --- enoughthought  <\|channel>thought  <\|channel>thought  <\|c |
| summarization | 512 | 502 | turbo4v2 | 87.9 | 24.4 | 231 | 5714ms | 2.4869 | 2.2106 | 1.4667 | 1.3249 | 38.45GB | 38.45GB | 0MB | 33MB | --- <\|channel>thought  <\|channel>thought  ->andters- @- \| wa |
| summarization | 1024 | 1014 | turbo4v2 | 208.2 | 23.8 | 400 | 4871ms | 1.0571 | 1.4882 | 0.5789 | 0.5286 | 38.45GB | 38.45GB | 0MB | 63MB | ---leprint-expression-try own-off- ---  Summarize the<\|chann |
| summarization | 4096 | 4094 | turbo4v2 | 502.3 | 22.0 | 400 | 8151ms | 1.3236 | 1.2148 | 0.8123 | 0.2171 | 38.45GB | 38.45GB | 0MB | 200MB | --- <\|channel>thought  *   Source Material: The first chapte |
