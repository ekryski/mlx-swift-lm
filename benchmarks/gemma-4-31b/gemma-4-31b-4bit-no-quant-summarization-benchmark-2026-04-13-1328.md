# Inference Benchmark - Gemma 4 31B

- **Date**: 2026-04-13 13:28
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-31b-it-4bit`

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
| summarization | 128 | 116 | no-quant | 53.9 | 14.2 | 400 | 2152ms | 13.0126 | 9.6546 | 8.1273 | 8.4415 | 16.09GB | 16.60GB | 0MB | 113MB | <\|channel><\|channel>thought   <\|channel><\|channel><\|channel> |
| summarization | 512 | 502 | no-quant | 48.6 | 13.9 | 400 | 10332ms | 3.7257 | 2.0712 | 6.9384 | 3.7234 | 46.47GB | 46.47GB | 0MB | 197MB | <\|channel><\|channel>   s <\|channel><\|channel>  mniej<\|channe |
| summarization | 1024 | 1014 | no-quant | 52.4 | 13.3 | 196 | 19338ms | 1.4052 | 1.1265 | 1.5646 | 0.5073 | 46.47GB | 46.47GB | 0MB | 265MB | <\|channel><\|channel>    <\|channel><\|channel><\|channel><\|chan |
