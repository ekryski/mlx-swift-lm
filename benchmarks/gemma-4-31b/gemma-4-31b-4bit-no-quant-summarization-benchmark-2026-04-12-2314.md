# Inference Benchmark - Gemma 4 31B

- **Date**: 2026-04-12 23:14
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `caddf7a perf: dtype fixes, GDN ops fallback, SSM reshape, warp MoE (disabled)`
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
| summarization | 128 | 114 | no-quant | 53.7 | 14.9 | 109 | 2124ms | — | — | — | — | 16.09GB | 16.44GB | 89MB | 49MB | <channel\|><channel\|>s<\|channel>thought <channel\|>senthy  y<c |
| summarization | 512 | 500 | no-quant | 63.6 | 14.6 | 400 | 7858ms | — | — | — | — | 16.09GB | 17.08GB | 406MB | 197MB | <channel\|>s <\|channel>s thought conclusion<\|channel>isunouve |
| summarization | 1024 | 1012 | no-quant | 64.7 | 13.3 | 246 | 15642ms | — | — | — | — | 16.09GB | 17.78GB | 783MB | 275MB | <channel\|>t <\|channel> thestartTime all C,startTime <channel |
| summarization | 4096 | 4092 | no-quant | 50.5 | 13.7 | 359 | 81039ms | — | — | — | — | 16.09GB | 20.83GB | 976MB | 974MB | The provided text consists of the opening pages of F. Scott  |
