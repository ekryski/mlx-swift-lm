# Inference Benchmark - Gemma 4 31B

- **Date**: 2026-04-15 15:52
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
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
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 200 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 114 | turbo4v2 | 53.0 | 14.6 | 50 | 2152ms | — | — | — | — | 16.09GB | 17.02GB | 0MB | 7MB | The provided text is the opening front matter of the novel * |
| summarization | 1024 | 1012 | turbo4v2 | 66.6 | 14.2 | 176 | 15198ms | — | — | — | — | 16.09GB | 18.08GB | 0MB | 53MB | The provided text is the opening of F. Scott Fitzgerald's *T |
| summarization | 4096 | 4092 | turbo4v2 | 53.0 | 13.9 | 389 | 77259ms | — | — | — | — | 16.09GB | 20.86GB | 0MB | 199MB | The provided text is the opening section of F. Scott Fitzger |
| summarization | 32768 | 32768 | turbo4v2 | 20.4 | 11.6 | 400 | 1608661ms | — | — | — | — | 16.09GB | 25.25GB | 0MB | 1.44GB | This text comprises the first five chapters of **"The Great  |
