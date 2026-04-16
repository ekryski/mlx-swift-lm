# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-15 14:54
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
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
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 200 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 114 | no-quant | 277.5 | 26.8 | 101 | 412ms | — | — | — | — | 13.48GB | 14.46GB | 0MB | 47MB | The provided text is the opening front matter of F. Scott Fi |
| summarization | 1024 | 1012 | no-quant | 600.5 | 28.1 | 347 | 1686ms | — | — | — | — | 13.48GB | 15.18GB | 0MB | 297MB | The provided text serves as the opening of F. Scott Fitzgera |
| summarization | 4096 | 4092 | no-quant | 578.8 | 24.4 | 397 | 7072ms | — | — | — | — | 13.48GB | 18.07GB | 0MB | 982MB | This text contains the opening of F. Scott Fitzgerald’s nove |
| summarization | 32768 | 32768 | no-quant | 357.4 | 17.9 | 400 | 92128ms | — | — | — | — | 13.48GB | 26.96GB | 0MB | 7.09GB | This text is the complete first five chapters of F. Scott Fi |
