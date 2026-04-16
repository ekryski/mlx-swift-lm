# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-15 22:47
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
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
| summarization | 128 | 109 | no-quant | 179.5 | 148.1 | 238 | 608ms | — | — | — | — | 1010MB | 1.20GB | 0MB | 76MB | Based on the text you provided, here is a summary of its con |
| summarization | 1024 | 1011 | no-quant | 223.0 | 145.3 | 400 | 4535ms | — | — | — | — | 1010MB | 1.30GB | 0MB | 309MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 4096 | 4077 | no-quant | 215.2 | 135.7 | 400 | 18950ms | — | — | — | — | 1010MB | 1.40GB | 0MB | 979MB | Here is a summary of **F. Scott Fitzgerald's** *The Great Ga |
| summarization | 32768 | 32692 | no-quant | 201.2 | 81.8 | 400 | 162496ms | — | — | — | — | 1010MB | 2.07GB | 0MB | 7.07GB | This text is the first half of *The Great Gatsby* by F. Scot |
