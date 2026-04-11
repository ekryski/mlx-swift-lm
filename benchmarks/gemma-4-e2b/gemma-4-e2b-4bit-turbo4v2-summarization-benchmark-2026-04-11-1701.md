# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 17:01
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `3ee1954 Merge pull request #16 from TheTom/pr/native-prefill-v2`
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
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 100 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 110 | turbo4v2 | 960.2 | 107.4 | 217 | 115ms | — | — | — | — | 2.45GB | 2.63GB | 5MB | 15MB | The provided text is a fragmented collection of elements fro |
| summarization | 1024 | 1008 | turbo4v2 | 2109.3 | 100.7 | 400 | 479ms | — | — | — | — | 2.45GB | 3.28GB | 8MB | 63MB | This excerpt comes from **The Great Gatsby** by F. Scott Fit |
| summarization | 4096 | 4088 | turbo4v2 | 1510.4 | 97.8 | 400 | 2708ms | — | — | — | — | 2.45GB | 3.33GB | 30MB | 199MB | This excerpt comes from **The Great Gatsby** by F. Scott Fit |
| summarization | 32768 | 32768 | turbo4v2 | 396.6 | 74.0 | 400 | 82629ms | — | — | — | — | 2.45GB | 3.49GB | 131MB | 1.44GB | This is an extensive collection of excerpts from **F. Scott  |
