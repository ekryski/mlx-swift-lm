# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 16:53
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
| summarization | 128 | 110 | no-quant | 951.4 | 107.4 | 183 | 117ms | — | — | — | — | 2.45GB | 2.63GB | 5MB | 64MB | The provided text appears to be a **fragment of a poem or li |
| summarization | 1024 | 1008 | no-quant | 2083.1 | 99.7 | 400 | 485ms | — | — | — | — | 2.45GB | 3.28GB | 11MB | 308MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | no-quant | 1502.9 | 98.0 | 400 | 2721ms | — | — | — | — | 2.45GB | 3.33GB | 29MB | 982MB | This excerpt is a collection of narrative and reflective pas |
| summarization | 32768 | 32768 | no-quant | 404.9 | 75.3 | 400 | 80926ms | — | — | — | — | 2.45GB | 3.49GB | 132MB | 7.09GB | This is a selection of text from **The Great Gatsby** by F.  |
