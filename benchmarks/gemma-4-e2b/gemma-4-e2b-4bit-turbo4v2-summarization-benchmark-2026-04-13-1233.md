# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-13 12:33
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
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
| Collect per-token data | true |
| Track perplexity | true |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 0 |
| Thinking (effective) | No |
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
| summarization | 128 | 110 | turbo4v2 | 988.1 | 107.7 | 400 | 112ms | — | 2.4350 | — | 0.6122 | 2.45GB | 2.87GB | 0MB | 23MB | This appears to summarize the provided text provided text pr |
| summarization | 512 | 496 | turbo4v2 | 1527.9 | 93.8 | 290 | 326ms | — | 1.5623 | — | 0.5783 | 2.45GB | 2.77GB | 0MB | 35MB | This text provided text provided text above contains the beg |
| summarization | 1024 | 1008 | turbo4v2 | 2333.2 | 87.6 | 400 | 433ms | — | 1.5661 | — | 0.5557 | 2.45GB | 2.88GB | 0MB | 63MB | This excerpt provided is an excerpt from *The Great Gatsby*  |
| summarization | 4096 | 4088 | turbo4v2 | 2657.4 | 96.8 | 400 | 1539ms | — | 1.7627 | — | 0.4282 | 2.45GB | 2.91GB | 0MB | 199MB | This excerpt appears to be an excerpt from a section of **Th |
| summarization | 32768 | 32768 | turbo4v2 | 971.4 | 72.9 | 400 | 33735ms | — | 2.2098 | — | 0.6561 | 2.45GB | 3.14GB | 0MB | 1.44GB | This is a substantial amount of text, clearly drawn from **N |
