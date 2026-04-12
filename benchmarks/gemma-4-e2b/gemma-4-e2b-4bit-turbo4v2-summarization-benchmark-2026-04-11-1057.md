# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 10:57
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `698c137 adding nemotron cascade 2 4bit benches`
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
| summarization | 128 | 116 | turbo4v2 | 778.9 | 78.8 | 400 | 150ms | — | 1.6383 | 1.7237 | — | 2.45GB | 2.87GB | 6MB | 23MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | turbo4v2 | 1823.6 | 77.3 | 400 | 557ms | — | 1.2954 | 0.7519 | — | 2.45GB | 3.22GB | 0MB | 63MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | turbo4v2 | 1424.2 | 75.5 | 400 | 2876ms | — | 1.4115 | 0.9048 | — | 2.45GB | 3.32GB | 39MB | 200MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 32768 | 32768 | turbo4v2 | 395.7 | 61.3 | 400 | 82816ms | — | 1.8902 | 1.0754 | — | 2.45GB | 3.44GB | 229MB | 1.44GB | <\|channel>thought Here's a thinking process that leads to th |
