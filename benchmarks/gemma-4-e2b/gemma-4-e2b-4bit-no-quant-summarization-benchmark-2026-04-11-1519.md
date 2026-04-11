# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 15:19
- **Branch**: `ek/more-c-kernels`
- **Commit**: `f1af83a perf: migrate NR0 flash kernels to framework dispatch — zero JIT remaining`
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
| summarization | 128 | 116 | no-quant | 678.9 | 82.5 | 400 | 172ms | — | 1.1192 | 0.4905 | — | 2.45GB | 2.87GB | 3MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 1822.3 | 78.4 | 400 | 558ms | — | 1.3022 | 0.6897 | — | 2.45GB | 3.29GB | 15MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1477.9 | 77.8 | 400 | 2771ms | — | 1.4886 | 0.8585 | — | 2.45GB | 3.33GB | 39MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 32768 | 32768 | no-quant | 399.1 | 61.0 | 400 | 82105ms | — | 1.9367 | 1.2707 | — | 2.45GB | 3.49GB | 263MB | 7.09GB | <\|channel>thought Here's a thinking process that leads to th |
