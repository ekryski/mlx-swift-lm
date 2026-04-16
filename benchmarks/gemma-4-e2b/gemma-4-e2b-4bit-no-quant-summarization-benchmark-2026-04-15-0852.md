# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-15 08:52
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `062a628 fixing buffer overflow bug in rotating KV cache`
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
| summarization | 128 | 116 | no-quant | 845.4 | 82.1 | 400 | 138ms | — | 1.3133 | 1.1625 | — | 2.45GB | 2.87GB | 0MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 1038.9 | 80.0 | 400 | 247ms | — | 1.5288 | 1.4750 | — | 2.45GB | 2.87GB | 0MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 1523.9 | 77.0 | 400 | 331ms | — | 1.1867 | 0.8391 | — | 2.45GB | 3.02GB | 0MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 2016.9 | 80.1 | 400 | 504ms | — | 1.3771 | 0.7361 | — | 2.45GB | 3.29GB | 0MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 2054.3 | 80.0 | 400 | 993ms | — | 1.4619 | 1.0933 | — | 2.45GB | 3.31GB | 0MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1652.2 | 78.5 | 400 | 2479ms | — | 1.3710 | 0.8631 | — | 2.45GB | 3.35GB | 0MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | no-quant | 1238.9 | 75.7 | 400 | 6614ms | — | 1.7794 | 1.3711 | — | 2.45GB | 3.43GB | 0MB | 1.84GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 16384 | 16384 | no-quant | 790.5 | 70.7 | 400 | 20727ms | — | 1.8494 | 1.1701 | — | 2.45GB | 3.49GB | 0MB | 3.59GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 32768 | 32768 | no-quant | 431.0 | 61.7 | 400 | 76032ms | — | 1.8119 | 1.2379 | — | 2.45GB | 3.62GB | 0MB | 7.09GB | <\|channel>thought Here's a thinking process that leads to th |
