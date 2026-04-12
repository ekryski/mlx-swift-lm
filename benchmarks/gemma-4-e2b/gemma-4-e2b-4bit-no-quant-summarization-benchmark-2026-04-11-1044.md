# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 10:44
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
| summarization | 128 | 116 | no-quant | 661.3 | 80.2 | 400 | 176ms | — | 1.1823 | 0.9499 | — | 2.45GB | 2.87GB | 8MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 1875.9 | 78.1 | 400 | 542ms | — | 1.3668 | 0.5926 | — | 2.45GB | 3.22GB | 15MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1392.5 | 75.6 | 400 | 2941ms | — | 1.4082 | 0.8442 | — | 2.45GB | 3.32GB | 37MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 32768 | 32768 | no-quant | 392.9 | 61.6 | 400 | 83411ms | — | 1.7698 | 1.2591 | — | 2.45GB | 3.44GB | 262MB | 7.09GB | <\|channel>thought Here's a thinking process that leads to th |
