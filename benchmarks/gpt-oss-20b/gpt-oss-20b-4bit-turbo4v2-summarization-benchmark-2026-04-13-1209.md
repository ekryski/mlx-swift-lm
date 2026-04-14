# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-13 12:09
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
- **Quantization**: 4bit
- **Model**: `loan-star/gpt-oss-20b-mlx-4Bit`

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
| Temperature | 0.8 |
| Top P | 0.8 |
| Top K | 0 |
| Min P | 0.0 |
| Repetition penalty | nil |
| Repetition context size | 20 |
| Presence penalty | nil |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | medium |
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
| summarization | 128 | 128 | turbo4v2 | 387.9 | 62.1 | 340 | 330ms | — | 2.3273 | — | 0.5314 | 10.41GB | 10.86GB | 0MB | 21MB | <\|channel\|>analysis<\|message\|>We need to respond. The user p |
| summarization | 512 | 512 | turbo4v2 | 558.6 | 62.1 | 400 | 917ms | — | 1.7609 | — | 0.7994 | 10.41GB | 11.56GB | 0MB | 41MB | <\|channel\|>analysis<\|message\|>We have a user who posted a te |
| summarization | 1024 | 1024 | turbo4v2 | 664.0 | 59.8 | 400 | 1543ms | — | 1.7555 | — | 0.7649 | 10.41GB | 11.72GB | 0MB | 63MB | <\|channel\|>analysis<\|message\|>We have a user who has posted  |
| summarization | 4096 | 4055 | turbo4v2 | 640.4 | 54.8 | 346 | 6333ms | — | 2.9076 | — | 0.6774 | 10.41GB | 12.36GB | 0MB | 196MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 32768 | 31717 | turbo4v2 | 533.6 | 39.8 | 400 | 59435ms | — | 6.6629 | — | 0.8922 | 10.41GB | 13.62GB | 0MB | 1.39GB | <\|channel\|>analysis<\|message\|>The user has provided a long t |
