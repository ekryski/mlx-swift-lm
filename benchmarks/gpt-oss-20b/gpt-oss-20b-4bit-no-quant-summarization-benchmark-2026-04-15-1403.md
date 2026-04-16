# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-15 14:03
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
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
| KV cache strategy | None (FP16) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
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
| summarization | 128 | 128 | no-quant | 344.1 | 58.9 | 400 | 373ms | — | — | — | — | 10.41GB | 10.86GB | 0MB | 116MB | <\|channel\|>analysis<\|message\|>The user message: They typed a |
| summarization | 1024 | 1024 | no-quant | 644.7 | 52.9 | 305 | 1589ms | — | — | — | — | 10.41GB | 11.72GB | 0MB | 291MB | <\|channel\|>analysis<\|message\|>We have a user who posted a cr |
| summarization | 4096 | 4055 | no-quant | 679.9 | 52.6 | 400 | 5965ms | — | — | — | — | 10.41GB | 12.54GB | 0MB | 975MB | <\|channel\|>analysis<\|message\|>We have a user-provided text:  |
| summarization | 32768 | 31717 | no-quant | 553.2 | 39.7 | 400 | 57339ms | — | — | — | — | 10.41GB | 15.01GB | 0MB | 6.86GB | <\|channel\|>analysis<\|message\|>We have a long text: appears t |
