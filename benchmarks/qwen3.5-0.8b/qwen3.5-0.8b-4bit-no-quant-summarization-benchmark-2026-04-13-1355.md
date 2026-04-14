# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-13 13:55
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-0.8B-4bit`

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
| Think start token id | 248068 |
| Think end token id | 248069 |
| Thinking phase prefilled | true |
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
| summarization | 128 | 117 | no-quant | 244.6 | 163.6 | 400 | 479ms | 4.5573 | 1.1576 | 0.7255 | 0.1679 | 404MB | 775MB | 0MB | 113MB | Thinking, the user is asking to summarize a text-based table |
| summarization | 1024 | 1019 | no-quant | 286.0 | 163.8 | 400 | 3564ms | 3.9527 | 3.5309 | 0.4524 | 0.2979 | 404MB | 785MB | 0MB | 310MB | The user is asking for a summary of the provided text. I'll  |
| summarization | 4096 | 4085 | no-quant | 286.9 | 162.6 | 400 | 14241ms | 2.8553 | 1.7831 | 0.5095 | 0.1765 | 404MB | 821MB | 0MB | 981MB | The user is asking for a summary of the provided text, which |
| summarization | 32768 | 32700 | no-quant | 269.9 | 111.1 | 400 | 121165ms | 3.4483 | 1.2624 | 0.6649 | 0.1620 | 404MB | 1.13GB | 0MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   The  |
