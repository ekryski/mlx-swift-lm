# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-15 23:35
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
- **Quantization**: bf16
- **Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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
| Collect per-token data | false |
| Track perplexity | true |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 1 |
| Thinking token budget (processor) | 200 |
| Thinking (effective) | Yes |
| Perplexity tracking (MLX_BENCH_PPL) | Yes |
| KL divergence (MLX_BENCH_KLD) | Yes (not evaluated — baseline configuration) |
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
| summarization | 128 | 117 | no-quant | 217.6 | 118.3 | 400 | 538ms | 2.3630 | 2.7216 | — | — | 1.40GB | 1.76GB | 0MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 231.3 | 113.2 | 400 | 1077ms | 2.3592 | 3.6627 | — | — | 1.40GB | 1.77GB | 0MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 233.3 | 116.7 | 400 | 2161ms | 2.6040 | 2.4774 | — | — | 1.40GB | 1.76GB | 0MB | 198MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 226.8 | 112.7 | 400 | 4494ms | 3.7835 | 3.6542 | — | — | 1.40GB | 1.78GB | 0MB | 310MB | The user is asking for a summary of the provided text, which |
| summarization | 2048 | 2042 | no-quant | 234.4 | 116.1 | 400 | 8711ms | 2.1517 | 2.4345 | — | — | 1.40GB | 1.82GB | 0MB | 534MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 240.6 | 105.8 | 400 | 16979ms | 2.6624 | 2.2576 | — | — | 1.40GB | 1.81GB | 0MB | 981MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 8192 | 8190 | no-quant | 227.2 | 108.7 | 400 | 36050ms | 3.6852 | 4.0908 | — | — | 1.40GB | 1.98GB | 0MB | 1.84GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 234.9 | 99.7 | 400 | 69663ms | 3.3659 | 2.9927 | — | — | 1.40GB | 2.13GB | 0MB | 3.58GB | The user wants a summary of "The Great Gatsby" by F. Scott F |
| summarization | 32768 | 32700 | no-quant | 228.6 | 86.4 | 400 | 143034ms | 2.7690 | 3.8387 | — | — | 1.40GB | 2.44GB | 0MB | 7.07GB | The user wants a summary of the provided text, which is *The |
| summarization | 65536 | 65468 | no-quant | 216.6 | 67.8 | 400 | 302267ms | 3.3560 | 1.4755 | — | — | 1.40GB | 3.19GB | 0MB | 14.07GB | The user wants a summary of the provided text, which is a no |
| summarization | 131072 | 130773 | no-quant | 193.9 | 47.8 | 400 | 674440ms | 3.1305 | 2.5116 | — | — | 1.40GB | 4.75GB | 0MB | 28.02GB | The user wants a summary of the provided text. The text is a |
