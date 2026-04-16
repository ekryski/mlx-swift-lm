# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 08:11
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
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
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 200 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 117 | no-quant | 197.8 | 148.0 | 400 | 592ms | 3.5361 | 2.0353 | 0.2717 | 0.2899 | 404MB | 731MB | 0MB | 113MB | The user has shared an excerpt from "The Great Gatsby" writt |
| summarization | 256 | 249 | no-quant | 224.9 | 150.6 | 400 | 1108ms | 3.7807 | 1.2135 | 0.4087 | 0.0529 | 404MB | 738MB | 0MB | 142MB | The user is providing a text excerpt, specifically from the  |
| summarization | 512 | 504 | no-quant | 232.4 | 147.2 | 400 | 2169ms | 3.0474 | 2.3790 | 0.6525 | 0.3526 | 404MB | 751MB | 0MB | 198MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 235.5 | 148.2 | 400 | 4328ms | 2.7102 | 3.1673 | 0.6641 | 0.4284 | 404MB | 742MB | 0MB | 310MB | Here's a thinking process that leads to the suggested summar |
| summarization | 2048 | 2042 | no-quant | 238.7 | 150.7 | 400 | 8557ms | 2.4231 | 1.1618 | 0.4420 | 0.0340 | 404MB | 785MB | 0MB | 534MB | Thinking Process:  1.  **Analyze the Request:** The user wan |
| summarization | 4096 | 4085 | no-quant | 240.4 | 155.5 | 400 | 16995ms | 3.1486 | 1.8521 | 0.2828 | 0.2823 | 404MB | 798MB | 0MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 233.6 | 149.8 | 400 | 35061ms | 1.6838 | 1.1542 | 0.4519 | 0.0555 | 404MB | 908MB | 0MB | 1.84GB | The user wants a summary of the provided text, which appears |
| summarization | 16384 | 16361 | no-quant | 235.1 | 139.1 | 400 | 69586ms | 3.4501 | 4.1337 | 0.5841 | 0.3664 | 404MB | 1.08GB | 0MB | 3.58GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 226.7 | 118.7 | 400 | 144279ms | 2.6241 | 2.7120 | 0.4759 | 0.3386 | 404MB | 1.45GB | 0MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 210.9 | 85.5 | 400 | 310394ms | 3.4304 | 3.7521 | 0.3950 | 0.2247 | 405MB | 2.19GB | 0MB | 14.07GB | The user wants a summary of the provided text, which appears |
| summarization | 131072 | 130773 | no-quant | 184.7 | 55.9 | 400 | 708114ms | 2.9019 | 2.7618 | 0.4790 | 0.1784 | 405MB | 3.75GB | 0MB | 28.02GB | The user has provided a text titled "The Great Gatsby" by F. |
