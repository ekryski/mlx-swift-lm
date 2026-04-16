# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 05:11
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-0.8B-8bit`

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
| KV cache strategy | Affine (4-bit, group 64, start 512) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | 4 |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 512 |
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
| summarization | 128 | 117 | affine-4 | 207.0 | 151.8 | 400 | 566ms | 2.7701 | 1.7448 | 0.4697 | 0.3470 | 763MB | 1.06GB | 0MB | 35MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 226.9 | 152.6 | 400 | 1098ms | 1.9498 | 2.1479 | 0.4069 | 0.2183 | 763MB | 1.07GB | 0MB | 44MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 234.0 | 153.1 | 400 | 2155ms | 4.5857 | 2.2839 | 0.3754 | 0.2302 | 763MB | 1.07GB | 0MB | 62MB | The user has provided a text snippet titled "Once again to Z |
| summarization | 1024 | 1019 | affine-4 | 239.9 | 150.8 | 400 | 4248ms | 3.2596 | 2.4690 | 0.4644 | 0.1730 | 763MB | 1.08GB | 0MB | 97MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 237.0 | 152.8 | 400 | 8617ms | 3.0689 | 3.8039 | 0.4584 | 0.2360 | 763MB | 1.12GB | 0MB | 167MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 4096 | 4085 | affine-4 | 236.8 | 150.8 | 400 | 17253ms | 3.0114 | 2.4101 | 0.3989 | 0.1776 | 763MB | 1.13GB | 0MB | 307MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-4 | 235.8 | 144.7 | 400 | 34732ms | 3.3679 | 3.0074 | 0.3292 | 0.1508 | 763MB | 1.22GB | 0MB | 587MB | The user wants a summary of the provided text, which is an e |
| summarization | 16384 | 16361 | affine-4 | 229.6 | 129.7 | 400 | 71267ms | 2.7659 | 2.8360 | 0.3487 | 0.2466 | 763MB | 1.43GB | 0MB | 1.12GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | affine-4 | 224.4 | 108.8 | 212 | 145705ms | 2.7896 | 2.0347 | 0.2547 | 0.7922 | 763MB | 1.78GB | 0MB | 2.20GB | The user wants a summary of the text provided. The text is a |
| summarization | 65536 | 65468 | affine-4 | 211.5 | 80.6 | 400 | 309608ms | 2.6657 | 2.6227 | 0.3922 | 0.1649 | 763MB | 2.53GB | 0MB | 4.40GB | Thinking Process:  1.  **Analyze the Request:**     *   **So |
| summarization | 131072 | 130773 | affine-4 | 191.4 | 54.6 | 400 | 683074ms | 2.7889 | 3.2937 | 0.2016 | 0.1680 | 763MB | 4.10GB | 0MB | 8.76GB | The user wants a summary of the text "The Great Gatsby" by F |
