# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 03:42
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
| summarization | 128 | 117 | no-quant | 222.4 | 153.6 | 400 | 526ms | 1.9265 | 1.7796 | 0.6138 | 0.4216 | 763MB | 1.07GB | 0MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 225.5 | 154.1 | 400 | 1105ms | 2.1629 | 2.5205 | 0.4567 | 0.3332 | 763MB | 1.07GB | 0MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 243.2 | 150.9 | 400 | 2073ms | 2.6952 | 1.9926 | 0.4376 | 0.2248 | 763MB | 1.07GB | 0MB | 198MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 237.1 | 153.7 | 400 | 4298ms | 2.3716 | 2.6243 | 0.4873 | 0.2682 | 763MB | 1.08GB | 0MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 241.7 | 154.9 | 400 | 8450ms | 2.3434 | 2.0985 | 0.5425 | 0.1861 | 763MB | 1.12GB | 0MB | 534MB | Thinking Process:  1.  **Analyze the Request:**     *   Sour |
| summarization | 4096 | 4085 | no-quant | 240.1 | 153.9 | 400 | 17017ms | 2.5196 | 2.3595 | 0.4839 | 0.1436 | 763MB | 1.13GB | 0MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 239.1 | 143.4 | 400 | 34258ms | 4.1481 | 3.1958 | 0.4074 | 0.1414 | 763MB | 1.22GB | 0MB | 1.84GB | The user has provided a text from "Once Again to Zelda".  He |
| summarization | 16384 | 16361 | no-quant | 234.5 | 131.7 | 400 | 69763ms | 2.9970 | 3.1131 | 0.2496 | 0.1673 | 763MB | 1.43GB | 0MB | 3.58GB | The user wants a summary of the text "The Great Gatsby" by F |
| summarization | 32768 | 32700 | no-quant | 226.6 | 108.6 | 400 | 144320ms | 3.4597 | 3.5321 | 0.2392 | 0.2329 | 763MB | 1.78GB | 0MB | 7.07GB | The user wants a summary of the provided text, which is a pa |
| summarization | 65536 | 65468 | no-quant | 214.8 | 80.9 | 400 | 304807ms | 2.7312 | 3.6193 | 0.3074 | 0.2955 | 763MB | 2.55GB | 0MB | 14.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 192.2 | 54.6 | 400 | 680379ms | 3.4631 | 2.8093 | 0.2557 | 0.1365 | 763MB | 4.10GB | 0MB | 28.02GB | The user wants a summary of the story "The Great Gatsby" by  |
