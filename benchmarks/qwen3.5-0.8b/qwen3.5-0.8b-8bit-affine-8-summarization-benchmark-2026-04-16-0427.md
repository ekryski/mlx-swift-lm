# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 04:27
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
| KV cache strategy | Affine (8-bit, group 64, start 512) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | 8 |
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
| summarization | 128 | 117 | affine-8 | 209.5 | 153.0 | 400 | 559ms | 2.1947 | 1.7627 | 0.3825 | 0.2988 | 763MB | 1.07GB | 0MB | 64MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-8 | 222.8 | 155.1 | 400 | 1118ms | 2.4021 | 2.3983 | 0.3864 | 0.2569 | 763MB | 1.07GB | 0MB | 80MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-8 | 234.0 | 154.8 | 400 | 2154ms | 3.9958 | 3.8634 | 0.3456 | 0.1839 | 763MB | 1.07GB | 0MB | 111MB | The user wants a summary of the provided text. I need to car |
| summarization | 1024 | 1019 | affine-8 | 237.0 | 148.0 | 400 | 4300ms | 2.9148 | 2.4472 | 0.5312 | 0.1658 | 763MB | 1.07GB | 0MB | 175MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-8 | 235.8 | 152.3 | 400 | 8661ms | 2.8181 | 2.3409 | 0.4112 | 0.1421 | 763MB | 1.12GB | 0MB | 300MB | Thinking Process:  1.  **Analyze the Request:**     *   Sour |
| summarization | 4096 | 4085 | affine-8 | 236.2 | 151.7 | 400 | 17297ms | 2.3885 | 2.2777 | 0.5533 | 0.1394 | 763MB | 1.14GB | 0MB | 552MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 8192 | 8190 | affine-8 | 236.3 | 144.0 | 400 | 34658ms | 2.9572 | 2.5039 | 0.4795 | 0.1999 | 763MB | 1.22GB | 0MB | 1.03GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-8 | 231.4 | 132.1 | 400 | 70699ms | 2.1932 | 3.0822 | 0.3144 | 0.1481 | 763MB | 1.43GB | 0MB | 2.01GB | Thinking Process:  1.  **Analyze the Request:**     *   **In |
| summarization | 32768 | 32700 | affine-8 | 227.1 | 109.8 | 400 | 143972ms | 3.3279 | 2.4524 | 0.3229 | 0.1024 | 763MB | 1.78GB | 0MB | 3.98GB | The user has asked for a summary of the provided text, which |
| summarization | 65536 | 65468 | affine-8 | 215.4 | 81.5 | 400 | 303967ms | 2.9522 | 2.4091 | 0.3341 | 0.1610 | 763MB | 2.53GB | 0MB | 7.91GB | The user wants a summary of the provided text, which is an e |
| summarization | 131072 | 130773 | affine-8 | 191.8 | 54.6 | 400 | 681808ms | 3.6314 | 2.9971 | 0.3050 | 0.1811 | 763MB | 4.10GB | 0MB | 15.76GB | The user is asking for a summary of the provided text. The t |
