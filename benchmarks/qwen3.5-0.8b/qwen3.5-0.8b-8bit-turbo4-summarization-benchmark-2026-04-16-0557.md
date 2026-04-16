# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 05:57
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
| KV cache strategy | TurboQuant (turbo4) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4 |
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
| summarization | 128 | 117 | turbo4 | 204.6 | 152.4 | 400 | 572ms | 2.2816 | 3.5734 | 0.5731 | 0.5044 | 763MB | 1.06GB | 0MB | 30MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4 | 224.7 | 152.7 | 400 | 1108ms | 3.1668 | 2.5620 | 0.4654 | 0.3223 | 763MB | 1.07GB | 0MB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4 | 242.5 | 154.0 | 400 | 2079ms | 2.3030 | 2.7462 | 0.4677 | 0.2992 | 763MB | 1.07GB | 0MB | 53MB | Thinking Process:  1.  **Analyze the Request:**     *   **In |
| summarization | 1024 | 1019 | turbo4 | 238.9 | 151.7 | 400 | 4266ms | 2.3647 | 2.2933 | 0.5063 | 0.1675 | 763MB | 1.08GB | 0MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 238.7 | 151.0 | 400 | 8555ms | 3.4633 | 1.3172 | 0.3551 | 0.0365 | 763MB | 1.12GB | 0MB | 142MB | The user wants a summary of the provided text from "Once aga |
| summarization | 4096 | 4085 | turbo4 | 238.6 | 150.1 | 400 | 17121ms | 2.1018 | 2.3475 | 0.4458 | 0.1912 | 763MB | 1.13GB | 0MB | 261MB | Thinking Process:  1.  **Analyze the Request:**     *   Sour |
| summarization | 8192 | 8190 | turbo4 | 239.7 | 143.8 | 400 | 34169ms | 3.0133 | 2.9063 | 0.3182 | 0.2221 | 763MB | 1.22GB | 0MB | 499MB | The user wants a summary of the provided text, which is an e |
| summarization | 16384 | 16361 | turbo4 | 233.3 | 131.0 | 400 | 70134ms | 2.8083 | 3.1208 | 0.2804 | 0.1357 | 763MB | 1.43GB | 0MB | 974MB | The user wants a summary of the provided text, which is "The |
| summarization | 32768 | 32700 | turbo4 | 228.1 | 109.0 | 400 | 143371ms | 3.1119 | 2.9462 | 0.2614 | 0.1410 | 763MB | 1.79GB | 0MB | 1.88GB | The user wants a summary of the provided text, which is a lo |
| summarization | 65536 | 65468 | turbo4 | 215.0 | 81.1 | 400 | 304450ms | 3.5663 | 3.1366 | 0.2721 | 0.0881 | 763MB | 2.54GB | 0MB | 3.74GB | The user wants a summary of the provided text. The text is a |
| summarization | 131072 | 130773 | turbo4 | 190.4 | 54.3 | 400 | 686744ms | 3.1355 | 2.7818 | 0.2909 | 0.1933 | 763MB | 4.10GB | 0MB | 7.44GB | The user wants a summary of the provided text, which is "The |
