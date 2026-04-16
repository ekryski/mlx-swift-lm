# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 01:27
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
| summarization | 128 | 117 | turbo4 | 228.8 | 120.4 | 400 | 512ms | 2.5099 | 1.5657 | 0.5306 | 0.3066 | 1.40GB | 1.66GB | 0MB | 30MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4 | 228.0 | 119.0 | 400 | 1092ms | 2.2561 | 1.9460 | 0.4232 | 0.1801 | 1.40GB | 1.70GB | 0MB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4 | 236.9 | 118.2 | 400 | 2128ms | 2.2085 | 2.4963 | 0.5183 | 0.1702 | 1.40GB | 1.70GB | 0MB | 53MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4 | 231.8 | 118.6 | 400 | 4397ms | 2.4767 | 2.5255 | 0.4170 | 0.2618 | 1.40GB | 1.72GB | 0MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 241.6 | 117.3 | 400 | 8453ms | 2.8894 | 2.8726 | 0.5560 | 0.1936 | 1.40GB | 1.82GB | 0MB | 142MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 4096 | 4085 | turbo4 | 238.8 | 114.9 | 400 | 17105ms | 2.0801 | 3.5377 | 0.4747 | 0.2400 | 1.40GB | 1.81GB | 0MB | 261MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4 | 238.9 | 109.4 | 400 | 34279ms | 3.4087 | 3.2572 | 0.4821 | 0.1808 | 1.40GB | 1.89GB | 0MB | 499MB | The user wants a summary of the text provided. The text is * |
| summarization | 16384 | 16361 | turbo4 | 235.2 | 101.2 | 400 | 69558ms | 2.8298 | 2.5193 | 0.2690 | 0.0122 | 1.40GB | 2.14GB | 0MB | 974MB | The user is asking for a summary of the provided text, which |
| summarization | 32768 | 32700 | turbo4 | 228.3 | 87.2 | 400 | 143248ms | 3.2395 | 3.1420 | 0.3088 | 0.1594 | 1.40GB | 2.44GB | 0MB | 1.88GB | The user wants a summary of the provided text, which appears |
| summarization | 65536 | 65468 | turbo4 | 216.1 | 68.6 | 400 | 302920ms | 2.7347 | 1.5244 | 0.2325 | 0.0882 | 1.40GB | 3.23GB | 0MB | 3.74GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo4 | 194.2 | 48.4 | 400 | 673533ms | 2.7448 | 2.3092 | 0.2965 | 0.1684 | 1.40GB | 4.75GB | 0MB | 7.44GB | The user wants a summary of the text "The Great Gatsby" by F |
