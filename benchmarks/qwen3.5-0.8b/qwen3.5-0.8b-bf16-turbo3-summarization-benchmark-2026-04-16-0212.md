# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 02:12
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
| KV cache strategy | TurboQuant (turbo3) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo3 |
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
| summarization | 128 | 117 | turbo3 | 224.7 | 119.7 | 400 | 521ms | 2.2356 | 2.0524 | 0.5003 | 0.2829 | 1.40GB | 1.70GB | 0MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo3 | 228.7 | 119.6 | 400 | 1089ms | 2.0429 | 2.2908 | 0.4025 | 0.2328 | 1.40GB | 1.67GB | 0MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 227.3 | 117.1 | 400 | 2218ms | 3.3116 | 2.1739 | 0.3843 | 0.1804 | 1.40GB | 1.68GB | 0MB | 40MB | The user is asking me to summarize a specific text from "The |
| summarization | 1024 | 1019 | turbo3 | 236.3 | 116.8 | 400 | 4312ms | 2.4657 | 2.3427 | 0.5253 | 0.2317 | 1.40GB | 1.72GB | 0MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo3 | 240.3 | 118.0 | 400 | 8499ms | 2.3875 | 2.2693 | 0.5464 | 0.1778 | 1.40GB | 1.82GB | 0MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo3 | 241.6 | 114.7 | 400 | 16910ms | 1.9968 | 2.4402 | 0.4390 | 0.2079 | 1.40GB | 1.81GB | 0MB | 199MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 8192 | 8190 | turbo3 | 239.4 | 108.2 | 400 | 34211ms | 2.4515 | 2.2010 | 0.3896 | 0.2898 | 1.40GB | 1.88GB | 0MB | 382MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 16384 | 16361 | turbo3 | 236.7 | 101.0 | 400 | 69133ms | 3.6410 | 2.6246 | 0.3268 | 0.1607 | 1.40GB | 2.14GB | 0MB | 745MB | The user wants a summary of the provided text. The text is f |
| summarization | 32768 | 32700 | turbo3 | 229.8 | 86.6 | 400 | 142290ms | 2.7892 | 2.9957 | 0.4117 | 0.2233 | 1.40GB | 2.44GB | 0MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo3 | 214.5 | 68.3 | 400 | 305195ms | 3.2905 | 3.4017 | 0.2245 | 0.1514 | 1.40GB | 3.19GB | 0MB | 2.86GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo3 | 193.9 | 48.7 | 400 | 674291ms | 4.0113 | 2.3210 | 0.2118 | 0.1290 | 1.40GB | 4.75GB | 0MB | 5.69GB | The user is asking for a summary of the provided text, which |
