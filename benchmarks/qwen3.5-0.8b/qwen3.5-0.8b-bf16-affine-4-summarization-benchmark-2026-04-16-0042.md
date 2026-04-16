# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 00:42
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
| summarization | 128 | 117 | affine-4 | 234.5 | 120.6 | 400 | 499ms | 2.9192 | 2.4262 | 0.4683 | 0.3332 | 1.40GB | 1.67GB | 0MB | 35MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 221.2 | 120.3 | 400 | 1126ms | 2.5956 | 2.4340 | 0.4975 | 0.3591 | 1.40GB | 1.67GB | 0MB | 44MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 224.8 | 118.5 | 400 | 2243ms | 2.4841 | 3.4444 | 0.4055 | 0.2196 | 1.40GB | 1.70GB | 0MB | 62MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 230.6 | 119.3 | 400 | 4419ms | 2.8274 | 2.6901 | 0.4277 | 0.2026 | 1.40GB | 1.73GB | 0MB | 97MB | Here's a thinking process that leads to the suggested summar |
| summarization | 2048 | 2042 | affine-4 | 239.0 | 117.2 | 400 | 8546ms | 2.4919 | 3.6280 | 0.4056 | 0.1562 | 1.40GB | 1.82GB | 0MB | 167MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 239.0 | 115.3 | 400 | 17095ms | 2.6019 | 2.1622 | 0.4714 | 0.2089 | 1.40GB | 1.81GB | 0MB | 307MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-4 | 237.7 | 109.1 | 400 | 34463ms | 2.9593 | 2.9784 | 0.5292 | 0.2488 | 1.40GB | 1.89GB | 0MB | 587MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-4 | 234.9 | 100.6 | 400 | 69643ms | 4.2013 | 1.2613 | 0.4296 | 0.0409 | 1.40GB | 2.14GB | 0MB | 1.12GB | 用户希望对提供的文本进行总结。我需要仔细阅读给定的两段选段文本，分析其内容和主题。  首先，我注意到提供的文本包含两部分 |
| summarization | 32768 | 32700 | affine-4 | 227.4 | 87.3 | 400 | 143821ms | 3.2621 | 3.2166 | 0.4510 | 0.3265 | 1.40GB | 2.46GB | 0MB | 2.21GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | affine-4 | 210.7 | 68.2 | 400 | 310781ms | 2.7122 | 2.5296 | 0.2926 | 0.1373 | 1.40GB | 3.19GB | 0MB | 4.40GB | The user wants a summary of the provided text, which is "The |
| summarization | 131072 | 130773 | affine-4 | 192.0 | 48.6 | 400 | 681000ms | 3.0003 | 2.8433 | 0.2267 | 0.0322 | 1.40GB | 4.75GB | 0MB | 8.76GB | The user wants a summary of the provided text, which is clea |
