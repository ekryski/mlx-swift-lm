# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 07:26
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
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
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
| summarization | 128 | 117 | turbo4v2 | 210.6 | 147.5 | 400 | 556ms | 2.8608 | 1.7920 | 0.5674 | 0.4035 | 763MB | 1.07GB | 0MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Sour |
| summarization | 256 | 249 | turbo4v2 | 223.0 | 148.5 | 400 | 1117ms | 2.1661 | 2.2181 | 0.4420 | 0.1990 | 763MB | 1.07GB | 0MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4v2 | 238.0 | 148.6 | 400 | 2118ms | 2.4015 | 1.9492 | 0.5684 | 0.2656 | 763MB | 1.08GB | 0MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 236.8 | 152.6 | 400 | 4304ms | 2.8425 | 2.1014 | 0.5192 | 0.1769 | 763MB | 1.08GB | 0MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4v2 | 233.2 | 151.2 | 400 | 8757ms | 3.9759 | 3.3042 | 0.4437 | 0.0936 | 763MB | 1.12GB | 0MB | 109MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 4096 | 4085 | turbo4v2 | 234.1 | 149.0 | 400 | 17452ms | 2.3341 | 2.7449 | 0.4258 | 0.2860 | 763MB | 1.14GB | 0MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4v2 | 232.7 | 142.9 | 400 | 35193ms | 3.3431 | 3.4964 | 0.6221 | 0.2051 | 763MB | 1.22GB | 0MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4v2 | 230.7 | 131.0 | 400 | 70933ms | 3.5183 | 3.4902 | 0.3137 | 0.2015 | 763MB | 1.43GB | 0MB | 745MB | The user wants a summary of the provided text, which is a de |
| summarization | 32768 | 32700 | turbo4v2 | 227.8 | 109.8 | 400 | 143568ms | 2.7823 | 2.9436 | 0.4801 | 0.2030 | 763MB | 1.79GB | 0MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4v2 | 214.9 | 81.3 | 400 | 304637ms | 3.2269 | 3.2533 | 0.3052 | 0.1890 | 763MB | 2.53GB | 0MB | 2.86GB | The user wants a summary of the provided text, which is a fa |
| summarization | 131072 | 130773 | turbo4v2 | 191.6 | 54.5 | 400 | 682659ms | 2.7509 | 3.5401 | 0.2399 | 0.1331 | 763MB | 4.10GB | 0MB | 5.69GB | The user wants a summary of the provided text. The text is a |
