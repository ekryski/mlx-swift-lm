# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 08:57
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
| summarization | 128 | 117 | affine-8 | 206.4 | 134.8 | 400 | 567ms | 4.3609 | 1.2402 | 0.5853 | 0.2398 | 404MB | 732MB | 0MB | 64MB | The user has provided a book snippet titled "The Great Gatsb |
| summarization | 256 | 249 | affine-8 | 216.1 | 133.1 | 400 | 1153ms | 2.8368 | 2.1567 | 0.5191 | 0.4041 | 404MB | 742MB | 0MB | 80MB | Thinking Process:  1.  **Analyze the Request:** The user wan |
| summarization | 512 | 504 | affine-8 | 233.2 | 149.4 | 400 | 2162ms | 6.1523 | 1.6622 | 0.6528 | 0.0972 | 404MB | 752MB | 0MB | 111MB | Okay, so I need to summarize the content from the provided t |
| summarization | 1024 | 1019 | affine-8 | 230.9 | 140.6 | 400 | 4415ms | 2.8541 | 2.9320 | 0.5194 | 0.3720 | 404MB | 740MB | 0MB | 175MB | Thinking Process:  1.  **Analyze the Request:**     *   **So |
| summarization | 2048 | 2042 | affine-8 | 234.1 | 132.3 | 400 | 8722ms | 3.2609 | 2.8852 | 0.6507 | 0.4119 | 404MB | 785MB | 0MB | 300MB | Thinking Process:  1.  **Analyze the Request:**     *   Task |
| summarization | 4096 | 4085 | affine-8 | 231.3 | 151.7 | 399 | 17661ms | 2.6253 | 2.3781 | 0.6690 | 0.3700 | 404MB | 810MB | 0MB | 552MB | Thinking Process:  1.  **Analyze the Request:**     *   Task |
| summarization | 8192 | 8190 | affine-8 | 227.2 | 111.4 | 400 | 36055ms | 3.1723 | 3.2995 | 0.5574 | 0.3031 | 404MB | 897MB | 0MB | 1.03GB | The user wants a summary of the provided text. The text is f |
| summarization | 16384 | 16361 | affine-8 | 222.2 | 136.5 | 400 | 73642ms | 2.9461 | 3.9022 | 0.4215 | 0.3722 | 404MB | 1.08GB | 0MB | 2.01GB | The user is asking for a summary of the provided text, which |
| summarization | 32768 | 32700 | affine-8 | 222.1 | 113.1 | 400 | 147258ms | 3.1246 | 2.8492 | 0.4203 | 0.3941 | 404MB | 1.43GB | 0MB | 3.98GB | The user has provided a text from F. Scott Fitzgerald's nove |
| summarization | 65536 | 65468 | affine-8 | 206.4 | 85.0 | 400 | 317265ms | 3.2825 | 4.7041 | 0.4794 | 0.2959 | 405MB | 2.18GB | 0MB | 7.91GB | The user wants a summary of the provided text. The text is a |
| summarization | 131072 | 130773 | affine-8 | 165.6 | 54.6 | 400 | 789908ms | 3.4518 | 1.1499 | 0.4509 | 0.0067 | 405MB | 3.75GB | 0MB | 15.76GB | The user wants a summary of the provided text, which is the  |
