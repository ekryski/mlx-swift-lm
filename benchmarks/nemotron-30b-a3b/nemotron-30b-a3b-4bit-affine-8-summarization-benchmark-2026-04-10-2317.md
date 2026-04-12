# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-10 23:17
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `b52bbd7 updating gemma4-e2b 4bit turbo4v2 benchmark without thinking`
- **Quantization**: 4bit
- **Model**: `mlx-community/Nemotron-Cascade-2-30B-A3B-4bit`

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
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | nil |
| Repetition context size | 20 |
| Presence penalty | nil |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | 12 |
| Think end token id | 13 |
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
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 100 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 129 | affine-8 | 257.6 | 26.0 | 398 | 501ms | 1.3425 | 1.2436 | 0.2254 | 0.1404 | 16.56GB | 17.71GB | 49MB | 65MB | Okay, the user has shared a partial text that looks like the |
| summarization | 256 | 257 | affine-8 | 26.7 | 26.8 | 400 | 9610ms | 1.9530 | 1.4004 | 0.5825 | 0.2433 | 16.56GB | 18.36GB | 50MB | 81MB | The user posted text: "The Great Gatsby... Table of Contents |
| summarization | 512 | 513 | affine-8 | 59.9 | 26.6 | 400 | 8565ms | 1.7851 | 1.3686 | 0.7009 | 0.2516 | 16.56GB | 19.78GB | 50MB | 112MB | We need to respond. The user posted a text that appears to b |
| summarization | 1024 | 1025 | affine-8 | 105.8 | 26.5 | 400 | 9691ms | 1.9500 | 1.4369 | 0.9030 | 0.2828 | 16.56GB | 24.74GB | 49MB | 175MB | We need to respond to the user. The user posted a text that  |
| summarization | 2048 | 2049 | affine-8 | 113.3 | 26.4 | 400 | 18086ms | 1.7319 | 1.0354 | 0.9619 | 0.0383 | 16.56GB | 43.98GB | 56MB | 301MB | We need to respond to the user. The user posted a text: "The |
| summarization | 4096 | 4097 | affine-8 | 88.0 | 26.3 | 400 | 46554ms | 1.4087 | 1.2912 | 0.3358 | 0.3087 | 16.56GB | 44.14GB | 46MB | 553MB | Okay, the user has shared a lengthy excerpt from F. Scott Fi |
| summarization | 8192 | 8181 | affine-8 | 150.9 | 25.6 | 400 | 54200ms | 1.7299 | 2.2091 | 0.1516 | 0.1577 | 16.56GB | 44.16GB | 76MB | 1.03GB | We need to summarize the content above, which presumably is  |
| summarization | 16384 | 16344 | affine-8 | 229.9 | 26.4 | 400 | 71079ms | 1.7841 | 1.6119 | 0.0942 | 0.0866 | 16.56GB | 44.25GB | 98MB | 2.01GB | We need to summarize the provided text, which appears to be  |
| summarization | 32768 | 32553 | affine-8 | 282.3 | 24.3 | 400 | 115313ms | 1.7216 | 1.4497 | 0.2052 | 0.0497 | 16.56GB | 44.44GB | 133MB | 3.96GB | We need to provide a summary of the provided text. It's a fa |
| summarization | 65536 | 64945 | affine-8 | 297.0 | 22.8 | 400 | 218672ms | 2.2436 | 1.4301 | 0.0610 | 0.0611 | 16.56GB | 44.82GB | 223MB | 7.85GB | The user provides a huge excerpt from The Great Gatsby, but  |
| summarization | 131072 | 129844 | affine-8 | 262.3 | 20.1 | 400 | 495080ms | 2.4425 | 1.4988 | 0.0365 | 0.0366 | 16.56GB | 45.57GB | 396MB | 15.65GB | This is a summary of the text, which is the first chapter of |
