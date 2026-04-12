# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-11 05:09
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
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
| KV group size | 64 |
| Quantized KV start | 0 |
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
| summarization | 128 | 129 | turbo4v2 | 226.0 | 13.5 | 400 | 571ms | 1.7703 | 1.5259 | 0.7539 | 0.2215 | 16.56GB | 17.71GB | 26MB | 24MB | The user posted a text that appears to be an excerpt of "The |
| summarization | 256 | 257 | turbo4v2 | 40.9 | 19.5 | 400 | 6282ms | 1.9497 | 1.3511 | 0.5848 | 0.1103 | 16.56GB | 18.36GB | 45MB | 29MB | We need to respond. The user posted a text: "The Great Gatsb |
| summarization | 512 | 513 | turbo4v2 | 84.1 | 13.5 | 398 | 6099ms | 1.2365 | 1.1964 | 0.6058 | 0.2721 | 16.56GB | 19.78GB | 40MB | 40MB | Okay, the user has shared a passage from "The Great Gatsby"  |
| summarization | 1024 | 1025 | turbo4v2 | 129.1 | 13.5 | 400 | 7938ms | 2.0217 | 1.5049 | 1.1060 | 0.4321 | 16.56GB | 24.74GB | 60MB | 63MB | We need to respond to the user's input. The user posted the  |
| summarization | 2048 | 2049 | turbo4v2 | 113.0 | 13.6 | 400 | 18132ms | 1.7957 | 1.3747 | 1.3354 | 0.2397 | 16.56GB | 43.98GB | 68MB | 109MB | We need to respond to the user. The user posted the beginnin |
| summarization | 4096 | 4097 | turbo4v2 | 95.6 | 13.3 | 400 | 42865ms | 2.0444 | 1.8028 | 0.9385 | 0.2791 | 16.56GB | 44.14GB | 71MB | 200MB | We need to respond to the user's request. The user posted a  |
| summarization | 8192 | 8181 | turbo4v2 | 147.6 | 13.2 | 400 | 55431ms | 1.8573 | 1.6677 | 0.2962 | 0.1720 | 16.56GB | 44.16GB | 92MB | 381MB | We need to respond: Summarize the content above. The content |
| summarization | 16384 | 16344 | turbo4v2 | 226.4 | 12.9 | 400 | 72204ms | 1.7830 | 1.9660 | 0.4078 | 0.1558 | 16.56GB | 44.25GB | 181MB | 744MB | We need to provide a summary of the provided content. The co |
| summarization | 32768 | 32553 | turbo4v2 | 283.7 | 12.3 | 400 | 114759ms | 2.1602 | 1.5915 | 0.2027 | 0.1410 | 16.56GB | 44.44GB | 315MB | 1.43GB | We need to summarize the content above, which is a weird mas |
| summarization | 65536 | 64945 | turbo4v2 | 299.7 | 11.2 | 400 | 216734ms | 2.2457 | 1.7377 | 0.0622 | 0.1319 | 16.56GB | 44.82GB | 145MB | 2.84GB | We need to summarize the given text. It's a long passage, pr |
| summarization | 131072 | 129844 | turbo4v2 | 263.9 | 9.0 | 400 | 492067ms | 1.9940 | 1.7490 | 0.0692 | 0.0814 | 16.56GB | 45.57GB | 565MB | 5.65GB | 1. The Great Gatsby. 2. The Age of Innocence 3. The Great Ga |
