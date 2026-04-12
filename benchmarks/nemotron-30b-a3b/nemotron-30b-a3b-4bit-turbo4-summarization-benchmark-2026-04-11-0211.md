# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-11 02:11
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
| KV cache strategy | TurboQuant (turbo4) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4 |
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
| summarization | 128 | 129 | turbo4 | 227.4 | 13.5 | 400 | 568ms | 2.0581 | 1.3743 | 0.8801 | 0.2505 | 16.56GB | 17.71GB | 45MB | 31MB | The user posted a snippet of text: "The Great Gatsby" with a |
| summarization | 256 | 257 | turbo4 | 46.7 | 13.5 | 400 | 5502ms | 1.2893 | 1.1199 | 0.4302 | 0.2481 | 16.56GB | 18.42GB | 54MB | 38MB | Okay, the user has shared a passage from "The Great Gatsby"  |
| summarization | 512 | 513 | turbo4 | 58.3 | 13.4 | 400 | 8801ms | 1.9740 | 1.7303 | 0.8246 | 0.2992 | 16.56GB | 19.81GB | 34MB | 53MB | We need to respond to the user. The user posted a text that  |
| summarization | 1024 | 1025 | turbo4 | 98.1 | 13.4 | 400 | 10448ms | 1.8351 | 1.3720 | 0.6935 | 0.2509 | 16.56GB | 24.74GB | 60MB | 83MB | The user posted a large block of text that seems to be a mod |
| summarization | 2048 | 2049 | turbo4 | 119.3 | 13.3 | 400 | 17177ms | 2.0458 | 1.3746 | 0.7445 | 0.3418 | 16.56GB | 43.98GB | 58MB | 142MB | The user posted a large block of text that seems to be a mas |
| summarization | 4096 | 4097 | turbo4 | 81.3 | 13.3 | 400 | 50366ms | 1.7593 | 1.4320 | 1.4892 | 0.1515 | 16.56GB | 44.14GB | 83MB | 261MB | We need to respond. The user posted a long text that seems l |
| summarization | 8192 | 8181 | turbo4 | 168.6 | 13.0 | 400 | 48513ms | 1.9120 | 1.6021 | 0.2861 | 0.0630 | 16.56GB | 44.16GB | 74MB | 499MB | We need to summarize the content above, which appears to be  |
| summarization | 16384 | 16344 | turbo4 | 236.5 | 12.7 | 400 | 69116ms | 2.1042 | 2.1680 | 0.3209 | 0.0866 | 16.56GB | 44.25GB | 109MB | 973MB | We need to provide a summary of the content above. The conte |
| summarization | 32768 | 32553 | turbo4 | 287.2 | 12.1 | 400 | 113340ms | 1.8181 | 2.2936 | 0.1877 | 0.0740 | 16.56GB | 44.44GB | 323MB | 1.87GB | We need to provide a summary of the provided text. The text  |
| summarization | 65536 | 64945 | turbo4 | 297.9 | 11.0 | 400 | 218005ms | 1.8795 | 1.5180 | 0.0736 | 0.0509 | 16.56GB | 44.82GB | 323MB | 3.71GB | We need to summarize the content above, which is a lengthy t |
| summarization | 131072 | 129844 | turbo4 | 264.1 | 8.8 | 400 | 491673ms | 1.9521 | 1.9537 | 0.0459 | 0.0590 | 16.56GB | 45.57GB | 1.11GB | 7.39GB | We need to produce a summary. The user gave a text that is a |
