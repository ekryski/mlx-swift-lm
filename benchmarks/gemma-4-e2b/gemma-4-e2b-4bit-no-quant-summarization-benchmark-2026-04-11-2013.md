# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 20:13
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `a24b6da docs: expanded setup guide — prerequisites, model download, agent configs`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| KV cache strategy | None (FP16) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 2048 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Repetition penalty | nil |
| Repetition context size | 20 |
| Presence penalty | nil |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Collect per-token data | false |
| Track perplexity | false |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 0 |
| Thinking (effective) | No |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
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
| summarization | 128 | 110 | no-quant | 893.6 | 102.0 | 203 | 124ms | — | — | — | — | 2.45GB | 2.49GB | 4MB | 68MB | The provided text is a highly fragmented and incomplete coll |
| summarization | 1024 | 1008 | no-quant | 2132.0 | 99.9 | 400 | 474ms | — | — | — | — | 2.45GB | 2.51GB | 4MB | 308MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 4096 | 4088 | no-quant | 2580.0 | 98.2 | 400 | 1586ms | — | — | — | — | 2.45GB | 2.58GB | 4MB | 982MB | This excerpt is a collection of prose fragments from F. Scot |
| summarization | 32768 | 32768 | no-quant | 847.1 | 73.0 | 400 | 38695ms | — | — | — | — | 2.45GB | 3.22GB | 196MB | 7.09GB | This is a very dense and complex excerpt from a work that ap |
