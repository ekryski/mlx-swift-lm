# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-14 12:51
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `d400abc Merge pull request #26 from TheTom/pr/minimax-m2-endpoints`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`

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
| Think start token id | 100 |
| Think end token id | 101 |
| Thinking phase prefilled | false |
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
| summarization | 128 | 116 | no-quant | 291.4 | 24.1 | 400 | 399ms | 1.9970 | 1.8393 | 3.1350 | 4.2234 | 13.48GB | 14.51GB | 0MB | 113MB | --- waywardly-thought  <\|channel>thought  <\|channel>thought  |
| summarization | 1024 | 1014 | no-quant | 104.3 | 22.4 | 400 | 9722ms | — | 1.1234 | 0.6677 | — | 13.48GB | 15.05GB | 0MB | 309MB | ---le-offers-person-bed-door- own- own- own- own- own- own-  |
| summarization | 4096 | 4094 | no-quant | 316.2 | 21.7 | 400 | 12947ms | 1.4321 | 1.2889 | 0.9010 | 0.3002 | 13.48GB | 18.04GB | 0MB | 983MB | --- <\|channel>thought  The provided text (the first chapter  |
