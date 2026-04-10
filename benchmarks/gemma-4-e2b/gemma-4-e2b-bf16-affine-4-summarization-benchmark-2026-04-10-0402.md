# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 04:02
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: bf16
- **Model**: `mlx-community/gemma-4-e2b-it-bf16`

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
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | affine-4 | 1088.1 | 45.0 | 400 | 107ms | 1.5056 | 1.5481 | 1.1254 | 1.5087 | 8.66GB | 8.82GB | 6MB | 35MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | affine-4 | 723.6 | 44.1 | 395 | 354ms | 1.3262 | — | 0.7599 | — | 8.66GB | 8.96GB | 5MB | 44MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | affine-4 | 1245.3 | 43.2 | 400 | 405ms | 1.2716 | — | 0.3666 | — | 8.66GB | 9.05GB | 10MB | 62MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | affine-4 | 1853.5 | 44.2 | 400 | 549ms | 1.2518 | — | 0.2092 | — | 8.66GB | 9.21GB | 14MB | 97MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | affine-4 | 1938.7 | 43.9 | 400 | 1052ms | 1.2379 | — | 0.3028 | — | 8.66GB | 9.25GB | 22MB | 167MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | affine-4 | 1650.8 | 43.5 | 400 | 2482ms | 1.4324 | — | 0.1629 | — | 8.66GB | 9.29GB | 39MB | 307MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | affine-4 | 1196.5 | 42.7 | 400 | 6848ms | 1.4048 | — | 0.5773 | — | 8.66GB | 9.33GB | 71MB | 587MB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 16384 | 16384 | affine-4 | 756.1 | 40.8 | 400 | 21669ms | 1.4906 | — | 0.5566 | — | 8.66GB | 9.37GB | 134MB | 1.12GB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 32768 | 32768 | affine-4 | 430.3 | 38.1 | 400 | 76153ms | 1.6866 | — | 0.6669 | — | 8.66GB | 9.46GB | 262MB | 2.21GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 65536 | 65536 | affine-4 | 231.1 | 32.4 | 400 | 283576ms | 1.5275 | — | 0.4602 | — | 8.66GB | 9.65GB | 517MB | 4.40GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | affine-4 | 120.2 | 27.8 | 400 | 1085989ms | 1.5560 | — | 0.0328 | — | 8.66GB | 10.20GB | 257MB | 8.74GB | <\|channel>thought Here's a thinking process to arrive at the |
