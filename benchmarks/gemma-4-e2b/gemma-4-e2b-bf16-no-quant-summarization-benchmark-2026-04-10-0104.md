# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 01:04
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
| summarization | 128 | 116 | no-quant | 945.1 | 45.7 | 252 | 123ms | 1.3111 | — | — | — | 8.66GB | 8.93GB | 6MB | 80MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 1566.9 | 44.8 | 400 | 163ms | 1.3733 | 1.3656 | — | — | 8.66GB | 9.08GB | 7MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 2052.7 | 43.8 | 400 | 245ms | 1.1735 | — | — | — | 8.66GB | 9.08GB | 7MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 2673.4 | 44.7 | 400 | 380ms | 1.3057 | — | — | — | 8.66GB | 9.21GB | 10MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 1924.9 | 44.5 | 400 | 1060ms | 1.2695 | — | — | — | 8.66GB | 9.25GB | 23MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1670.3 | 44.1 | 400 | 2453ms | 1.3589 | — | — | — | 8.66GB | 9.29GB | 38MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | no-quant | 1186.0 | 43.4 | 400 | 6909ms | 1.4457 | — | — | — | 8.66GB | 9.33GB | 70MB | 1.84GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 16384 | 16384 | no-quant | 753.2 | 41.4 | 400 | 21753ms | 1.4500 | — | — | — | 8.66GB | 9.37GB | 133MB | 3.59GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 32768 | 32768 | no-quant | 431.5 | 38.6 | 400 | 75943ms | 1.5003 | — | — | — | 8.66GB | 9.46GB | 261MB | 7.09GB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 65536 | 65536 | no-quant | 225.9 | 32.0 | 400 | 290093ms | 1.6543 | — | — | — | 8.66GB | 9.65GB | 519MB | 14.09GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | no-quant | 118.1 | 27.8 | 400 | 1105592ms | 1.5504 | — | — | — | 8.66GB | 10.20GB | 774MB | 27.98GB | <\|channel>thought Here's a thinking process to arrive at the |
