# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-10 02:10
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: bf16
**Model**: `mlx-community/gemma-4-e2b-it-bf16`

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
| summarization | 128 | 116 | affine-8 | 1222.4 | 44.8 | 400 | 95ms | 1.0830 | — | 0.7316 | — | 8.66GB | 8.82GB | 4MB | 63MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | affine-8 | 742.4 | 44.2 | 395 | 345ms | 1.4541 | — | 0.7857 | — | 8.66GB | 8.96GB | 8MB | 80MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | affine-8 | 1091.0 | 43.2 | 400 | 462ms | 1.1991 | — | 0.2857 | — | 8.66GB | 9.05GB | 10MB | 111MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | affine-8 | 1849.2 | 44.2 | 400 | 550ms | 1.2461 | — | 0.2033 | — | 8.66GB | 9.21GB | 14MB | 174MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | affine-8 | 1953.1 | 44.0 | 400 | 1044ms | 1.3517 | — | 0.1746 | — | 8.66GB | 9.25GB | 22MB | 300MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | affine-8 | 1640.8 | 43.5 | 400 | 2497ms | 1.4627 | — | 0.1836 | — | 8.66GB | 9.29GB | 38MB | 553MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | affine-8 | 1194.4 | 42.7 | 400 | 6860ms | 1.4613 | — | 0.8958 | — | 8.66GB | 9.33GB | 69MB | 1.03GB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 16384 | 16384 | affine-8 | 757.5 | 41.0 | 400 | 21631ms | 1.6526 | — | 0.6352 | — | 8.66GB | 9.37GB | 134MB | 2.02GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 32768 | 32768 | affine-8 | 430.5 | 38.3 | 400 | 76123ms | 1.5749 | — | 0.5299 | — | 8.66GB | 9.46GB | 262MB | 3.99GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 65536 | 65536 | affine-8 | 230.2 | 32.6 | 400 | 284694ms | 1.4480 | — | 0.4128 | — | 8.66GB | 9.65GB | 518MB | 7.92GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | affine-8 | 119.7 | 27.6 | 400 | 1090727ms | 1.5457 | — | 0.0339 | — | 8.66GB | 10.20GB | 774MB | 15.74GB | <\|channel>thought Here's a thinking process to arrive at the |
