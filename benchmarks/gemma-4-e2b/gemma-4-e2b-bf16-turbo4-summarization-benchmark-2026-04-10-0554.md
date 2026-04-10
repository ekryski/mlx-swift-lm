# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 05:54
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
| summarization | 128 | 116 | turbo4 | 1080.4 | 45.1 | 330 | 108ms | 1.4973 | — | 1.8342 | — | 8.66GB | 8.82GB | 4MB | 26MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | turbo4 | 721.0 | 44.4 | 400 | 355ms | 1.2974 | — | 0.6164 | — | 8.66GB | 8.96GB | 6MB | 38MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | turbo4 | 1147.3 | 43.4 | 400 | 439ms | 1.1993 | — | 0.3680 | — | 8.66GB | 9.05GB | 10MB | 52MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | turbo4 | 1794.4 | 44.3 | 400 | 567ms | 1.2588 | — | 0.2961 | — | 8.66GB | 9.21GB | 13MB | 82MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | turbo4 | 1940.4 | 44.1 | 400 | 1051ms | 1.2845 | — | 0.1169 | — | 8.66GB | 9.25GB | 23MB | 142MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | turbo4 | 1614.4 | 43.7 | 400 | 2537ms | 1.3397 | — | 0.1955 | — | 8.66GB | 9.29GB | 38MB | 261MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | turbo4 | 1196.8 | 42.9 | 400 | 6846ms | 1.5472 | — | 0.7925 | — | 8.66GB | 9.33GB | 69MB | 499MB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 16384 | 16384 | turbo4 | 756.3 | 40.8 | 400 | 21665ms | 1.3814 | — | 0.6816 | — | 8.66GB | 9.37GB | 100MB | 975MB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 32768 | 32768 | turbo4 | 428.3 | 38.1 | 400 | 76507ms | 1.6073 | — | 0.5389 | — | 8.66GB | 9.46GB | 262MB | 1.88GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 65536 | 65536 | turbo4 | 229.7 | 32.2 | 400 | 285356ms | 1.4654 | — | 0.3613 | — | 8.66GB | 9.65GB | 518MB | 3.74GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | turbo4 | 119.4 | 27.7 | 400 | 1093417ms | 1.5452 | — | 0.0558 | — | 8.66GB | 10.20GB | 773MB | 7.43GB | <\|channel>thought Here's a thinking process to arrive at the |
