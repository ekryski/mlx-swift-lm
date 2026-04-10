# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 09:38
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
| summarization | 128 | 116 | turbo4v2 | 1239.3 | 45.0 | 366 | 94ms | 1.7133 | — | 1.6697 | — | 8.66GB | 8.82GB | 5MB | 21MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | turbo4v2 | 742.3 | 44.5 | 400 | 345ms | 1.3233 | — | 0.5981 | — | 8.66GB | 8.96GB | 8MB | 29MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | turbo4v2 | 1267.2 | 43.6 | 400 | 398ms | 1.4183 | — | 0.3914 | — | 8.66GB | 9.05GB | 10MB | 40MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | turbo4v2 | 1830.9 | 44.4 | 400 | 555ms | 1.2784 | — | 0.1310 | — | 8.66GB | 9.21GB | 13MB | 63MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | turbo4v2 | 1957.8 | 44.4 | 400 | 1042ms | 1.3748 | — | 0.2369 | — | 8.66GB | 9.25GB | 22MB | 108MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | turbo4v2 | 1607.1 | 43.2 | 400 | 2549ms | 1.3904 | — | 0.2197 | — | 8.66GB | 9.29GB | 39MB | 200MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | turbo4v2 | 1175.1 | 42.5 | 400 | 6973ms | 1.5896 | — | 0.8549 | — | 8.66GB | 9.33GB | 71MB | 382MB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 16384 | 16384 | turbo4v2 | 753.7 | 40.9 | 400 | 21739ms | 1.4738 | — | 0.5511 | — | 8.66GB | 9.37GB | 134MB | 746MB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 32768 | 32768 | turbo4v2 | 422.9 | 38.1 | 400 | 77481ms | 1.5968 | — | 0.6111 | — | 8.66GB | 9.46GB | 262MB | 1.44GB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 65536 | 65536 | turbo4v2 | 228.0 | 32.3 | 400 | 287453ms | 1.4785 | — | 0.4016 | — | 8.66GB | 9.65GB | 518MB | 2.86GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | turbo4v2 | 118.7 | 27.7 | 400 | 1099872ms | 1.5495 | — | 0.0115 | — | 8.66GB | 10.20GB | 517MB | 5.68GB | <\|channel>thought Here's a thinking process to arrive at the |
