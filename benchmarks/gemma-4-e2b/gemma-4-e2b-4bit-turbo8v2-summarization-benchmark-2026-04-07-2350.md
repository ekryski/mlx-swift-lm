# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-07 23:50
- **Branch**: `ek/tom-eric-moe-tuning`
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
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 1024 | 1008 | turbo8v2 | 969.7 | 81.2 | 200 | 1041ms | — | 1.5117 | — | — | 2.45GB | 4.76GB | 21MB | 87MB | This excerpt appears to be from **The Great Gatsby** by F. S |
| summarization | 4096 | 4088 | turbo8v2 | 1619.6 | 78.4 | 200 | 2611ms | — | 1.5965 | — | — | 2.45GB | 5.53GB | 76MB | 308MB | This text is an excerpt from **The Great Gatsby** by F. Scot |
| summarization | 16384 | 16384 | turbo8v2 | 2572.8 | 70.2 | 200 | 6432ms | — | 1.5618 | — | — | 2.45GB | 6.45GB | 233MB | 1.16GB | This text is an excerpt from **The Great Gatsby by F. Scott  |
