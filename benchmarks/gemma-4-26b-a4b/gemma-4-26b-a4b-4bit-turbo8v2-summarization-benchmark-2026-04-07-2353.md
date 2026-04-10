# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-07 23:53
- **Branch**: `ek/tom-eric-moe-tuning`
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
| summarization | 1024 | 1012 | turbo8v2 | 544.2 | 30.7 | 200 | 2041ms | — | 1.2121 | — | — | 13.48GB | 16.13GB | 336MB | 87MB | The provided text is the opening of F. Scott Fitzgerald's *T |
| summarization | 4096 | 4092 | turbo8v2 | 610.0 | 29.7 | 200 | 7002ms | — | 1.2787 | — | — | 13.48GB | 18.41GB | 536MB | 308MB | The provided text is the opening of F. Scott Fitzgerald’s ** |
| summarization | 16384 | 16383 | turbo8v2 | 588.9 | 25.5 | 200 | 28146ms | — | 2.0120 | — | — | 13.48GB | 20.38GB | 1008MB | 1.16GB | ...to extravagant energetic outbursts.  “He’s a very good ho |
