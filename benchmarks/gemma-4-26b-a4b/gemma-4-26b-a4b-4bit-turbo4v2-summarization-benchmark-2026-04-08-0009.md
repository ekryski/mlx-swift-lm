# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-08 00:09
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
| summarization | 1024 | 1012 | turbo4v2 | 612.1 | 31.0 | 200 | 1825ms | — | 1.2639 | — | — | 13.48GB | 16.13GB | 432MB | 54MB | The provided text serves as the opening to F. Scott Fitzgera |
| summarization | 4096 | 4092 | turbo4v2 | 608.7 | 29.7 | 200 | 7020ms | — | 1.1603 | — | — | 13.48GB | 18.41GB | 504MB | 191MB | This text constitutes the opening of F. Scott Fitzgerald’s n |
| summarization | 16384 | 16383 | turbo4v2 | 589.0 | 25.6 | 200 | 28146ms | — | 1.2120 | — | — | 13.48GB | 20.38GB | 992MB | 737MB | This text is the opening of **F. Scott Fitzgerald's** classi |
