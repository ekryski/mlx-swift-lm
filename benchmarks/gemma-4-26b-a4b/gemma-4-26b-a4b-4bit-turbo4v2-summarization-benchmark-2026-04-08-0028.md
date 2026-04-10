# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-08 00:28
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`

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
| summarization | 1024 | 1012 | turbo4v2 | 589.3 | 31.1 | 200 | 1825ms | — | 1.2819 | — | — | 13.48GB | 16.13GB | 384MB | 54MB | The provided text constitutes the opening of F. Scott Fitzge |
| summarization | 4096 | 4092 | turbo4v2 | 612.5 | 30.0 | 200 | 6978ms | — | 1.2027 | — | — | 13.48GB | 18.41GB | 256MB | 191MB | This text contains the opening chapter of F. Scott Fitzgeral |
| summarization | 16384 | 16383 | turbo4v2 | 588.1 | 25.5 | 200 | 28181ms | — | 1.2672 | — | — | 13.48GB | 20.38GB | 576MB | 737MB | The text provided is the opening of **F. Scott Fitzgerald's* |
