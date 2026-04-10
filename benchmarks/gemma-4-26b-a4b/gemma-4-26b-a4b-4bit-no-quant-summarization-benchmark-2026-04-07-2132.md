# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 21:32
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
| summarization | 1024 | 1012 | no-quant | 585.3 | 31.1 | 200 | 1840ms | — | 1.1890 | — | — | 13.48GB | 16.13GB | 432MB | 265MB | The provided text serves as the opening of F. Scott Fitzgera |
| summarization | 4096 | 4092 | no-quant | 610.2 | 29.8 | 200 | 6974ms | — | 1.2200 | — | — | 13.48GB | 18.41GB | 512MB | 939MB | This text is the opening chapter of F. Scott Fitzgerald’s no |
