# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 23:38
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
| summarization | 1024 | 1012 | turbo8v4 | 544.2 | 30.7 | 200 | 2039ms | — | 1.2409 | — | — | 13.48GB | 16.13GB | 304MB | 104MB | The provided text contains the opening of F. Scott Fitzgeral |
| summarization | 4096 | 4092 | turbo8v4 | 611.9 | 29.7 | 200 | 6987ms | — | 1.2535 | — | — | 13.48GB | 18.41GB | 488MB | 367MB | The provided text is the opening of F. Scott Fitzgerald’s *T |
| summarization | 16384 | 16383 | turbo8v4 | 588.4 | 25.5 | 200 | 28122ms | — | 1.3491 | — | — | 13.48GB | 20.38GB | 824MB | 1.38GB | This provided text contains the opening chapters of F. Scott |
