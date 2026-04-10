# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 21:27
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
| summarization | 1024 | 1012 | turbo4v2 | 536.7 | 30.8 | 200 | 2050ms | — | 1.3305 | — | — | 13.48GB | 16.13GB | 400MB | 54MB | This text contains the opening of F. Scott Fitzgerald’s nove |
| summarization | 4096 | 4092 | turbo4v2 | 607.7 | 29.8 | 200 | 7005ms | — | 1.1633 | — | — | 13.48GB | 18.41GB | 544MB | 191MB | The provided text is the opening chapter of F. Scott Fitzger |
