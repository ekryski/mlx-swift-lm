# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 23:13
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
| summarization | 1024 | 1012 | affine-8 | 597.6 | 31.0 | 200 | 1827ms | — | 1.1536 | — | — | 13.48GB | 16.13GB | 352MB | 149MB | The provided text is the opening of F. Scott Fitzgerald’s *T |
| summarization | 4096 | 4092 | affine-8 | 614.5 | 29.7 | 200 | 6967ms | — | 1.1874 | — | — | 13.48GB | 18.41GB | 480MB | 528MB | The provided text is the opening of F. Scott Fitzgerald’s ** |
| summarization | 16384 | 16383 | affine-8 | 586.9 | 25.6 | 200 | 28210ms | — | 1.2632 | — | — | 13.48GB | 20.38GB | 1000MB | 1.99GB | This text is the opening of **F. Scott Fitzgerald's** master |
