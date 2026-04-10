# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 21:05
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
| summarization | 1024 | 1012 | no-quant | 540.0 | 31.2 | 200 | 2013ms | — | 1.2177 | — | — | 13.48GB | 16.13GB | 296MB | 265MB | This text contains the opening pages of F. Scott Fitzgerald’ |
| summarization | 4096 | 4092 | no-quant | 532.9 | 29.4 | 200 | 8026ms | — | 1.2302 | — | — | 13.48GB | 21.12GB | 520MB | 939MB | The provided text is the opening of F. Scott Fitzgerald’s no |
