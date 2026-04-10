# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 22:58
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
| summarization | 1024 | 1012 | turbo4v2 | 599.0 | 31.0 | 200 | 1842ms | — | 1.2498 | — | — | 13.48GB | 16.13GB | 428MB | 54MB | This text comprises the opening of F. Scott Fitzgerald’s nov |
| summarization | 4096 | 4092 | turbo4v2 | 613.1 | 29.7 | 200 | 6998ms | — | 1.3042 | — | — | 13.48GB | 18.41GB | 528MB | 191MB | This text comprises the title page, table of contents, epigr |
| summarization | 16384 | 16383 | turbo4v2 | 588.5 | 25.4 | 200 | 28182ms | — | 1.7634 | — | — | 13.48GB | 20.38GB | 984MB | 737MB | to late-night arguments.  "You're a lucky man, Nick," Jordan |
