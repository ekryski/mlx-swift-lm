# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-07 23:09
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| summarization | 1024 | 1008 | affine-8 | 1046.1 | 83.7 | 200 | 965ms | — | 1.5458 | — | — | 2.45GB | 4.76GB | 27MB | 149MB | This excerpt comes from **The Great Gatsby** by F. Scott Fit |
| summarization | 4096 | 4088 | affine-8 | 1661.1 | 80.4 | 200 | 2555ms | — | 1.5676 | — | — | 2.45GB | 5.53GB | 73MB | 528MB | This excerpt appears to be from **The Great Gatsby** by F. S |
| summarization | 16384 | 16384 | affine-8 | 2561.7 | 68.6 | 200 | 6451ms | — | 1.4217 | — | — | 2.45GB | 6.45GB | 268MB | 1.99GB | This text appears to be an excerpt from **Nick Carraway's pe |
