# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-08 21:01
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
| summarization | 1024 | 1008 | no-quant | 1064.9 | 0.0 | 0 | 951ms | — | 2.2630 | — | — | 2.45GB | 4.77GB | 23MB | 220MB |  |
| summarization | 4096 | 4088 | no-quant | 1628.6 | 73.7 | 200 | 2580ms | — | nan | — | — | 2.45GB | 5.57GB | 73MB | 938MB | <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad> |
| summarization | 16384 | 16384 | no-quant | 2547.3 | 68.1 | 200 | 6500ms | — | nan | — | — | 2.45GB | 6.50GB | 268MB | 3.54GB | <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad> |
