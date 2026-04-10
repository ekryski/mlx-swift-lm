# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-08 13:50
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
| Turbo Recompress Interval | 256 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 32768 | 32768 | turbo4v2 | 2512.4 | 61.2 | 200 | 13072ms | — | 1.7976 | — | — | 2.45GB | 7.67GB | 389MB | 1.43GB | This is a dense and complex excerpt from a work that appears |
| summarization | 65536 | 65536 | turbo4v2 | 1709.0 | 30.0 | 200 | 38397ms | — | 2.2700 | — | — | 2.45GB | 10.11GB | 1.01GB | 2.85GB | This is a fascinating collection of prose, clearly acting as |
| summarization | 131072 | 130557 | turbo4v2 | 1070.3 | 37.2 | 200 | 122120ms | — | 1.8633 | — | — | 2.45GB | 13.69GB | 1.26GB | 5.67GB | This is a summary of the provided excerpts from *The Great G |
