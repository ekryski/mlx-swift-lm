# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-05 19:14
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 117 | turbo4v2 | 225.9 | 50.8 | 400 | 520ms | 1.7636 | 1.5211 | — | — | 18.16GB | 18.43GB | 45MB | 23MB | Here's a thinking process that leads to the suggested summar |
| summarization | 1024 | 1019 | turbo4v2 | 489.0 | 49.9 | 400 | 2325ms | 1.3807 | 2.1291 | — | — | 18.16GB | 19.72GB | 57MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 511.7 | 48.6 | 400 | 8436ms | 1.3434 | 1.6086 | — | — | 18.16GB | 23.25GB | 120MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 482.0 | 37.6 | 400 | 68464ms | 1.4765 | 1.9122 | — | — | 18.16GB | 27.16GB | 680MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
