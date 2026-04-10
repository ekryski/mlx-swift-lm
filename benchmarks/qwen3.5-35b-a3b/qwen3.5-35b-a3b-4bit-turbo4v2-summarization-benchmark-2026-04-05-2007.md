# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-05 20:07
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
| summarization | 128 | 117 | turbo4v2 | 243.2 | 51.9 | 364 | 483ms | 1.2426 | 1.4111 | — | — | 18.16GB | 18.43GB | 42MB | 21MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | turbo4v2 | 473.8 | 51.4 | 400 | 2454ms | 1.5979 | 1.7121 | — | — | 18.16GB | 19.72GB | 62MB | 63MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | turbo4v2 | 502.1 | 49.7 | 400 | 8604ms | 1.3280 | 1.7631 | — | — | 18.16GB | 23.25GB | 121MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 486.7 | 40.3 | 400 | 67762ms | 1.4471 | 1.5452 | — | — | 18.16GB | 27.16GB | 682MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
