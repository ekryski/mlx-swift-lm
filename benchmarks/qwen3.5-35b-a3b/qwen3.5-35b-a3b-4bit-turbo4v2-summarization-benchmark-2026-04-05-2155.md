# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-05 21:55
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
| summarization | 128 | 117 | turbo4v2 | 233.1 | 52.2 | 400 | 504ms | 1.5565 | 2.0110 | — | — | 18.16GB | 18.44GB | 41MB | 23MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | turbo4v2 | 476.5 | 51.6 | 400 | 2407ms | 2.0160 | 1.1666 | — | — | 18.16GB | 19.72GB | 62MB | 63MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | turbo4v2 | 507.8 | 49.5 | 400 | 8600ms | 1.6242 | 1.8431 | — | — | 18.16GB | 23.25GB | 95MB | 199MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | turbo4v2 | 413.7 | 37.9 | 400 | 91470ms | 1.3411 | 1.7358 | — | — | 18.16GB | 27.16GB | 682MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
