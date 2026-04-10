# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-05 19:55
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
| summarization | 128 | 117 | turbo4v2 | 234.1 | 52.3 | 400 | 502ms | 1.6014 | 1.4507 | — | — | 18.16GB | 18.44GB | 40MB | 23MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | turbo4v2 | 420.7 | 51.9 | 400 | 2785ms | 1.3281 | 1.7615 | — | — | 18.16GB | 19.72GB | 62MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 456.7 | 50.4 | 400 | 9380ms | 1.3712 | 1.6875 | — | — | 18.16GB | 23.25GB | 120MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 437.4 | 40.0 | 400 | 75336ms | 1.3142 | 1.4483 | — | — | 18.16GB | 27.16GB | 679MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
