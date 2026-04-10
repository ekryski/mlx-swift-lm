# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-05 19:26
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| summarization | 128 | 117 | turbo4v2 | 238.4 | 49.0 | 392 | 493ms | 1.5702 | 1.5763 | — | — | 18.16GB | 18.44GB | 41MB | 23MB | The user wants a summary of the provided text snippet.  **1. |
| summarization | 1024 | 1019 | turbo4v2 | 483.6 | 48.3 | 400 | 2373ms | 1.3421 | 1.6833 | — | — | 18.16GB | 19.72GB | 60MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 502.3 | 47.1 | 400 | 8571ms | 1.3369 | 1.2201 | — | — | 18.16GB | 23.25GB | 95MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 444.1 | 34.6 | 400 | 78281ms | 1.6356 | 1.8918 | — | — | 18.16GB | 27.16GB | 679MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
