# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-06 01:20
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
| summarization | 128 | 117 | turbo4v2 | 223.0 | 51.8 | 400 | 526ms | 1.3934 | 1.5734 | — | — | 18.16GB | 18.44GB | 45MB | 23MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | turbo4v2 | 483.8 | 51.7 | 400 | 2396ms | 1.4224 | 1.9660 | — | — | 18.16GB | 19.72GB | 56MB | 63MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | turbo4v2 | 507.3 | 50.0 | 400 | 8592ms | 1.3553 | 1.4887 | — | — | 18.16GB | 23.25GB | 120MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 483.3 | 40.8 | 400 | 68420ms | 1.4852 | 1.6760 | — | — | 18.16GB | 27.16GB | 546MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
