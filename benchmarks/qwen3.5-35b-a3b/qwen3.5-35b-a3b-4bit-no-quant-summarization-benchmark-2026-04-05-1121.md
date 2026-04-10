# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-05 11:21
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
| summarization | 128 | 117 | no-quant | 155.5 | 47.5 | 400 | 754ms | 1.3605 | 1.4885 | — | — | 18.16GB | 18.44GB | 42MB | 113MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | no-quant | 472.7 | 47.4 | 400 | 2462ms | 1.3451 | 1.4010 | — | — | 18.16GB | 19.65GB | 61MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 502.4 | 45.7 | 400 | 8642ms | 1.7073 | 1.7465 | — | — | 18.16GB | 23.21GB | 121MB | 981MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 399.4 | 35.1 | 400 | 89884ms | 1.3819 | 1.5373 | — | — | 18.16GB | 27.12GB | 680MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
