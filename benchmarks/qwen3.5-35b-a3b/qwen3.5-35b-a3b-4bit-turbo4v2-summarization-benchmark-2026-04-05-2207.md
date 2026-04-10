# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-05 22:07
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
| summarization | 128 | 117 | turbo4v2 | 230.3 | 52.0 | 393 | 510ms | 1.8355 | 1.4132 | — | — | 18.16GB | 18.43GB | 42MB | 23MB | The user wants a summary of the text provided. The text is n |
| summarization | 1024 | 1019 | turbo4v2 | 475.1 | 52.0 | 400 | 2438ms | 1.3579 | 1.0276 | — | — | 18.16GB | 19.72GB | 61MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 427.1 | 36.7 | 400 | 84212ms | 1.1765 | 1.5935 | — | — | 18.16GB | 27.16GB | 679MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
