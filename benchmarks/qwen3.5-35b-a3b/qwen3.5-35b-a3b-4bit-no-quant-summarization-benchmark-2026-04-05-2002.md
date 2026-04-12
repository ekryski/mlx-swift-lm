# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-05 20:02
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
| summarization | 128 | 117 | no-quant | 238.6 | 51.6 | 400 | 493ms | 1.7017 | 1.8884 | — | — | 18.16GB | 18.43GB | 42MB | 113MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | no-quant | 469.1 | 51.7 | 400 | 2489ms | 1.3019 | 1.0351 | — | — | 18.16GB | 19.72GB | 60MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 496.4 | 49.5 | 400 | 8688ms | 1.4108 | 1.5666 | — | — | 18.16GB | 23.25GB | 121MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 479.5 | 39.9 | 399 | 68813ms | 1.3226 | 1.8608 | — | — | 18.16GB | 27.16GB | 680MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
