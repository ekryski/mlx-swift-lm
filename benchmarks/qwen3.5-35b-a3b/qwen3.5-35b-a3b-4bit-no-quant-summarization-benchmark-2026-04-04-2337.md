# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 23:37
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
| summarization | 128 | 117 | no-quant | 184.6 | 44.7 | 373 | 636ms | 1.4693 | 1.5988 | — | — | 18.16GB | 18.43GB | 42MB | 107MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | no-quant | 453.8 | 44.1 | 400 | 2584ms | 1.3365 | 1.6621 | — | — | 18.16GB | 19.67GB | 60MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 509.6 | 44.4 | 400 | 8512ms | 1.2759 | 1.6560 | — | — | 18.16GB | 23.21GB | 109MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 490.7 | 38.2 | 400 | 67245ms | 1.5197 | 1.6285 | — | — | 18.16GB | 27.12GB | 612MB | 7.07GB | The user wants a summary of the provided text, which is the  |
