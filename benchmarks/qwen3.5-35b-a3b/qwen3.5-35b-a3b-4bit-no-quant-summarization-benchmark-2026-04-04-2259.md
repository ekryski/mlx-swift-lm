# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 22:59
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
| summarization | 128 | 117 | no-quant | 186.1 | 47.2 | 400 | 631ms | 1.4355 | 1.9063 | — | — | 18.16GB | 18.44GB | 43MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 471.9 | 46.6 | 400 | 2469ms | 1.6991 | 1.4464 | — | — | 18.16GB | 19.65GB | 55MB | 310MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | no-quant | 497.7 | 44.5 | 400 | 8731ms | 1.3119 | 1.0644 | — | — | 18.16GB | 23.21GB | 122MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 483.9 | 37.9 | 400 | 68164ms | 1.3773 | 1.0251 | — | — | 18.16GB | 27.12GB | 680MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
