# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-05 17:17
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
| summarization | 128 | 117 | no-quant | 193.0 | 47.1 | 400 | 608ms | 1.5774 | 2.2081 | — | — | 18.16GB | 18.44GB | 46MB | 113MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 1024 | 1019 | no-quant | 473.1 | 46.8 | 400 | 2560ms | 1.2260 | 1.5466 | — | — | 18.16GB | 19.65GB | 61MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 500.8 | 45.6 | 400 | 8643ms | 1.3452 | 1.3227 | — | — | 18.16GB | 23.21GB | 108MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 478.5 | 37.1 | 400 | 68800ms | 1.2541 | 1.9862 | — | — | 18.16GB | 27.12GB | 681MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
