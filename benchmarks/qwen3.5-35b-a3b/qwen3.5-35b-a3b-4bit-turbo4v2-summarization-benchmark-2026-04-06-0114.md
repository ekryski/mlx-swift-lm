# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-06 01:14
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
| summarization | 128 | 117 | turbo4v2 | 223.1 | 51.9 | 364 | 526ms | 1.5888 | 1.6630 | — | — | 18.16GB | 18.43GB | 37MB | 21MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | turbo4v2 | 472.7 | 51.1 | 400 | 2441ms | 1.5069 | 1.5901 | — | — | 18.16GB | 19.72GB | 62MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 507.1 | 50.0 | 400 | 8586ms | 1.4087 | 1.2681 | — | — | 18.16GB | 23.25GB | 122MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 449.1 | 37.8 | 400 | 76095ms | 1.2980 | 1.3489 | — | — | 18.16GB | 27.16GB | 543MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
