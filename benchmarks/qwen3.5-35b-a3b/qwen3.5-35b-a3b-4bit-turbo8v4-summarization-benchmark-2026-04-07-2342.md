# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-07 23:42
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
| Max Tokens | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 1024 | 1009 | turbo8v4 | 526.2 | 65.5 | 200 | 2153ms | — | 1.1917 | — | — | 18.16GB | 19.70GB | 55MB | 103MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | turbo8v4 | 540.2 | 63.0 | 200 | 8028ms | — | 1.4510 | — | — | 18.16GB | 23.24GB | 117MB | 365MB | Here's a thinking process that leads to the suggested summar |
| summarization | 16384 | 16351 | turbo8v4 | 536.2 | 55.8 | 200 | 31017ms | — | 1.3148 | — | — | 18.16GB | 24.95GB | 355MB | 1.38GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
