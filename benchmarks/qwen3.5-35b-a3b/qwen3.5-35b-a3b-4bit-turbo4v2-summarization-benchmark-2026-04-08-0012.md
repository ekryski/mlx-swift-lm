# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-08 00:12
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
| summarization | 1024 | 1009 | turbo4v2 | 541.6 | 65.0 | 200 | 2107ms | — | 1.2638 | — | — | 18.16GB | 19.70GB | 57MB | 54MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | turbo4v2 | 540.5 | 63.7 | 200 | 8030ms | — | 1.2427 | — | — | 18.16GB | 23.24GB | 117MB | 190MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16351 | turbo4v2 | 536.2 | 54.6 | 200 | 30978ms | — | 1.2836 | — | — | 18.16GB | 24.95GB | 320MB | 735MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
