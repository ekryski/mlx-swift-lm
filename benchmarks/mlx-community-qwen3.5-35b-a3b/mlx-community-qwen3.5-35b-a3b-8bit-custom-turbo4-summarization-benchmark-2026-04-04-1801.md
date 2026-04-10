# Inference Benchmark - mlx-community/Qwen3.5-35B-A3B-8bit

**Date**: 2026-04-04 18:01
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: custom
**Model**: `mlx-community/Qwen3.5-35B-A3B-8bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 115GB |
| macOS | 26.3.1 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 107 | turbo4 | 415.7 | 79.4 | 200 | 258ms | — | 1.1114 | — | — | 34.30GB | 34.55GB | 42MB | 18MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1009 | turbo4 | 2997.9 | 78.3 | 200 | 337ms | — | 1.2062 | — | — | 34.30GB | 35.77GB | 57MB | 70MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | turbo4 | 3899.1 | 77.7 | 200 | 1046ms | — | 1.2180 | — | — | 34.30GB | 36.92GB | 115MB | 248MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
