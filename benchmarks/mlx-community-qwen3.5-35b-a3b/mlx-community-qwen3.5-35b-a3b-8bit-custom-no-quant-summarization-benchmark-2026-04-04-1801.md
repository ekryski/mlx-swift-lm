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
| summarization | 128 | 107 | no-quant | 151.9 | 78.7 | 200 | 705ms | — | 1.1399 | — | — | 34.30GB | 34.57GB | 40MB | 67MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1009 | no-quant | 1494.1 | 77.9 | 200 | 676ms | — | 1.2323 | — | — | 34.30GB | 35.77GB | 56MB | 264MB | Thinking Process:  1.  **Analyze the Request:**     *   **In |
| summarization | 4096 | 4075 | no-quant | 3613.7 | 76.1 | 200 | 1128ms | — | 1.1631 | — | — | 34.30GB | 36.92GB | 117MB | 935MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
