# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

**Date**: 2026-04-04 19:42
**Branch**: `feature/turboquant-plus-optimizations`
**Quantization**: custom
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 109 | turbo4 | 2240.8 | 147.8 | 200 | 49ms | — | 2.4075 | — | — | 1.86GB | 2.07GB | 14MB | 18MB | The text you provided appears to be a **fragment** from the  |
| summarization | 1024 | 1011 | turbo4 | 8734.4 | 132.7 | 200 | 116ms | — | 2.1758 | — | — | 1.86GB | 3.25GB | 22MB | 70MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 4096 | 4077 | turbo4 | 10623.9 | 147.5 | 200 | 384ms | — | 2.3029 | — | — | 1.86GB | 3.94GB | 10MB | 249MB | Here is a summary of the provided text from *The Great Gatsb |
