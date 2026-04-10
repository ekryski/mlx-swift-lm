# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

**Date**: 2026-04-04 19:29
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
| summarization | 128 | 109 | no-quant | 2212.7 | 156.6 | 200 | 50ms | — | 1.8405 | — | — | 1.86GB | 2.07GB | 16MB | 68MB | The text provided contains a title page for the novel *The G |
| summarization | 1024 | 1011 | no-quant | 8985.3 | 154.8 | 200 | 113ms | — | 1.9635 | — | — | 1.86GB | 3.25GB | 25MB | 265MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 4096 | 4077 | no-quant | 10656.3 | 152.5 | 200 | 383ms | — | 2.1110 | — | — | 1.86GB | 3.94GB | 60MB | 936MB | This text is a summary of the opening chapters of **The Grea |
