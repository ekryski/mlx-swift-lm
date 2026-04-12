# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

- **Date**: 2026-04-04 19:42
- **Branch**: `feature/turboquant-plus-optimizations`
- **Quantization**: custom
- **Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 109 | turbo3 | 2270.7 | 157.3 | 200 | 49ms | — | 1.9194 | — | — | 1.86GB | 2.07GB | 16MB | 14MB | The text you provided appears to be a **title page and a ded |
| summarization | 1024 | 1011 | turbo3 | 9064.0 | 155.3 | 200 | 112ms | — | 1.7901 | — | — | 1.86GB | 3.25GB | 25MB | 54MB | Based on the text provided, here is a summary of the work:   |
| summarization | 4096 | 4077 | turbo3 | 10706.0 | 148.3 | 200 | 381ms | — | 1.8132 | — | — | 1.86GB | 3.94GB | 61MB | 190MB | Here is a summary of the provided text from *The Great Gatsb |
