# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

- **Date**: 2026-04-04 19:29
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
| summarization | 128 | 109 | turbo4 | 2409.4 | 150.2 | 200 | 46ms | — | 2.3057 | — | — | 1.86GB | 2.07GB | 16MB | 18MB | The text you provided is **not a summary** of *The Great Gat |
| summarization | 1024 | 1011 | turbo4 | 8567.1 | 144.9 | 200 | 118ms | — | 1.7674 | — | — | 1.86GB | 3.25GB | 25MB | 70MB | Based on the text provided, here is a summary of the key con |
| summarization | 4096 | 4077 | turbo4 | 10081.3 | 144.3 | 200 | 405ms | — | 1.7715 | — | — | 1.86GB | 3.94GB | 61MB | 249MB | Here is a summary of the provided text from *The Great Gatsb |
