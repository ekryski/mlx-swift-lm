# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

- **Date**: 2026-04-04 20:04
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
| summarization | 128 | 109 | turbo3 | 2243.2 | 158.0 | 200 | 49ms | — | 2.1376 | — | — | 1.86GB | 2.07GB | 15MB | 14MB | The text you provided is a fragment from the famous 1925 nov |
| summarization | 1024 | 1011 | turbo3 | 9057.7 | 157.0 | 200 | 112ms | — | 1.8500 | — | — | 1.86GB | 3.25GB | 24MB | 54MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 4096 | 4077 | turbo3 | 10482.9 | 153.9 | 200 | 389ms | — | 1.7254 | — | — | 1.86GB | 3.94GB | 60MB | 190MB | This text is a **summary and analysis** of the opening secti |
