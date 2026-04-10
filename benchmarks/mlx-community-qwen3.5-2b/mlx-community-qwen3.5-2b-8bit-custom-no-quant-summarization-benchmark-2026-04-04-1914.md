# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

- **Date**: 2026-04-04 19:14
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
| summarization | 128 | 109 | no-quant | 2059.6 | 159.3 | 200 | 54ms | — | 2.2525 | — | — | 1.86GB | 2.07GB | 16MB | 68MB | The text you provided is a fragment from **F. Scott Fitzgera |
| summarization | 1024 | 1011 | no-quant | 9008.3 | 157.2 | 200 | 113ms | — | 1.9568 | — | — | 1.86GB | 3.25GB | 25MB | 265MB | Here is a summary of the provided text from F. Scott Fitzger |
| summarization | 4096 | 4077 | no-quant | 10855.6 | 153.7 | 200 | 376ms | — | 2.2378 | — | — | 1.86GB | 3.94GB | 61MB | 936MB | Here is a summary of the provided text from F. Scott Fitzger |
