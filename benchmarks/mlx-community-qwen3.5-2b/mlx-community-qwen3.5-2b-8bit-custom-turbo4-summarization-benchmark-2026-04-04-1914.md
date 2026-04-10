# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

**Date**: 2026-04-04 19:14
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
| summarization | 128 | 109 | turbo4 | 2455.7 | 158.6 | 200 | 45ms | — | 2.2615 | — | — | 1.86GB | 2.07GB | 15MB | 18MB | The text you provided is a fragment from **F. Scott Fitzgera |
| summarization | 1024 | 1011 | turbo4 | 9083.3 | 153.1 | 200 | 112ms | — | 2.2912 | — | — | 1.86GB | 3.25GB | 25MB | 70MB | Here is a summary of the provided text, which is the Table o |
| summarization | 4096 | 4077 | turbo4 | 10772.2 | 148.1 | 200 | 379ms | — | 1.6893 | — | — | 1.86GB | 3.94GB | 61MB | 249MB | This text is the opening chapter of **The Great Gatsby** by  |
