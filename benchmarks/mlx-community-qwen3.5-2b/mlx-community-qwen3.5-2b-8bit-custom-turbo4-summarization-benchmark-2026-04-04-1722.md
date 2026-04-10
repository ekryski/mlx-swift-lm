# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

**Date**: 2026-04-04 17:22
**Branch**: `ek/consolidated-benchmarks`
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
| summarization | 128 | 109 | turbo4 | 2240.4 | 158.5 | 200 | 49ms | — | 1.7550 | — | — | 1.86GB | 2.07GB | 15MB | 18MB | The text you provided appears to be a **selection of poems b |
| summarization | 1024 | 1011 | turbo4 | 9213.4 | 154.4 | 200 | 110ms | — | 2.1295 | — | — | 1.86GB | 3.25GB | 24MB | 70MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 4096 | 4077 | turbo4 | 10759.3 | 150.9 | 200 | 379ms | — | 2.0664 | — | — | 1.86GB | 3.94GB | 60MB | 249MB | Here is a summary of the provided text from F. Scott Fitzger |
