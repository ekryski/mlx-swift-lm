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
| summarization | 128 | 109 | no-quant | 2206.1 | 154.2 | 147 | 50ms | — | 2.0951 | — | — | 1.86GB | 2.07GB | 14MB | 56MB | The text provided is the **Table of Contents** for the novel |
| summarization | 1024 | 1011 | no-quant | 9228.1 | 151.1 | 200 | 110ms | — | 1.6736 | — | — | 1.86GB | 3.25GB | 24MB | 265MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 4096 | 4077 | no-quant | 8789.3 | 153.2 | 200 | 464ms | — | 1.8864 | — | — | 1.86GB | 3.94GB | 60MB | 936MB | Here is a summary of the provided text from *The Great Gatsb |
