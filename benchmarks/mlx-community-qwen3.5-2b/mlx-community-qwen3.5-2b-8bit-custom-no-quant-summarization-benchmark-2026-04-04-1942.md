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
| summarization | 128 | 109 | no-quant | 2114.5 | 158.8 | 200 | 52ms | — | 1.9741 | — | — | 1.86GB | 2.07GB | 16MB | 68MB | The text you provided is a **poem** titled "Once again to Ze |
| summarization | 1024 | 1011 | no-quant | 9020.9 | 155.7 | 200 | 112ms | — | 1.9671 | — | — | 1.86GB | 3.25GB | 25MB | 265MB | Here is a summary of the provided text, which appears to be  |
| summarization | 4096 | 4077 | no-quant | 10526.1 | 153.9 | 200 | 388ms | — | 1.9278 | — | — | 1.86GB | 3.94GB | 60MB | 936MB | This text is the opening chapter of F. Scott Fitzgerald's *T |
