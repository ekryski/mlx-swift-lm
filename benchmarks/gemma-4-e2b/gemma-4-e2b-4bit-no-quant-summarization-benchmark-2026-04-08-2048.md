# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-08 20:48
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e2b-it-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M1 Max (applegpu_g13s) |
| System RAM | 64GB |
| GPU Memory Limit | 48GB |
| macOS | 15.7.4 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 1024 | 1008 | no-quant | 766.4 | 80.8 | 200 | 1323ms | — | 1.4387 | — | — | 2.45GB | 4.77GB | 26MB | 264MB | The provided text is an excerpt from *The Great Gatsby* by F |
| summarization | 4096 | 4088 | no-quant | 1368.8 | 79.5 | 200 | 3074ms | — | 1.6821 | — | — | 2.45GB | 5.57GB | 75MB | 938MB | This excerpt is a collection of interconnected pieces from * |
| summarization | 16384 | 16384 | no-quant | 2382.6 | 66.2 | 200 | 6910ms | — | 1.5676 | — | — | 2.45GB | 6.50GB | 268MB | 3.54GB | This text appears to be an excerpt from **The Great Gatsby** |
