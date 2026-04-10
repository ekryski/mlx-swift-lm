# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-08 20:32
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| summarization | 1024 | 1008 | no-quant | 1155.7 | 82.8 | 200 | 873ms | — | — | — | — | 2.45GB | 4.77GB | 25MB | 264MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | no-quant | 1639.2 | 80.2 | 200 | 2511ms | — | — | — | — | 2.45GB | 5.57GB | 55MB | 938MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 16384 | 16384 | no-quant | 2576.9 | 72.8 | 200 | 6431ms | — | — | — | — | 2.45GB | 6.50GB | 0MB | 3.54GB | This text appears to be an excerpt from **Nick Carraway's pe |
