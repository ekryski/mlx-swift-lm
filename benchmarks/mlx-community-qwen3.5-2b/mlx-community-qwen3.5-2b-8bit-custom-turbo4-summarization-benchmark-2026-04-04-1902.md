# Inference Benchmark - mlx-community/Qwen3.5-2B-8bit

**Date**: 2026-04-04 19:02
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
| summarization | 128 | 109 | turbo4 | 2204.8 | 161.7 | 200 | 50ms | — | 2.6658 | — | — | 1.86GB | 2.07GB | 15MB | 18MB | The text you provided contains a brief excerpt from the endi |
| summarization | 1024 | 1011 | turbo4 | 9179.3 | 158.0 | 200 | 110ms | — | 1.8456 | — | — | 1.86GB | 3.25GB | 22MB | 70MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 4096 | 4077 | turbo4 | 10763.9 | 155.6 | 200 | 379ms | — | 1.9503 | — | — | 1.86GB | 3.94GB | 60MB | 249MB | This text is the beginning of F. Scott Fitzgerald's novel ** |
