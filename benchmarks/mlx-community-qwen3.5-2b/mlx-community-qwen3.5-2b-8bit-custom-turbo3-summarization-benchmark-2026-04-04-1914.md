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
| summarization | 128 | 109 | turbo3 | 2282.2 | 158.3 | 200 | 49ms | — | 2.5582 | — | — | 1.86GB | 2.07GB | 16MB | 14MB | The text you provided is a famous, though largely untitled,  |
| summarization | 1024 | 1011 | turbo3 | 9090.6 | 157.5 | 200 | 111ms | — | 2.0100 | — | — | 1.86GB | 3.25GB | 24MB | 54MB | Here is a summary of the provided text, which is the **Intro |
| summarization | 4096 | 4077 | turbo3 | 10634.0 | 153.6 | 200 | 384ms | — | 2.0770 | — | — | 1.86GB | 3.94GB | 60MB | 190MB | Here is a summary of the provided text from *The Great Gatsb |
