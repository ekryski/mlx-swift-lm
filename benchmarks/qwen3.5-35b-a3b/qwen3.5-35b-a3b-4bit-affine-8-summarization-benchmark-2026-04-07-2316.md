# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-07 23:16
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 1024 | 1009 | affine-8 | 467.7 | 57.2 | 200 | 2426ms | — | 1.3252 | — | — | 18.16GB | 19.38GB | 42MB | 149MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | affine-8 | 536.6 | 56.3 | 200 | 8094ms | — | 1.3887 | — | — | 18.16GB | 23.21GB | 72MB | 526MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16351 | affine-8 | 531.1 | 46.6 | 200 | 31251ms | — | 1.2327 | — | — | 18.16GB | 24.82GB | 203MB | 1.99GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
