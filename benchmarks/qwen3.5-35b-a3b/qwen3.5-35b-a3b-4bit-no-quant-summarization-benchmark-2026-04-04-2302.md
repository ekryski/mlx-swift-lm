# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 23:02
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| Max Tokens | 400 |
| Thinking Budget | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 117 | no-quant | 191.9 | 47.3 | 366 | 611ms | 1.5373 | 1.6201 | — | — | 18.16GB | 18.43GB | 32MB | 106MB | The user wants a summary of the text provided.  1.  **Identi |
| summarization | 1024 | 1019 | no-quant | 478.2 | 47.0 | 400 | 2390ms | 1.3191 | 1.7411 | — | — | 18.16GB | 19.65GB | 60MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
