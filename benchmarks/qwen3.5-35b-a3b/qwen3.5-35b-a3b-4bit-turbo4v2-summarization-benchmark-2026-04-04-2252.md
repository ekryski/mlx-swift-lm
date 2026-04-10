# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-04 22:52
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
| Max Tokens | 400 |
| Thinking Budget | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 117 | turbo4v2 | 187.5 | 47.0 | 400 | 626ms | 1.3790 | 1.9854 | 0.0877 | 0.1060 | 18.16GB | 18.43GB | 46MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 51.3 | 45.0 | 400 | 20324ms | 1.2576 | 1.6990 | 0.0403 | 0.1333 | 18.16GB | 19.65GB | 61MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 205.2 | 42.5 | 400 | 20400ms | 1.2343 | 1.4526 | 0.0533 | 0.0718 | 18.16GB | 20.80GB | 122MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 412.7 | 37.0 | 400 | 79770ms | 1.2627 | 1.4365 | 0.0536 | 0.0659 | 18.16GB | 22.98GB | 614MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
