# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-05 11:30
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
| summarization | 128 | 117 | turbo4v2 | 203.7 | 47.3 | 336 | 576ms | 1.4593 | 1.4704 | — | — | 18.16GB | 18.43GB | 38MB | 20MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | turbo4v2 | 473.0 | 47.3 | 400 | 2478ms | 1.5143 | 1.4220 | — | — | 18.16GB | 19.65GB | 61MB | 63MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | turbo4v2 | 499.5 | 45.9 | 400 | 8722ms | 1.4906 | 1.5315 | — | — | 18.16GB | 23.21GB | 122MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 479.7 | 37.5 | 400 | 68776ms | 1.4530 | 1.5452 | — | — | 18.16GB | 27.12GB | 646MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
