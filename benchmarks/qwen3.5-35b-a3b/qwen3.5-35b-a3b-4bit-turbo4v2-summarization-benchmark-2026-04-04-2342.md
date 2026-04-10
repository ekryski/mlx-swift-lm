# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-04 23:42
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
| summarization | 128 | 117 | turbo4v2 | 210.8 | 48.0 | 400 | 557ms | 1.6652 | 1.6274 | 0.0537 | 0.0930 | 18.16GB | 18.43GB | 38MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 55.6 | 47.4 | 400 | 18789ms | 1.2920 | 1.5929 | 0.0553 | 0.1303 | 18.16GB | 19.65GB | 59MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 220.4 | 45.7 | 400 | 18960ms | 1.2480 | 1.7864 | 0.0308 | 0.1239 | 18.16GB | 23.21GB | 109MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 416.3 | 37.2 | 400 | 79101ms | 1.5579 | 1.5284 | 0.0786 | 0.1055 | 18.16GB | 27.12GB | 682MB | 1.44GB | The user wants a summary of the provided text, which is the  |
