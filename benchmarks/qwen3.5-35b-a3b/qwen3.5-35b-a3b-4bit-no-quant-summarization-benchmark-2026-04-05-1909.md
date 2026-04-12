# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-05 19:09
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
| summarization | 128 | 117 | no-quant | 219.7 | 48.9 | 386 | 556ms | 1.4465 | 1.4036 | — | — | 18.16GB | 18.43GB | 29MB | 110MB | The user wants a summary of the provided text snippet.  1.   |
| summarization | 1024 | 1019 | no-quant | 481.8 | 48.7 | 400 | 2377ms | 1.2078 | 1.6890 | — | — | 18.16GB | 19.72GB | 62MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 508.2 | 47.9 | 400 | 8488ms | 1.9626 | 1.7275 | — | — | 18.16GB | 23.25GB | 121MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 413.6 | 34.9 | 400 | 84829ms | 1.3431 | 1.5299 | — | — | 18.16GB | 27.16GB | 646MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
