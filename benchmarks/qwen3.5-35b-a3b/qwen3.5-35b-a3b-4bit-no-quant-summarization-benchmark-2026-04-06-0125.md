# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-06 01:25
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
| summarization | 128 | 117 | no-quant | 256.2 | 51.6 | 394 | 497ms | — | — | — | — | 18.16GB | 18.43GB | 42MB | 112MB | The user wants a summary of the text provided in the prompt. |
| summarization | 1024 | 1019 | no-quant | 514.2 | 51.9 | 400 | 2161ms | — | — | — | — | 18.16GB | 19.72GB | 61MB | 310MB | The user wants a summary of the provided text from F. Scott  |
| summarization | 4096 | 4085 | no-quant | 535.0 | 51.2 | 400 | 8101ms | — | — | — | — | 18.16GB | 23.25GB | 98MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 463.2 | 36.0 | 400 | 75556ms | — | — | — | — | 18.16GB | 27.16GB | 544MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
