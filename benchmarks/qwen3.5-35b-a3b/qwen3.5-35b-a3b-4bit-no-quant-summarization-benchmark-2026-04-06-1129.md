# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-06 11:29
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
| summarization | 128 | 117 | no-quant | 74.9 | 52.3 | 400 | 1915ms | — | — | — | — | 18.16GB | 18.43GB | 42MB | 113MB | Here's a thinking process that leads to the suggested summar |
| summarization | 256 | 249 | no-quant | 335.8 | 52.1 | 400 | 743ms | — | — | — | — | 18.16GB | 18.69GB | 47MB | 142MB | The user wants a summary of the provided text, which is the  |
| summarization | 512 | 504 | no-quant | 418.5 | 51.8 | 400 | 1206ms | — | — | — | — | 18.16GB | 19.02GB | 41MB | 198MB | The user wants a summary of the provided text from *The Grea |
| summarization | 1024 | 1019 | no-quant | 550.7 | 51.6 | 400 | 2138ms | — | — | — | — | 18.16GB | 19.72GB | 56MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 594.7 | 51.4 | 400 | 3941ms | — | — | — | — | 18.16GB | 20.74GB | 64MB | 534MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 537.7 | 50.4 | 400 | 8234ms | — | — | — | — | 18.16GB | 23.25GB | 109MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 542.9 | 47.9 | 399 | 15672ms | — | — | — | — | 18.16GB | 23.84GB | 202MB | 1.83GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 525.7 | 45.5 | 400 | 31625ms | — | — | — | — | 18.16GB | 24.96GB | 253MB | 3.58GB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 444.8 | 38.9 | 350 | 77636ms | — | — | — | — | 18.16GB | 27.16GB | 682MB | 7.06GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | no-quant | 368.8 | 33.4 | 400 | 178075ms | — | — | — | — | 18.16GB | 31.72GB | 1.29GB | 14.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 230.1 | 24.3 | 399 | 570146ms | — | — | — | — | 18.16GB | 39.65GB | 2.54GB | 28.02GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
