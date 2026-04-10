# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-08 17:21
**Branch**: `tom/m5-max-gemma4-e2b-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 108GB |
| macOS | 26.3.1 |

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
| summarization | 128 | 107 | turbo4v2 | 1089.2 | 144.4 | 200 | 99ms | — | — | — | — | 18.16GB | 18.41GB | 39MB | 14MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 239 | turbo4v2 | 525.1 | 145.2 | 200 | 456ms | — | — | — | — | 18.16GB | 18.67GB | 41MB | 20MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 494 | turbo4v2 | 1930.7 | 145.9 | 200 | 258ms | — | — | — | — | 18.16GB | 19.01GB | 46MB | 31MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1009 | turbo4v2 | 3518.5 | 141.6 | 200 | 293ms | — | — | — | — | 18.16GB | 19.70GB | 56MB | 54MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2032 | turbo4v2 | 3696.5 | 144.9 | 200 | 566ms | — | — | — | — | 18.16GB | 20.73GB | 71MB | 99MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | turbo4v2 | 4384.8 | 140.9 | 200 | 959ms | — | — | — | — | 18.16GB | 23.24GB | 114MB | 190MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8180 | turbo4v2 | 3981.2 | 126.4 | 200 | 2082ms | — | — | — | — | 18.16GB | 23.83GB | 194MB | 372MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16351 | turbo4v2 | 3712.8 | 124.1 | 200 | 4443ms | — | — | — | — | 18.16GB | 24.95GB | 355MB | 735MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32690 | turbo4v2 | 3253.4 | 112.9 | 200 | 10124ms | — | — | — | — | 18.16GB | 27.14GB | 674MB | 1.43GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
