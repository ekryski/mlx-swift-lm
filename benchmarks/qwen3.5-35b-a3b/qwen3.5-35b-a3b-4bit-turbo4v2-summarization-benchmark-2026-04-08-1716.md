# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-08 17:16
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
| summarization | 128 | 107 | turbo4v2 | 632.8 | 146.2 | 200 | 170ms | — | — | — | — | 18.16GB | 18.38GB | 40MB | 14MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 239 | turbo4v2 | 515.7 | 146.1 | 200 | 464ms | — | — | — | — | 18.16GB | 18.67GB | 38MB | 20MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 494 | turbo4v2 | 2579.2 | 147.5 | 200 | 193ms | — | — | — | — | 18.16GB | 19.01GB | 44MB | 31MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1009 | turbo4v2 | 3527.2 | 144.8 | 200 | 291ms | — | — | — | — | 18.16GB | 19.70GB | 52MB | 54MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2032 | turbo4v2 | 3659.9 | 144.7 | 200 | 573ms | — | — | — | — | 18.16GB | 20.73GB | 74MB | 99MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | turbo4v2 | 4418.7 | 142.8 | 200 | 952ms | — | — | — | — | 18.16GB | 23.24GB | 115MB | 190MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8180 | turbo4v2 | 4430.2 | 133.0 | 200 | 1874ms | — | — | — | — | 18.16GB | 23.83GB | 195MB | 372MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16351 | turbo4v2 | 3670.9 | 127.4 | 200 | 4478ms | — | — | — | — | 18.16GB | 24.95GB | 354MB | 735MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32690 | turbo4v2 | 3360.8 | 115.9 | 200 | 9810ms | — | — | — | — | 18.16GB | 27.14GB | 676MB | 1.43GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
