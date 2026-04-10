# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-04 10:57
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-35B-A3B-8bit`

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
| summarization | 128 | 117 | turbo4 | 45.2 | 43.9 | 267 | 2781ms | 1.2070 | 1.4845 | -0.0056 | -0.0014 | 34.30GB | 34.51GB | 41MB | 22MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4 | 3.9 | 44.3 | 400 | 65062ms | 1.3789 | 1.7120 | -0.0069 | 0.0053 | 34.30GB | 34.70GB | 46MB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4 | 8.0 | 42.9 | 400 | 63378ms | 1.2290 | 1.5509 | 0.0270 | 0.0231 | 34.30GB | 35.07GB | 45MB | 53MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4 | 16.3 | 42.8 | 400 | 62831ms | 1.3171 | 1.6114 | -0.0122 | 0.0110 | 34.30GB | 35.78GB | 19MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 32.3 | 42.5 | 400 | 63606ms | 1.3181 | 1.6403 | 0.0030 | 0.0096 | 34.30GB | 36.86GB | 80MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4 | 56.2 | 42.0 | 400 | 73171ms | 1.2932 | 1.6442 | 0.0122 | 0.0313 | 34.30GB | 36.93GB | 70MB | 261MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4 | 115.3 | 40.6 | 400 | 71472ms | 1.4761 | 1.8401 | 0.0223 | 0.0024 | 34.30GB | 37.29GB | 140MB | 499MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4 | 184.8 | 38.6 | 400 | 89047ms | 1.2755 | 1.5828 | -0.0042 | 0.0148 | 34.30GB | 37.88GB | 253MB | 974MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4 | 272.0 | 35.6 | 400 | 120739ms | 1.3589 | 1.7656 | 0.0512 | 0.0469 | 34.30GB | 39.12GB | 545MB | 1.88GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4 | 293.5 | 30.0 | 400 | 223581ms | 1.2968 | 1.2655 | 0.0029 | -0.0051 | 34.30GB | 41.74GB | 1.29GB | 3.74GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo4 | 217.1 | 22.8 | 400 | 602993ms | 1.2958 | 1.7218 | -0.0040 | 0.0474 | 34.30GB | 46.43GB | 2.54GB | 7.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
