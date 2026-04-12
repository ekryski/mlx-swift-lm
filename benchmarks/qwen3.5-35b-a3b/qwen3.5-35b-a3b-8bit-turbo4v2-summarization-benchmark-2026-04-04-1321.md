# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-04 13:21
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
| summarization | 128 | 117 | turbo4v2 | 44.9 | 43.4 | 256 | 2783ms | 1.2506 | 1.3860 | 0.0005 | 0.0143 | 34.30GB | 34.51GB | 21MB | 17MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4v2 | 4.4 | 43.6 | 400 | 57473ms | 1.2109 | 1.4672 | 0.0132 | 0.0165 | 34.30GB | 34.70GB | 47MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4v2 | 7.9 | 43.1 | 400 | 64382ms | 1.2994 | 1.5588 | 0.0162 | 0.0127 | 34.30GB | 35.07GB | 26MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 15.5 | 42.8 | 400 | 66249ms | 1.2234 | 1.5648 | -0.0039 | 0.0404 | 34.30GB | 35.78GB | 29MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4v2 | 30.7 | 42.6 | 400 | 66939ms | 1.1915 | 1.0288 | 0.0196 | -0.0002 | 34.30GB | 36.86GB | 81MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 57.0 | 41.9 | 400 | 72129ms | 1.2238 | 1.7104 | 0.0040 | 0.0274 | 34.30GB | 36.93GB | 37MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4v2 | 110.1 | 40.9 | 400 | 74848ms | 1.2434 | 1.4855 | 0.0145 | 0.0127 | 34.30GB | 37.29GB | 140MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4v2 | 180.8 | 39.0 | 400 | 90983ms | 1.2813 | 1.5525 | 0.0216 | 0.0126 | 34.30GB | 37.88GB | 324MB | 745MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 268.7 | 35.9 | 400 | 122187ms | 1.2835 | 1.5567 | -0.0008 | 0.0075 | 34.30GB | 39.12GB | 544MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4v2 | 297.2 | 30.0 | 400 | 220772ms | 1.4097 | 1.7887 | 0.0227 | 0.0379 | 34.30GB | 41.74GB | 1.29GB | 2.86GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo4v2 | 214.3 | 22.8 | 400 | 610699ms | 1.2905 | 1.5450 | 0.0348 | -0.0003 | 34.30GB | 46.43GB | 2.53GB | 5.69GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
