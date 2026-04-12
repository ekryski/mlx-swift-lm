# Inference Benchmark - Qwen3.5 4B

- **Date**: 2026-04-03 10:24
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-4B-4bit`

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
| summarization | 128 | 117 | turbo4v2 | 333.1 | 65.3 | 400 | 353ms | 2.5323 | 2.6138 | 0.1178 | 0.1477 | 2.20GB | 2.52GB | 49MB | 23MB | The user is asking for a summary of the provided text, which |
| summarization | 256 | 249 | turbo4v2 | 425.5 | 64.9 | 400 | 586ms | 1.4535 | 2.1400 | 0.0748 | 0.1124 | 2.20GB | 2.83GB | 45MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4v2 | 462.4 | 64.6 | 400 | 1090ms | 1.3744 | 2.0776 | 0.0336 | 0.1580 | 2.20GB | 3.33GB | 55MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 480.3 | 64.6 | 400 | 2217ms | 1.7377 | 1.8637 | 0.1021 | 0.0805 | 2.20GB | 3.80GB | 71MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4v2 | 484.2 | 63.7 | 400 | 4363ms | 1.4230 | 1.8593 | 0.0612 | 0.0366 | 2.20GB | 4.55GB | 104MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 526.3 | 62.3 | 400 | 7858ms | 1.6252 | 2.4558 | 0.0733 | 0.2008 | 2.20GB | 4.75GB | 170MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4v2 | 542.6 | 59.9 | 400 | 15206ms | 2.2424 | 1.7500 | 0.1529 | 0.0443 | 2.20GB | 5.16GB | 298MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4v2 | 533.3 | 55.5 | 400 | 30897ms | 1.5247 | 2.4027 | 0.0978 | 0.1354 | 2.20GB | 5.81GB | 481MB | 745MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 502.4 | 47.4 | 400 | 65443ms | 2.0320 | 2.5576 | 0.1168 | 0.0815 | 2.21GB | 7.22GB | 1.04GB | 1.44GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | turbo4v2 | 392.5 | 34.9 | 400 | 168883ms | 2.0794 | 2.1252 | 0.1020 | 0.1194 | 2.21GB | 10.04GB | 2.04GB | 2.86GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo4v2 | 247.5 | 24.9 | 400 | 537338ms | 1.6769 | 2.0227 | 0.0524 | 0.0870 | 2.21GB | 15.96GB | 3.53GB | 5.69GB | The user wants a summary of the provided text. The text cont |
