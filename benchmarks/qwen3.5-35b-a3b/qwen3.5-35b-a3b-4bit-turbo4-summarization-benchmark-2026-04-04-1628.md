# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-04 16:28
- **Branch**: `ek/consolidated-benchmarks`
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
| summarization | 128 | 117 | turbo4 | 212.5 | 52.8 | 400 | 552ms | 1.4673 | 1.6483 | 0.0874 | 0.0308 | 18.16GB | 18.43GB | 27MB | 30MB | The user wants a summary of the provided text snippet.  1.   |
| summarization | 256 | 249 | turbo4 | 46.8 | 52.5 | 400 | 5583ms | 1.3832 | 1.8772 | 0.0879 | 0.1147 | 18.16GB | 18.60GB | 37MB | 38MB | The user wants a summary of the provided text, which is the  |
| summarization | 512 | 504 | turbo4 | 36.9 | 51.9 | 400 | 14071ms | 1.3835 | 1.7875 | 0.0684 | 0.1179 | 18.16GB | 18.94GB | 12MB | 53MB | The user wants a summary of the provided text, which is Chap |
| summarization | 1024 | 1019 | turbo4 | 76.1 | 51.9 | 400 | 13860ms | 1.2014 | 1.5093 | 0.0541 | 0.0694 | 18.16GB | 19.65GB | 37MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 122.7 | 51.6 | 400 | 17073ms | 1.3473 | 1.9356 | 0.1345 | 0.1150 | 18.16GB | 20.72GB | 82MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4 | 223.5 | 50.4 | 400 | 18708ms | 1.4571 | 1.9787 | 0.0370 | 0.1214 | 18.16GB | 20.80GB | 72MB | 261MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4 | 314.4 | 48.3 | 400 | 26488ms | 1.8125 | 1.4340 | 0.1064 | 0.0578 | 18.16GB | 21.15GB | 119MB | 499MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4 | 403.7 | 44.8 | 400 | 41014ms | 1.6272 | 1.2888 | 0.1359 | 0.0660 | 18.16GB | 21.76GB | 215MB | 974MB | The user wants a summary of the provided text. The text is t |
| summarization | 32768 | 32700 | turbo4 | 426.7 | 39.9 | 400 | 77114ms | 1.3190 | 1.9427 | 0.0612 | 0.1071 | 18.16GB | 22.98GB | 681MB | 1.88GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4 | 367.8 | 34.0 | 400 | 178525ms | 1.1452 | 1.5093 | 0.0375 | 0.0704 | 18.16GB | 25.60GB | 923MB | 3.74GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo4 | 250.5 | 24.9 | 400 | 522487ms | 1.2301 | 1.5906 | 0.0469 | 0.1062 | 18.16GB | 30.29GB | 2.53GB | 7.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
