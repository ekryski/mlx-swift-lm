# Inference Benchmark - Qwen3.5 4B

- **Date**: 2026-04-03 08:35
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
| summarization | 128 | 117 | no-quant | 337.1 | 65.0 | 400 | 349ms | 1.4591 | 2.1867 | 0.0611 | 0.0870 | 2.20GB | 2.52GB | 45MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 423.1 | 64.5 | 400 | 589ms | 1.3254 | 1.6523 | 0.0403 | 0.0781 | 2.20GB | 2.83GB | 36MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 461.3 | 64.1 | 400 | 1093ms | 1.9754 | 2.3081 | 0.1260 | 0.1243 | 2.20GB | 3.33GB | 50MB | 198MB | The user is asking for a summary of the provided text. The t |
| summarization | 1024 | 1019 | no-quant | 483.7 | 63.9 | 400 | 2139ms | 1.5381 | 1.7119 | 0.0431 | 0.0533 | 2.20GB | 3.80GB | 67MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 484.4 | 63.1 | 400 | 4274ms | 1.6079 | 2.3115 | 0.0209 | 0.1455 | 2.20GB | 4.55GB | 104MB | 534MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 526.1 | 61.8 | 400 | 7855ms | 1.3183 | 2.1825 | 0.0395 | -0.0137 | 2.20GB | 4.75GB | 166MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 541.9 | 59.4 | 400 | 15222ms | 1.5496 | 2.7699 | 0.1248 | 0.1394 | 2.20GB | 5.16GB | 295MB | 1.84GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 534.5 | 55.0 | 400 | 30810ms | 2.1546 | 3.4935 | 0.1873 | 0.1724 | 2.20GB | 5.81GB | 552MB | 3.58GB | The user wants a summary of the provided text, which contain |
| summarization | 32768 | 32700 | no-quant | 502.0 | 47.1 | 400 | 65557ms | 1.8714 | 1.6046 | 0.1780 | 0.0909 | 2.21GB | 7.22GB | 1.04GB | 7.07GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | no-quant | 394.0 | 32.5 | 400 | 169892ms | 1.4725 | 2.3394 | 0.0410 | 0.1121 | 2.21GB | 10.05GB | 2.04GB | 14.07GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | no-quant | 300.6 | 25.5 | 400 | 435608ms | 1.6650 | 2.2671 | 0.1142 | 0.1854 | 2.21GB | 15.97GB | 4.03GB | 28.02GB | The user wants a summary of the provided text. The text cont |
