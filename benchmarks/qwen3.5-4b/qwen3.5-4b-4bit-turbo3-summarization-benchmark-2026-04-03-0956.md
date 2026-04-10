# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 09:56
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-4B-4bit`

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
| summarization | 128 | 117 | turbo3 | 336.9 | 65.5 | 400 | 349ms | 1.5787 | 2.0235 | 0.0639 | 0.0164 | 2.20GB | 2.52GB | 47MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo3 | 424.1 | 65.1 | 400 | 588ms | 1.5802 | 1.8302 | 0.0499 | 0.1145 | 2.20GB | 2.83GB | 49MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 460.2 | 64.7 | 400 | 1096ms | 1.3579 | 3.1925 | 0.0513 | 0.1207 | 2.20GB | 3.33GB | 56MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo3 | 481.4 | 64.8 | 400 | 2149ms | 1.2966 | 1.7625 | 0.0610 | 0.0949 | 2.20GB | 3.80GB | 70MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo3 | 484.6 | 64.0 | 400 | 4314ms | 1.3112 | 2.1557 | 0.0242 | 0.1331 | 2.20GB | 4.55GB | 103MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo3 | 522.7 | 62.4 | 399 | 7919ms | 1.8033 | 2.4176 | 0.1062 | 0.1327 | 2.20GB | 4.75GB | 168MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo3 | 542.5 | 59.8 | 400 | 15216ms | 2.0111 | 2.5969 | 0.0591 | 0.0967 | 2.20GB | 5.16GB | 277MB | 382MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | turbo3 | 533.2 | 55.6 | 400 | 30902ms | 2.0147 | 2.3165 | 0.2191 | 0.1716 | 2.20GB | 5.81GB | 552MB | 745MB | The user wants a summary of the provided text, which is Chap |
| summarization | 32768 | 32700 | turbo3 | 501.9 | 47.4 | 400 | 65503ms | 1.7749 | 2.3771 | 0.1084 | 0.1160 | 2.21GB | 7.22GB | 1.04GB | 1.44GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | turbo3 | 426.9 | 38.9 | 201 | 153743ms | 2.0335 | 1.7477 | 0.0785 | 0.9281 | 2.21GB | 10.07GB | 1.90GB | 2.85GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo3 | 268.5 | 25.2 | 400 | 493572ms | 1.9331 | 2.2939 | 0.0817 | 0.0898 | 2.21GB | 15.97GB | 3.53GB | 5.69GB | The user wants a summary of the provided text. The text cont |
