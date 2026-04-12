# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-04 17:09
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
| summarization | 128 | 117 | turbo3 | 210.8 | 52.3 | 296 | 556ms | 1.4864 | 1.8509 | 0.0789 | 0.1280 | 18.16GB | 18.43GB | 42MB | 18MB | The user has provided the front matter of F. Scott Fitzgeral |
| summarization | 256 | 249 | turbo3 | 49.2 | 52.7 | 325 | 5270ms | 1.2371 | 1.4753 | 0.0614 | 0.0309 | 18.16GB | 18.60GB | 47MB | 26MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 39.6 | 52.0 | 400 | 13096ms | 1.5885 | 1.5902 | 0.0793 | 0.0596 | 18.16GB | 18.94GB | 42MB | 40MB | The user wants a summary of the provided text.  **Source Tex |
| summarization | 1024 | 1019 | turbo3 | 80.4 | 52.0 | 400 | 13114ms | 1.5981 | 2.0625 | 0.0891 | 0.0869 | 18.16GB | 19.65GB | 42MB | 63MB | The user wants a summary of the provided text from "The Grea |
| summarization | 2048 | 2042 | turbo3 | 125.9 | 51.6 | 400 | 16651ms | 1.4625 | 1.5206 | 0.0901 | 0.1018 | 18.16GB | 20.72GB | 76MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo3 | 226.8 | 50.4 | 400 | 18459ms | 1.6125 | 1.9734 | 0.0914 | 0.0776 | 18.16GB | 20.80GB | 71MB | 199MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | turbo3 | 311.9 | 48.6 | 400 | 26708ms | 1.7364 | 2.4820 | 0.1034 | 0.1093 | 18.16GB | 21.15GB | 141MB | 382MB | The user wants a summary of the text provided, which is Chap |
| summarization | 16384 | 16361 | turbo3 | 409.9 | 45.3 | 400 | 40337ms | 1.3562 | 1.8404 | 0.1086 | 0.1267 | 18.16GB | 21.75GB | 360MB | 745MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | turbo3 | 437.2 | 39.7 | 400 | 75277ms | 1.2398 | 1.4602 | 0.0666 | 0.1125 | 18.16GB | 22.98GB | 679MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo3 | 376.1 | 33.5 | 398 | 174597ms | 1.6416 | 1.4560 | 0.1049 | 0.0589 | 18.16GB | 25.60GB | 1.29GB | 2.86GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo3 | 244.9 | 25.0 | 400 | 534503ms | 1.5207 | 1.8202 | 0.1021 | 0.1364 | 18.16GB | 30.29GB | 2.54GB | 5.69GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
