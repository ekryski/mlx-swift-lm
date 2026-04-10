# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 09:01
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
| summarization | 128 | 117 | affine-4 | 337.2 | 64.8 | 400 | 349ms | 1.6375 | 1.9821 | 0.0427 | 0.1174 | 2.20GB | 2.52GB | 33MB | 35MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 429.1 | 63.4 | 400 | 581ms | 1.2990 | 1.5445 | 0.0505 | 0.0844 | 2.20GB | 2.83GB | 25MB | 44MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 461.2 | 61.2 | 400 | 1094ms | 1.4619 | 1.8561 | 0.0581 | 0.0822 | 2.20GB | 3.33GB | 27MB | 62MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 468.3 | 60.6 | 400 | 2270ms | 1.3805 | 2.2480 | 0.0753 | 0.1407 | 2.20GB | 3.25GB | 39MB | 97MB | Here's a thinking process that leads to the suggested summar |
| summarization | 2048 | 2042 | affine-4 | 481.6 | 58.7 | 400 | 4300ms | 1.8822 | 1.9080 | 0.1410 | 0.0935 | 2.20GB | 4.32GB | 41MB | 167MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 521.2 | 57.4 | 400 | 7896ms | 1.3797 | 1.3938 | 0.0940 | 0.0255 | 2.20GB | 4.47GB | 63MB | 307MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-4 | 534.9 | 53.0 | 400 | 15367ms | 2.0082 | 2.1797 | 0.1251 | 0.0618 | 2.20GB | 4.79GB | 95MB | 587MB | The user wants a summary of the provided text, which is Chap |
| summarization | 16384 | 16361 | affine-4 | 529.3 | 46.5 | 400 | 30970ms | 1.7739 | 3.2786 | 0.1051 | 0.1189 | 2.20GB | 5.38GB | 172MB | 1.12GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | affine-4 | 498.4 | 38.3 | 400 | 65686ms | 1.7550 | 2.5958 | 0.1443 | 0.1184 | 2.21GB | 6.59GB | 318MB | 2.21GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | affine-4 | 383.5 | 27.6 | 400 | 170944ms | 1.7956 | 3.4950 | 0.0601 | 0.2000 | 2.21GB | 9.64GB | 605MB | 4.40GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 282.0 | 17.7 | 400 | 463939ms | 1.6495 | 3.8732 | 0.1199 | 0.2342 | 2.21GB | 15.97GB | 1.15GB | 8.76GB | Here's a thinking process that leads to the summary:  1.  ** |
