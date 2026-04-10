# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 06:47
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-4B-8bit`

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
| summarization | 128 | 117 | affine-4 | 325.9 | 47.2 | 317 | 361ms | 1.5672 | 2.2131 | 0.0318 | 0.0393 | 4.16GB | 4.48GB | 39MB | 30MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 416.3 | 46.4 | 400 | 599ms | 1.5517 | 1.7861 | 0.0253 | 0.0533 | 4.16GB | 4.76GB | 31MB | 44MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 459.0 | 45.7 | 400 | 1099ms | 1.4046 | 1.8631 | 0.0402 | 0.0281 | 4.16GB | 5.11GB | 32MB | 62MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 476.4 | 45.2 | 400 | 2307ms | 1.3391 | 1.8512 | 0.0098 | 0.0257 | 4.16GB | 5.21GB | 37MB | 97MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 485.1 | 44.7 | 201 | 4303ms | 1.8862 | 5.3429 | 0.0508 | -0.0202 | 4.16GB | 6.28GB | 44MB | 153MB | The user wants a summary of the provided text, which is Chap |
| summarization | 4096 | 4085 | affine-4 | 523.1 | 43.5 | 400 | 7945ms | 1.9722 | 2.1328 | 0.0574 | 0.0515 | 4.16GB | 6.43GB | 61MB | 307MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | affine-4 | 544.6 | 40.8 | 400 | 15129ms | 1.9847 | 2.7989 | 0.0336 | 0.0185 | 4.16GB | 6.75GB | 95MB | 587MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | affine-4 | 535.5 | 37.3 | 400 | 30653ms | 2.0766 | 2.9010 | 0.0463 | 0.0519 | 4.16GB | 7.33GB | 172MB | 1.12GB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | affine-4 | 502.0 | 31.5 | 400 | 65253ms | 2.1336 | 2.3338 | 0.0203 | 0.0864 | 4.16GB | 8.54GB | 298MB | 2.21GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | affine-4 | 403.1 | 24.0 | 400 | 162565ms | 1.4515 | 2.1785 | 0.0530 | 0.0507 | 4.16GB | 11.56GB | 603MB | 4.40GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 302.0 | 16.1 | 400 | 433176ms | 1.8376 | 1.6222 | 0.0453 | 0.0351 | 4.16GB | 17.94GB | 1.15GB | 8.76GB | The user wants a summary of the text provided. The text prov |
