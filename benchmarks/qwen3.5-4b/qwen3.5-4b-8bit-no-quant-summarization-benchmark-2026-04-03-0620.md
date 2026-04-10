# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 06:20
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
| summarization | 128 | 117 | no-quant | 329.8 | 47.6 | 400 | 356ms | 1.9574 | 2.2246 | 0.0124 | 0.0451 | 4.16GB | 4.48GB | 46MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 417.9 | 47.3 | 327 | 596ms | 1.5222 | 1.9396 | 0.0249 | 0.0693 | 4.16GB | 4.76GB | 46MB | 126MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 458.8 | 47.1 | 400 | 1099ms | 1.8522 | 2.1303 | 0.0015 | 0.0245 | 4.16GB | 5.11GB | 54MB | 198MB | The user wants a summary of the provided text. The text is t |
| summarization | 1024 | 1019 | no-quant | 483.3 | 47.3 | 400 | 2174ms | 1.4910 | 2.0610 | 0.0494 | 0.0113 | 4.16GB | 5.76GB | 70MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 490.1 | 47.0 | 400 | 4407ms | 1.8095 | 2.0827 | 0.0295 | 0.0180 | 4.16GB | 6.52GB | 104MB | 534MB | The user wants a summary of the provided text, which is Chap |
| summarization | 4096 | 4085 | no-quant | 532.6 | 46.2 | 400 | 7800ms | 2.0691 | 2.2159 | 0.0022 | 0.0195 | 4.16GB | 6.72GB | 169MB | 981MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | no-quant | 548.4 | 44.8 | 400 | 15089ms | 1.8331 | 2.1275 | 0.0233 | 0.0175 | 4.16GB | 7.05GB | 295MB | 1.84GB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | no-quant | 540.3 | 42.1 | 400 | 30527ms | 1.8292 | 2.1943 | 0.0622 | 0.0500 | 4.16GB | 7.78GB | 551MB | 3.58GB | The user wants a summary of the provided text, which is Chap |
| summarization | 32768 | 32700 | no-quant | 507.4 | 37.5 | 400 | 64850ms | 1.8404 | 2.8708 | -0.0077 | 0.0256 | 4.16GB | 9.19GB | 1.04GB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   **In |
| summarization | 65536 | 65468 | no-quant | 405.0 | 29.6 | 400 | 163691ms | 2.2807 | 2.6392 | 0.0532 | 0.0298 | 4.16GB | 12.16GB | 2.04GB | 14.07GB | The user wants a summary of the provided text. The text is t |
| summarization | 131072 | 130773 | no-quant | 300.7 | 23.6 | 400 | 435314ms | 1.3918 | 1.8960 | 0.0010 | 0.0438 | 4.16GB | 17.94GB | 4.03GB | 28.02GB | The user wants a summary of the text provided. The text cont |
