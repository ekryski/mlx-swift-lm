# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 07:40
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
| summarization | 128 | 117 | turbo3 | 331.5 | 47.3 | 400 | 355ms | 1.5190 | 1.8909 | 0.0350 | 0.0316 | 4.16GB | 4.48GB | 43MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo3 | 422.7 | 47.2 | 400 | 590ms | 1.2462 | 1.5227 | 0.0066 | 0.0221 | 4.16GB | 4.76GB | 45MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 457.8 | 46.9 | 400 | 1102ms | 1.4146 | 2.1326 | 0.0272 | 0.0182 | 4.16GB | 5.11GB | 55MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo3 | 483.9 | 47.0 | 400 | 2151ms | 1.8915 | 2.1083 | 0.0312 | 0.0204 | 4.16GB | 5.76GB | 64MB | 63MB | The user wants a summary of the provided text from "The Grea |
| summarization | 2048 | 2042 | turbo3 | 490.0 | 46.6 | 400 | 4300ms | 1.3811 | 2.2086 | 0.0228 | 0.0187 | 4.16GB | 6.52GB | 103MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo3 | 532.5 | 45.9 | 400 | 7799ms | 2.4444 | 2.1824 | 0.0358 | 0.0295 | 4.16GB | 6.72GB | 168MB | 199MB | The user wants a summary of the provided text from the novel |
| summarization | 8192 | 8190 | turbo3 | 548.8 | 44.7 | 400 | 15061ms | 1.9200 | 2.2888 | 0.0144 | 0.0535 | 4.16GB | 7.05GB | 294MB | 382MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | turbo3 | 539.4 | 41.8 | 400 | 30575ms | 1.9945 | 3.4292 | 0.0127 | 0.0388 | 4.16GB | 7.78GB | 550MB | 745MB | The user wants a summary of the provided text, which consist |
| summarization | 32768 | 32700 | turbo3 | 507.2 | 37.4 | 400 | 64885ms | 2.2797 | 1.8749 | 0.0458 | 0.0821 | 4.16GB | 9.19GB | 1.04GB | 1.44GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | turbo3 | 386.8 | 27.2 | 400 | 172467ms | 1.9836 | 1.0509 | 0.0305 | 0.0014 | 4.16GB | 12.17GB | 2.04GB | 2.86GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo3 | 291.0 | 22.6 | 400 | 453185ms | 1.5869 | 1.7275 | 0.0228 | 0.0233 | 4.16GB | 17.94GB | 4.03GB | 5.69GB | The user wants a summary of the provided text. The text cont |
