# Inference Benchmark - Gemma 4 E4B

**Date**: 2026-04-07 17:29
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e4b-it-4bit`

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
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | no-quant | 326.0 | 44.3 | 382 | 359ms | 2.0953 | 1.3690 | 1.0284 | 1.7659 | 3.98GB | 4.46GB | 12MB | 109MB | <\|channel>thought Thinking Process:  1.  **Analyze the Reque |
| summarization | 1024 | 1014 | no-quant | 570.2 | 42.0 | 400 | 1834ms | 1.7406 | — | 0.5817 | — | 3.98GB | 6.21GB | 36MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 730.0 | 38.5 | 400 | 5680ms | 2.1892 | — | 0.7220 | — | 3.98GB | 7.46GB | 122MB | 983MB | <\|channel>thought Here's a plan to summarize the provided te |
| summarization | 32768 | 32815 | no-quant | 851.6 | 21.4 | 400 | 38610ms | — | 2.3340 | 0.6420 | — | 3.98GB | 8.40GB | 1.03GB | 7.10GB | The provided text is the full version of F. Scott Fitzgerald |
