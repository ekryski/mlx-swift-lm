# Inference Benchmark - Gemma 4 E4B

**Date**: 2026-04-07 01:16
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: bf16
**Model**: `mlx-community/gemma-4-e4b-it-bf16`

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
| summarization | 128 | 116 | no-quant | 128.9 | 6.7 | 400 | 905ms | 1.6973 | — | — | — | 14.00GB | 18.21GB | 36MB | 113MB | <\|channel>thought Thinking Process:  1.  **Analyze the Input |
| summarization | 256 | 255 | no-quant | 305.5 | 6.6 | 398 | 835ms | 1.8979 | 1.1186 | — | — | 14.00GB | 18.35GB | 44MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 398.7 | 6.7 | 400 | 1451ms | 1.7455 | — | — | — | 14.00GB | 18.60GB | 51MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 432.9 | 6.7 | 400 | 2719ms | 1.9095 | — | — | — | 14.00GB | 19.12GB | 71MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 458.8 | 6.7 | 400 | 4714ms | 1.9712 | — | — | — | 14.00GB | 20.16GB | 103MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 559.1 | 6.7 | 400 | 7829ms | 2.0459 | — | — | — | 14.00GB | 20.24GB | 121MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8229 | no-quant | 896.2 | 6.5 | 52 | 9237ms | — | 2.1420 | — | — | 14.00GB | 18.10GB | 221MB | 1.77GB | The provided text is a complex excerpt, likely from the begi |
| summarization | 16384 | 16395 | no-quant | 907.0 | 6.3 | 109 | 18136ms | — | 2.7603 | — | — | 14.00GB | 18.30GB | 546MB | 3.53GB | The provided text is a collection of excerpts from **The Gre |
| summarization | 32768 | 32815 | no-quant | 857.7 | 5.8 | 24 | 38626ms | — | 4.3216 | — | — | 14.00GB | 18.78GB | 1.04GB | 7.02GB | This is a complex request, as you've provided a large, highl |
| summarization | 65536 | 65896 | no-quant | 692.9 | 5.3 | 110 | 95479ms | — | 2.3351 | — | — | 14.00GB | 21.82GB | 2.04GB | 14.10GB | The text you provided is a comprehensive compilation of the  |
| summarization | 131072 | 130563 | no-quant | 473.1 | 4.6 | 222 | 276196ms | — | 2.7905 | — | — | 14.00GB | 28.98GB | 4.03GB | 27.94GB | The provided text is a collection of chapter summaries, a hi |
