# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-10 00:15
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| Max Tokens | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 110 | no-quant | 938.2 | 107.2 | 200 | 118ms | — | — | — | — | 2.45GB | 2.62GB | 30MB | 68MB | This is a very fragmented piece of text, appearing to be a m |
| summarization | 1024 | 1008 | no-quant | 1948.4 | 98.7 | 200 | 535ms | — | — | — | — | 2.45GB | 3.22GB | 44MB | 264MB | The provided text is an excerpt from *The Great Gatsby* by F |
| summarization | 4096 | 4088 | no-quant | 1454.3 | 94.4 | 200 | 2812ms | — | — | — | — | 2.45GB | 3.33GB | 77MB | 938MB | This provided text appears to be an excerpt from **The Great |
| summarization | 32768 | 32768 | no-quant | 417.8 | 74.9 | 200 | 78440ms | — | — | — | — | 2.45GB | 3.44GB | 297MB | 7.04GB | This is a fascinating and dense collection of excerpts, clea |
