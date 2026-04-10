# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-08 16:57
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 108GB |
| macOS | 26.3.1 |

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
| summarization | 128 | 110 | turbo4v2 | 2021.7 | 166.9 | 200 | 56ms | — | — | — | — | 2.45GB | 2.88GB | 8MB | 14MB | The provided text is a very fragmented and disjointed collec |
| summarization | 256 | 249 | turbo4v2 | 4771.5 | 162.6 | 196 | 53ms | — | — | — | — | 2.45GB | 3.28GB | 13MB | 20MB | The provided text is a collection of fragments from or relat |
| summarization | 512 | 496 | turbo4v2 | 5225.7 | 160.3 | 200 | 95ms | — | — | — | — | 2.45GB | 3.92GB | 15MB | 31MB | This excerpt from *The Great Gatsby* begins with a poetic in |
| summarization | 1024 | 1008 | turbo4v2 | 5615.5 | 158.9 | 200 | 184ms | — | — | — | — | 2.45GB | 4.76GB | 23MB | 54MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 2048 | 2031 | turbo4v2 | 5070.1 | 157.6 | 200 | 403ms | — | — | — | — | 2.45GB | 5.81GB | 37MB | 99MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | turbo4v2 | 7772.7 | 150.0 | 200 | 554ms | — | — | — | — | 2.45GB | 5.53GB | 76MB | 191MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 8192 | 8192 | turbo4v2 | 10252.3 | 147.0 | 200 | 800ms | — | — | — | — | 2.45GB | 5.84GB | 139MB | 373MB | This is a fascinating and dense excerpt from **Nick Carraway |
| summarization | 16384 | 16384 | turbo4v2 | 12021.0 | 132.9 | 200 | 1386ms | — | — | — | — | 2.45GB | 6.45GB | 201MB | 737MB | This text appears to be an excerpt or section from **The Gre |
| summarization | 32768 | 32768 | turbo4v2 | 11202.9 | 115.1 | 200 | 2940ms | — | — | — | — | 2.45GB | 7.67GB | 521MB | 1.43GB | This is a highly fragmented and dense excerpt from a work of |
