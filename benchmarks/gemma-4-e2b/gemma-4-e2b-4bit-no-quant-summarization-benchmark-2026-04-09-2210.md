# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-09 22:10
**Branch**: `session/all-perf-fixes`
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
| summarization | 128 | 110 | no-quant | 2782.2 | 173.9 | 200 | 40ms | — | — | — | — | 2.45GB | 2.80GB | 8MB | 68MB | The provided text is a very fragmented collection of element |
| summarization | 1024 | 1008 | no-quant | 7493.0 | 133.9 | 200 | 135ms | — | — | — | — | 2.45GB | 3.57GB | 159MB | 264MB | This provided text is an excerpt from **The Great Gatsby** b |
| summarization | 4096 | 4088 | no-quant | 7510.4 | 136.1 | 200 | 545ms | — | — | — | — | 2.45GB | 3.83GB | 208MB | 938MB | This excerpt is a collection of excerpts from *The Great Gat |
| summarization | 8192 | 8192 | no-quant | 7403.3 | 134.4 | 200 | 1107ms | — | — | — | — | 2.45GB | 4.17GB | 328MB | 1.79GB | This is an excerpt from **The Great Gatsby** by **F. Scott F |
