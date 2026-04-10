# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-09 10:04
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
| summarization | 1024 | 1008 | no-quant | 1244.1 | 92.4 | 200 | 955ms | — | — | — | — | 2.45GB | 10.00GB | 27MB | 264MB | This excerpt from *The Great Gatsby* provides several distin |
| summarization | 4096 | 4088 | no-quant | 1583.4 | 89.2 | 200 | 2757ms | — | — | — | — | 2.45GB | 20.99GB | 53MB | 938MB | This provided text is an excerpt from **The Great Gatsby** b |
| summarization | 16384 | 16384 | no-quant | 2290.0 | 73.8 | 200 | 7332ms | — | — | — | — | 2.45GB | 22.29GB | 233MB | 3.54GB | This is an excerpt from **The Great Gatsby by F. Scott Fitzg |
