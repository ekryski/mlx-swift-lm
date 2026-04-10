# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-05 22:27
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| summarization | 128 | 117 | turbo4v2 | 231.9 | 52.4 | 400 | 506ms | 1.5631 | 1.9914 | — | — | 18.16GB | 18.43GB | 36MB | 23MB | The user wants a summary of the provided text.  **1. Analyze |
| summarization | 1024 | 1019 | turbo4v2 | 472.9 | 51.9 | 400 | 2440ms | 1.5487 | 1.5935 | — | — | 18.16GB | 19.72GB | 60MB | 63MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | turbo4v2 | 514.2 | 49.3 | 400 | 8476ms | 1.5222 | 1.6440 | — | — | 18.16GB | 23.25GB | 47MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 399.2 | 37.5 | 400 | 93243ms | 1.4723 | 1.4173 | — | — | 18.16GB | 27.16GB | 681MB | 1.44GB | The user wants a summary of the provided text, which is the  |
