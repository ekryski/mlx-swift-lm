# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 20:44
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | turbo4v2 | 586.6 | 79.1 | 400 | 204ms | 3.3967 | 2.2992 | 0.0438 | 0.0379 | 1.86GB | 2.08GB | 11MB | 23MB | The text provided appears to be an excerpt from **F. Scott F |
| summarization | 1024 | 1021 | turbo4v2 | 1040.8 | 79.0 | 201 | 982ms | 2.1437 | 1.4272 | 0.0247 | 0.3277 | 1.86GB | 3.27GB | 16MB | 54MB | This excerpt from **Chapter I** of *The Great Gatsby* by F.  |
| summarization | 4096 | 4087 | turbo4v2 | 1242.8 | 76.7 | 201 | 3351ms | 2.6766 | 1.3085 | 0.0216 | 0.0559 | 1.86GB | 3.95GB | 61MB | 191MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby*,  |
| summarization | 32768 | 32702 | turbo4v2 | 1275.1 | 63.1 | 400 | 25999ms | 2.9485 | 1.0160 | 0.0116 | -0.0005 | 1.86GB | 5.00GB | 399MB | 1.44GB | This excerpt is the opening sequence of F. Scott Fitzgerald' |
