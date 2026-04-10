# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-07 23:35
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| summarization | 1024 | 1008 | turbo8v4 | 994.2 | 80.3 | 200 | 1016ms | — | 1.6086 | — | — | 2.45GB | 4.76GB | 28MB | 103MB | The provided text is an excerpt from *The Great Gatsby* by F |
| summarization | 4096 | 4088 | turbo8v4 | 1589.1 | 77.9 | 200 | 2619ms | — | 1.4879 | — | — | 2.45GB | 5.53GB | 76MB | 366MB | This excerpt appears to be a collection of fragmented passag |
| summarization | 16384 | 16384 | turbo8v4 | 2570.7 | 68.6 | 200 | 6445ms | — | 1.6129 | — | — | 2.45GB | 6.45GB | 200MB | 1.38GB | This provided text is a collection of excerpts from **F. Sco |
