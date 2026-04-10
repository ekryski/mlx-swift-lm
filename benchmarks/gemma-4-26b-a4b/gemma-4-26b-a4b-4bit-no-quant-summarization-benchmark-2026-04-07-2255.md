# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-07 22:55
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`

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
| summarization | 1024 | 1012 | no-quant | 594.9 | 31.0 | 200 | 1825ms | — | 1.1993 | — | — | 13.48GB | 16.13GB | 424MB | 265MB | The provided text is the opening of F. Scott Fitzgerald’s *T |
| summarization | 4096 | 4092 | no-quant | 613.3 | 29.7 | 200 | 6965ms | — | 1.3193 | — | — | 13.48GB | 18.41GB | 552MB | 939MB | This text constitutes the opening of F. Scott Fitzgerald's n |
| summarization | 16384 | 16383 | no-quant | 588.5 | 25.6 | 200 | 28170ms | — | 1.2625 | — | — | 13.48GB | 20.38GB | 384MB | 3.54GB | The text you have provided is the beginning of **F. Scott Fi |
