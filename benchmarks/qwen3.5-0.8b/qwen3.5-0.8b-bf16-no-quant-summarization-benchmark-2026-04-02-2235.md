# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-02 22:35
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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
| summarization | 128 | 119 | no-quant | 1077.5 | 86.5 | 400 | 112ms | 4.9220 | 4.1472 | — | — | 1.40GB | 1.66GB | 18MB | 114MB | Based on the text you provided, here is a summary of *Once A |
| summarization | 256 | 251 | no-quant | 2681.7 | 87.7 | 400 | 94ms | 4.9714 | 4.9206 | — | — | 1.40GB | 1.82GB | 19MB | 142MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 512 | 506 | no-quant | 3128.2 | 85.8 | 400 | 162ms | 3.4555 | 2.6373 | — | — | 1.40GB | 2.24GB | 17MB | 198MB | This text, taken from the essay *In My Own Life* (or similar |
| summarization | 1024 | 1021 | no-quant | 3525.8 | 85.0 | 400 | 290ms | 3.0897 | 4.1766 | — | — | 1.40GB | 2.76GB | 27MB | 311MB | Based on the excerpt provided from *The Great Gatsby* by F.  |
| summarization | 2048 | 2044 | no-quant | 3603.3 | 87.7 | 400 | 568ms | 3.0153 | 4.0422 | — | — | 1.40GB | 3.51GB | 39MB | 535MB | Here is a summary of the provided text:  **1. Themes in Fitz |
| summarization | 4096 | 4087 | no-quant | 4184.9 | 86.7 | 400 | 977ms | 4.2727 | 3.1705 | — | — | 1.40GB | 3.43GB | 63MB | 982MB | Based on the provided text, here is a summary of *The Great  |
| summarization | 8192 | 8192 | no-quant | 4354.4 | 83.3 | 391 | 1890ms | 5.3131 | 3.5844 | — | — | 1.40GB | 3.72GB | 102MB | 1.83GB | Based on the text provided, here is a summary of the content |
| summarization | 16384 | 16363 | no-quant | 4080.6 | 77.6 | 400 | 4104ms | 3.7386 | 1.9803 | — | — | 1.40GB | 3.98GB | 207MB | 3.58GB | Based on the text *The Great Gatsby* by F. Scott Fitzgerald, |
| summarization | 32768 | 32702 | no-quant | 3425.4 | 70.1 | 214 | 10069ms | 4.1377 | 3.9360 | — | — | 1.40GB | 4.82GB | 396MB | 7.03GB | Based on the provided text of *The Great Gatsby*, here is a  |
| summarization | 65536 | 65470 | no-quant | 1848.2 | 47.7 | 400 | 39938ms | 3.2332 | 3.1015 | — | — | 1.40GB | 7.25GB | 783MB | 14.07GB | Based on the text provided (*The Great Gatsby*), here is a s |
| summarization | 131072 | 130775 | no-quant | 1036.3 | 32.9 | 400 | 127646ms | 3.7585 | 3.6352 | — | — | 1.40GB | 8.29GB | 1.01GB | 28.02GB | Based on the text provided, here is a summary of the story t |
