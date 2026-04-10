# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-02 22:39
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
| summarization | 128 | 119 | affine-4 | 1142.5 | 87.1 | 400 | 105ms | 5.3511 | 3.6760 | 0.0244 | 0.0653 | 1.40GB | 1.61GB | 11MB | 35MB | Based on the excerpt provided from *The Great Gatsby* (speci |
| summarization | 256 | 251 | affine-4 | 2091.5 | 86.2 | 400 | 120ms | 3.7309 | 3.4340 | 0.0296 | 0.0328 | 1.40GB | 1.82GB | 9MB | 45MB | This text appears to be an excerpt from a story titled **"On |
| summarization | 512 | 506 | affine-4 | 2688.6 | 79.7 | 400 | 189ms | 4.3804 | 3.9254 | 0.0321 | 0.0736 | 1.40GB | 2.24GB | 14MB | 62MB | Based on the text provided from *The Great Gatsby*, here is  |
| summarization | 1024 | 1021 | affine-4 | 3015.9 | 78.6 | 400 | 339ms | 3.9663 | 4.4988 | 0.0501 | 0.0630 | 1.40GB | 2.19GB | 15MB | 97MB | Here is a summary of the text provided:  The author, Zelda,  |
| summarization | 2048 | 2044 | affine-4 | 3264.1 | 78.3 | 400 | 627ms | 4.0105 | 4.9272 | 0.0592 | 0.0536 | 1.40GB | 2.98GB | 15MB | 167MB | Based on the text provided by F. Scott Fitzgerald, here is a |
| summarization | 4096 | 4087 | affine-4 | 3794.9 | 78.8 | 325 | 1077ms | 4.0323 | 4.7286 | 0.0560 | 0.0870 | 1.40GB | 3.05GB | 25MB | 302MB | Based on the provided excerpts from **The Great Gatsby** by  |
| summarization | 8192 | 8192 | affine-4 | 4201.1 | 75.7 | 385 | 1950ms | 3.5886 | 5.6367 | 0.0484 | 0.0519 | 1.40GB | 3.21GB | 37MB | 586MB | Here is a summary of the provided text, which is an excerpt  |
| summarization | 16384 | 16363 | affine-4 | 3947.3 | 70.2 | 400 | 4146ms | 3.6862 | 4.6965 | 0.0125 | 0.0677 | 1.40GB | 3.51GB | 43MB | 1.12GB | Here is a summary of the provided text, titled **"Once again |
| summarization | 32768 | 32702 | affine-4 | 3256.5 | 63.2 | 400 | 10063ms | 4.6972 | 4.4507 | 0.0804 | 0.0430 | 1.40GB | 4.76GB | 119MB | 2.21GB | Based on the text provided (*The Great Gatsby*), here is a s |
| summarization | 65536 | 65470 | affine-4 | 1649.6 | 51.2 | 400 | 39755ms | 3.8051 | 1.0299 | 0.0528 | -0.0051 | 1.40GB | 5.70GB | 227MB | 4.40GB | The text provided is a memoir by **F. Scott Fitzgerald** tit |
| summarization | 131072 | 130775 | affine-4 | 1023.0 | 38.0 | 400 | 127902ms | 3.3052 | 4.2768 | 0.0580 | 0.0811 | 1.40GB | 8.29GB | 442MB | 8.76GB | Based on the text provided, here is a summary of **The Great |
