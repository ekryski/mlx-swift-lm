# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 14:31
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-2B-bf16`

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
| summarization | 128 | 119 | no-quant | 568.8 | 54.4 | 400 | 210ms | 2.9030 | 3.0946 | — | — | 3.51GB | 3.76GB | 18MB | 114MB | The text provided is **not** the full content of F. Scott Fi |
| summarization | 256 | 251 | no-quant | 1400.3 | 54.3 | 400 | 180ms | 2.4807 | 1.7434 | — | — | 3.51GB | 3.94GB | 18MB | 142MB | The text you provided contains a mix of two distinct element |
| summarization | 512 | 506 | no-quant | 1554.3 | 55.0 | 376 | 326ms | 3.3572 | 1.0352 | — | — | 3.51GB | 4.15GB | 21MB | 193MB | The provided text presents F. Scott Fitzgerald's father, Art |
| summarization | 1024 | 1021 | no-quant | 1719.6 | 55.2 | 201 | 594ms | 5.4580 | 1.5440 | — | — | 3.51GB | 4.68GB | 20MB | 267MB | Here is a summary of the provided text:  **F. Scott Fitzgera |
| summarization | 2048 | 2044 | no-quant | 1766.9 | 54.5 | 400 | 1157ms | 3.8286 | 2.9475 | — | — | 3.51GB | 5.50GB | 39MB | 535MB | Here is a summary of the text provided:  **Overview** This t |
| summarization | 4096 | 4087 | no-quant | 2024.3 | 53.7 | 201 | 2042ms | 3.9342 | 1.5396 | — | — | 3.51GB | 5.41GB | 59MB | 938MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 8192 | 8192 | no-quant | 2139.2 | 53.4 | 201 | 3859ms | 2.8499 | 1.4823 | — | — | 3.51GB | 5.60GB | 99MB | 1.79GB | Here is a summary of the provided text:  **Setting the Scene |
| summarization | 16384 | 16363 | no-quant | 2115.5 | 50.1 | 400 | 7862ms | 3.7776 | 2.8285 | — | — | 3.51GB | 5.89GB | 190MB | 3.58GB | Here is a summary of *The Great Gatsby* from F. Scott Fitzge |
| summarization | 32768 | 32702 | no-quant | 1919.2 | 46.6 | 400 | 17409ms | 2.9747 | 3.3785 | — | — | 3.51GB | 6.51GB | 399MB | 7.07GB | Here is a summary of the provided text, **"The Great Gatsby" |
| summarization | 65536 | 65470 | no-quant | 1270.9 | 34.5 | 400 | 57107ms | 2.7701 | 2.4240 | — | — | 3.51GB | 7.84GB | 784MB | 14.07GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | no-quant | 776.0 | 27.9 | 400 | 173127ms | 3.0551 | 2.1542 | — | — | 3.51GB | 10.22GB | 1.51GB | 28.02GB | Based on the provided text, here is a summary of *The Great  |
