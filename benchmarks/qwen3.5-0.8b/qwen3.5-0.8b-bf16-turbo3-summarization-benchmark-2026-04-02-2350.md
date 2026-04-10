# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-02 23:50
**Branch**: `ek/consolidated-benchmarks`
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
| summarization | 128 | 119 | turbo3 | 1156.6 | 90.1 | 400 | 104ms | 4.3982 | 3.8147 | 0.0712 | 0.0174 | 1.40GB | 1.61GB | 5MB | 23MB | This text excerpt is titled "**Once again** to **Zelda**" an |
| summarization | 256 | 251 | turbo3 | 2090.4 | 88.5 | 301 | 120ms | 4.5355 | 3.0363 | 0.0193 | 0.0104 | 1.40GB | 1.82GB | 11MB | 25MB | Based on the text you provided, here is a summary of the exc |
| summarization | 512 | 506 | turbo3 | 2687.3 | 88.1 | 400 | 189ms | 4.1820 | 4.1458 | 0.0143 | 0.0356 | 1.40GB | 2.24GB | 21MB | 40MB | Based on the text provided, here is a summary of the content |
| summarization | 1024 | 1021 | turbo3 | 3299.8 | 88.1 | 400 | 310ms | 3.8844 | 4.0809 | 0.0318 | 0.0062 | 1.40GB | 2.76GB | 5MB | 63MB | Here is a summary of the provided text excerpt from *The Gre |
| summarization | 2048 | 2044 | turbo3 | 3596.9 | 87.6 | 202 | 569ms | 4.4524 | 2.2596 | 0.0325 | 0.0741 | 1.40GB | 3.51GB | 24MB | 100MB | Based on the text provided, here is a summary of George Orwe |
| summarization | 4096 | 4087 | turbo3 | 4103.5 | 86.4 | 400 | 997ms | 5.4469 | 5.4601 | 0.0661 | 0.0431 | 1.40GB | 3.43GB | 42MB | 199MB | Based on the text provided, here is a summary of **Once Agai |
| summarization | 8192 | 8192 | turbo3 | 4364.0 | 81.2 | 400 | 1894ms | 4.3960 | 3.8023 | 0.0202 | 0.0535 | 1.40GB | 3.72GB | 92MB | 382MB | Based on the text provided, here is a summary of **Once Agai |
| summarization | 16384 | 16363 | turbo3 | 4103.0 | 78.3 | 400 | 4057ms | 4.5611 | 1.9230 | 0.0483 | 0.0417 | 1.40GB | 3.98GB | 207MB | 745MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 32768 | 32702 | turbo3 | 3448.8 | 68.5 | 400 | 9944ms | 5.1887 | 3.0545 | 0.0531 | 0.0296 | 1.40GB | 4.82GB | 399MB | 1.44GB | This is a comprehensive summary of the book *The Great Gatsb |
| summarization | 65536 | 65470 | turbo3 | 2188.0 | 47.4 | 400 | 31299ms | 3.7610 | 3.7645 | 0.0332 | 0.0251 | 1.40GB | 7.25GB | 783MB | 2.86GB | This text is a literary analysis and critical commentary on  |
| summarization | 131072 | 130775 | turbo3 | 1172.5 | 27.7 | 400 | 112106ms | 4.6786 | 3.6060 | 0.0405 | 0.0012 | 1.40GB | 8.29GB | 1.51GB | 5.69GB | Based on the text provided, here is a summary of the narrato |
