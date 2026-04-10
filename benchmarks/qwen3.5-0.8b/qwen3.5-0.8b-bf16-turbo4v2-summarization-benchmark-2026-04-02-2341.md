# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-02 23:41
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
| summarization | 128 | 119 | turbo4v2 | 1064.3 | 86.9 | 400 | 113ms | 6.0231 | 2.9143 | 0.0546 | 0.0214 | 1.40GB | 1.61GB | 16MB | 23MB | Based on the text you provided, here is a summary of the *Ze |
| summarization | 256 | 251 | turbo4v2 | 2027.1 | 84.8 | 337 | 124ms | 3.9699 | 4.4041 | 0.0530 | 0.0489 | 1.40GB | 1.82GB | 18MB | 26MB | The content you provided is a short collection of excerpts f |
| summarization | 512 | 506 | turbo4v2 | 2560.0 | 84.2 | 400 | 198ms | 3.6636 | 4.0941 | 0.0580 | 0.0259 | 1.40GB | 2.24GB | 21MB | 40MB | Based on the text provided, here is a summary of the content |
| summarization | 1024 | 1021 | turbo4v2 | 3245.5 | 87.4 | 400 | 315ms | 3.3014 | 3.6668 | 0.0373 | 0.0521 | 1.40GB | 2.76GB | 27MB | 63MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 2048 | 2044 | turbo4v2 | 3541.6 | 86.2 | 400 | 578ms | 3.8195 | 4.5406 | 0.0407 | 0.0345 | 1.40GB | 3.51GB | 40MB | 109MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 4096 | 4087 | turbo4v2 | 4090.2 | 86.1 | 201 | 1000ms | 4.5608 | 2.0624 | 0.0098 | 0.0030 | 1.40GB | 3.43GB | 50MB | 191MB | Here is a summary of the provided text:  **Context and Theme |
| summarization | 8192 | 8192 | turbo4v2 | 4319.8 | 82.5 | 399 | 1902ms | 4.0203 | 2.9953 | 0.0149 | 0.0190 | 1.40GB | 3.72GB | 112MB | 382MB | Based on the provided text from "The Great Gatsby" by F. Sco |
| summarization | 16384 | 16363 | turbo4v2 | 4084.2 | 78.0 | 400 | 4077ms | 4.4578 | 4.3905 | 0.0417 | 0.0482 | 1.40GB | 3.98GB | 207MB | 745MB | Based on the text provided, here is a summary of the chapter |
| summarization | 32768 | 32702 | turbo4v2 | 3447.7 | 68.1 | 400 | 9859ms | 4.4763 | 1.9756 | -0.0151 | -0.0194 | 1.40GB | 4.82GB | 399MB | 1.44GB | This text is a **free-verse commentary on *The Great Gatsby* |
| summarization | 65536 | 65470 | turbo4v2 | 1886.0 | 50.2 | 400 | 36232ms | 3.5435 | 3.5969 | 0.0032 | 0.0549 | 1.40GB | 7.25GB | 783MB | 2.86GB | This text is a collection of excerpts from the novel **_The  |
| summarization | 131072 | 130775 | turbo4v2 | 1230.2 | 30.6 | 400 | 107347ms | 4.0354 | 4.4836 | 0.0401 | 0.0682 | 1.40GB | 8.29GB | 1.51GB | 5.69GB | Based on the text provided, this is a novel excerpt detailin |
