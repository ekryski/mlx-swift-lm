# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-07 00:21
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
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | no-quant | 477.2 | 67.2 | 400 | 245ms | 1.3071 | — | 2.2731 | — | 2.45GB | 2.88GB | 7MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 781.6 | 64.9 | 400 | 328ms | 1.3385 | — | 1.3254 | — | 2.45GB | 3.26GB | 13MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 986.3 | 64.4 | 400 | 510ms | 1.2653 | — | 0.7396 | — | 2.45GB | 3.96GB | 0MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 1089.0 | 60.6 | 400 | 932ms | 1.5580 | — | 0.9541 | — | 2.45GB | 4.78GB | 0MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 1015.5 | 60.5 | 400 | 2045ms | 1.6491 | — | 0.8813 | — | 2.45GB | 5.82GB | 0MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1544.4 | 57.5 | 400 | 2660ms | 1.4725 | — | 0.6955 | — | 2.45GB | 5.54GB | 59MB | 983MB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 8192 | 8229 | no-quant | 3671.5 | 57.9 | 111 | 2262ms | — | 1.7501 | 3.4544 | — | 2.45GB | 4.50GB | 16MB | 1.78GB | The provided text is an excerpt from a work, likely a novel, |
| summarization | 16384 | 16395 | no-quant | 3537.9 | 51.8 | 350 | 4669ms | — | 2.0885 | 1.2545 | — | 2.45GB | 5.14GB | 204MB | 3.58GB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 32768 | 32815 | no-quant | 3029.2 | 43.0 | 400 | 10880ms | — | 2.2172 | 0.9745 | — | 2.45GB | 6.35GB | 264MB | 7.10GB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 65536 | 65896 | no-quant | 1891.9 | 32.5 | 400 | 34833ms | — | 2.1289 | 0.9414 | — | 2.45GB | 8.74GB | 777MB | 14.16GB | The provided text is a collection of excerpts, likely from o |
| summarization | 131072 | 130563 | no-quant | 1072.8 | 30.9 | 400 | 121767ms | — | 2.2548 | 0.5706 | — | 2.45GB | 13.70GB | 1.51GB | 27.98GB | This is a complex collection of excerpts from a literary wor |
