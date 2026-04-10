# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-03 21:24
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: 4bit
- **Model**: `loan-star/gpt-oss-20b-mlx-4Bit`

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
| Temperature | 0.8 |
| Top P | 0.8 |
| Top K | 0 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Reasoning Effort | medium |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 162 | no-quant | 364.2 | 60.5 | 200 | 446ms | — | 2.4805 | — | 0.2862 | 10.41GB | 10.90GB | 6MB | 79MB | <\|channel\|>analysis<\|message\|>The user provided some text: l |
| summarization | 256 | 291 | no-quant | 472.7 | 60.4 | 200 | 616ms | — | 2.2817 | — | 0.3709 | 10.41GB | 11.14GB | 13MB | 107MB | <\|channel\|>analysis<\|message\|>The user asks: "Summarize the  |
| summarization | 512 | 544 | no-quant | 550.9 | 60.2 | 200 | 988ms | — | 2.3282 | — | 0.1923 | 10.41GB | 11.67GB | 1MB | 163MB | <\|channel\|>analysis<\|message\|>The user has provided a text e |
| summarization | 1024 | 1053 | no-quant | 605.8 | 59.4 | 200 | 1836ms | — | 2.7531 | — | 0.3251 | 10.41GB | 12.32GB | 0MB | 274MB | <\|channel\|>analysis<\|message\|>The user has provided a piece  |
| summarization | 2048 | 2061 | no-quant | 683.5 | 57.5 | 200 | 3019ms | — | 2.5588 | — | 0.3672 | 10.41GB | 12.21GB | 102MB | 495MB | <\|channel\|>analysis<\|message\|>We have a user query: "Summari |
| summarization | 4096 | 4055 | no-quant | 662.5 | 55.7 | 200 | 6385ms | — | 2.7743 | — | 0.3961 | 10.41GB | 13.41GB | 181MB | 931MB | <\|channel\|>analysis<\|message\|>We have to produce a summary o |
| summarization | 8192 | 8042 | no-quant | 661.7 | 51.8 | 200 | 12427ms | — | 3.6400 | — | 0.3780 | 10.41GB | 13.46GB | 32MB | 1.76GB | <\|channel\|>analysis<\|message\|>We need to summarize content.  |
| summarization | 16384 | 15955 | no-quant | 627.7 | 45.4 | 200 | 25974ms | — | 3.4960 | — | 0.3957 | 10.41GB | 13.48GB | 511MB | 3.45GB | <\|channel\|>analysis<\|message\|>We have a huge block of text.  |
| summarization | 32768 | 31717 | no-quant | 547.5 | 37.3 | 200 | 58312ms | — | 3.1508 | — | 0.3352 | 10.41GB | 13.54GB | 1.22GB | 6.82GB | <\|channel\|>analysis<\|message\|>We have a huge block of text t |
| summarization | 65536 | 63299 | no-quant | 429.6 | 24.3 | 200 | 147832ms | — | 3.3442 | — | 0.3331 | 10.41GB | 15.76GB | 0MB | 13.56GB | <\|channel\|>analysis<\|message\|>We have a huge block of text,  |
| summarization | 131072 | 126728 | no-quant | 300.4 | 14.4 | 200 | 422295ms | — | 2.7630 | — | 0.4666 | 10.41GB | 18.58GB | 2.18GB | 27.11GB | <\|channel\|>analysis<\|message\|>We have a huge text. The user  |
