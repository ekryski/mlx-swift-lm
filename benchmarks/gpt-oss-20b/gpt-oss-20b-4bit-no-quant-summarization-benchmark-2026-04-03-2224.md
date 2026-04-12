# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-03 22:24
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
| summarization | 128 | 162 | no-quant | 360.1 | 59.8 | 200 | 451ms | — | 2.0896 | — | 0.2614 | 10.41GB | 10.90GB | 11MB | 79MB | <\|channel\|>analysis<\|message\|>We have a user asking: "The Gr |
| summarization | 256 | 291 | no-quant | 480.3 | 59.9 | 200 | 606ms | — | 1.4432 | — | 0.1415 | 10.41GB | 11.14GB | 3MB | 107MB | <\|channel\|>analysis<\|message\|>The user has provided a passag |
| summarization | 512 | 544 | no-quant | 549.7 | 60.0 | 200 | 990ms | — | 2.2868 | — | 0.2414 | 10.41GB | 11.67GB | 1MB | 163MB | <\|channel\|>analysis<\|message\|>The user has posted a piece of |
| summarization | 1024 | 1053 | no-quant | 609.4 | 59.1 | 200 | 1810ms | — | 2.6964 | — | 0.2339 | 10.41GB | 12.32GB | 38MB | 274MB | <\|channel\|>analysis<\|message\|>The user has provided a text:  |
| summarization | 2048 | 2061 | no-quant | 688.5 | 57.5 | 200 | 2996ms | — | 2.7161 | — | 0.2983 | 10.41GB | 12.21GB | 43MB | 495MB | <\|channel\|>analysis<\|message\|>We have a prompt: user gave a  |
| summarization | 4096 | 4055 | no-quant | 666.7 | 55.5 | 200 | 6358ms | — | 2.9679 | — | 0.3354 | 10.41GB | 13.41GB | 132MB | 931MB | <\|channel\|>analysis<\|message\|>We have a user-provided text:  |
| summarization | 8192 | 8042 | no-quant | 665.6 | 51.5 | 200 | 12334ms | — | 2.8695 | — | 0.3561 | 10.41GB | 13.46GB | 325MB | 1.76GB | <\|channel\|>analysis<\|message\|>We have a large piece of text, |
| summarization | 16384 | 15955 | no-quant | 628.7 | 45.6 | 200 | 25742ms | — | 2.9157 | — | 0.2973 | 10.41GB | 13.48GB | 702MB | 3.45GB | <\|channel\|>analysis<\|message\|>We have a very long text. It's |
| summarization | 32768 | 31717 | no-quant | 544.5 | 37.7 | 200 | 58677ms | — | 2.6613 | — | 0.3414 | 10.41GB | 13.54GB | 1.22GB | 6.82GB | <\|channel\|>analysis<\|message\|>We have a long user prompt: us |
| summarization | 65536 | 63299 | no-quant | 425.6 | 24.5 | 200 | 149252ms | — | 2.8630 | — | 0.2998 | 10.41GB | 15.76GB | 373MB | 13.56GB | <\|channel\|>analysis<\|message\|>The user has pasted a huge blo |
| summarization | 131072 | 126728 | no-quant | 296.3 | 14.6 | 200 | 428182ms | — | 2.7961 | — | 0.4853 | 10.41GB | 18.58GB | 3.39GB | 27.11GB | <\|channel\|>analysis<\|message\|>We have a huge text: 38 sectio |
