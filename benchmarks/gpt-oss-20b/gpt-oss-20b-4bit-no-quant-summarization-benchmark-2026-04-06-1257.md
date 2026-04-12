# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-06 12:57
- **Branch**: `ek/tom-eric-moe-tuning`
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
| summarization | 128 | 162 | no-quant | 399.4 | 69.6 | 200 | 444ms | — | — | — | — | 10.41GB | 10.95GB | 11MB | 79MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 1024 | 1053 | no-quant | 648.9 | 70.2 | 200 | 1702ms | — | — | — | — | 10.41GB | 12.51GB | 54MB | 274MB | <\|channel\|>analysis<\|message\|>The user has provided a text:  |
| summarization | 4096 | 4055 | no-quant | 683.5 | 66.1 | 200 | 6137ms | — | — | — | — | 10.41GB | 13.67GB | 132MB | 931MB | <\|channel\|>analysis<\|message\|>We have to produce a summary o |
| summarization | 32768 | 31717 | no-quant | 547.6 | 44.3 | 200 | 58245ms | — | — | — | — | 10.41GB | 13.78GB | 1.29GB | 6.82GB | <\|channel\|>analysis<\|message\|>We have a long user-provided t |
