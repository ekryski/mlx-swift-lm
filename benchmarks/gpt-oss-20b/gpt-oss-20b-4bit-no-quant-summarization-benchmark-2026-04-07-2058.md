# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-07 20:58
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
| summarization | 1024 | 1024 | no-quant | 652.2 | 68.5 | 200 | 1658ms | — | 2.0578 | — | — | 10.41GB | 12.44GB | 49MB | 268MB | <\|channel\|>analysis<\|message\|>We need to continue? The user  |
| summarization | 4096 | 4055 | no-quant | 678.9 | 65.1 | 200 | 6174ms | — | 3.2059 | — | — | 10.41GB | 13.67GB | 198MB | 931MB | <\|channel\|>analysis<\|message\|>We need to summarize content.  |
