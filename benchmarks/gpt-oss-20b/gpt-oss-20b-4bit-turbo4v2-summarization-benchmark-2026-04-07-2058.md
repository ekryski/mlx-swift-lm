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
| summarization | 1024 | 1024 | turbo4v2 | 646.5 | 68.4 | 200 | 1667ms | — | 3.0215 | — | — | 10.41GB | 12.44GB | 49MB | 54MB | <\|channel\|>analysis<\|message\|>We have a user message contain |
| summarization | 4096 | 4055 | turbo4v2 | 671.8 | 64.2 | 200 | 6288ms | — | 3.2155 | — | — | 10.41GB | 13.67GB | 198MB | 189MB | <\|channel\|>analysis<\|message\|>We have a long text. The user  |
