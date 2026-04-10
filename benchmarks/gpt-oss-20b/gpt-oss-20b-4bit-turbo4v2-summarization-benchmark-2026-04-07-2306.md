# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-07 23:06
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `loan-star/gpt-oss-20b-mlx-4Bit`

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
| summarization | 1024 | 1024 | turbo4v2 | 641.6 | 67.6 | 176 | 1735ms | — | 2.4911 | — | — | 10.41GB | 12.44GB | 54MB | 53MB | <\|channel\|>analysis<\|message\|>We have a user message: It's a |
| summarization | 4096 | 4055 | turbo4v2 | 678.1 | 64.9 | 200 | 6174ms | — | 2.4535 | — | — | 10.41GB | 13.67GB | 198MB | 189MB | <\|channel\|>analysis<\|message\|>We have a user request: "Summa |
| summarization | 16384 | 15955 | turbo4v2 | 628.8 | 53.1 | 200 | 25611ms | — | 3.1108 | — | — | 10.41GB | 13.69GB | 575MB | 718MB | <\|channel\|>analysis<\|message\|>We have a user message that is |
