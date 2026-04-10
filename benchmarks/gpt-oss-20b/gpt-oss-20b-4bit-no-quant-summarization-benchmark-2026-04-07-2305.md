# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-07 23:05
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
| summarization | 1024 | 1024 | no-quant | 666.7 | 69.0 | 200 | 1657ms | — | 3.9480 | — | — | 10.41GB | 12.44GB | 45MB | 268MB | <\|channel\|>analysis<\|message\|>We have a user who has posted  |
| summarization | 4096 | 4055 | no-quant | 685.7 | 65.2 | 200 | 6164ms | — | 2.3349 | — | — | 10.41GB | 13.67GB | 149MB | 931MB | <\|channel\|>analysis<\|message\|>We need to analyze the task: T |
| summarization | 16384 | 15955 | no-quant | 629.9 | 53.5 | 200 | 25609ms | — | 2.7071 | — | — | 10.41GB | 13.69GB | 574MB | 3.45GB | <\|channel\|>analysis<\|message\|>We have a huge text: it's a no |
