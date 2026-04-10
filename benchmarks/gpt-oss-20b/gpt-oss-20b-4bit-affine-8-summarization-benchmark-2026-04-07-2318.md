# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-07 23:18
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
| summarization | 1024 | 1024 | affine-8 | 651.8 | 69.2 | 200 | 1659ms | — | 2.8437 | — | — | 10.41GB | 12.44GB | 54MB | 151MB | <\|channel\|>analysis<\|message\|>The user has pasted a text tha |
| summarization | 4096 | 4055 | affine-8 | 665.2 | 65.3 | 200 | 6310ms | — | 3.3559 | — | — | 10.41GB | 13.67GB | 181MB | 524MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | affine-8 | 627.5 | 53.8 | 200 | 25732ms | — | 3.1670 | — | — | 10.41GB | 13.69GB | 766MB | 1.94GB | <\|channel\|>analysis<\|message\|>We have a huge chunk of text,  |
