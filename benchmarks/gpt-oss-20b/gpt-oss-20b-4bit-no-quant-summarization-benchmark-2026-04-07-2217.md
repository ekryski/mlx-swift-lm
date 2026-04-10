# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-07 22:17
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
| summarization | 1024 | 1024 | no-quant | 468.8 | 73.9 | 200 | 2359ms | — | 3.3303 | — | — | 10.41GB | 11.41GB | 27MB | 268MB | <\|channel\|>analysis<\|message\|>We have a user who posted a te |
| summarization | 4096 | 4055 | no-quant | 488.4 | 68.9 | 200 | 8544ms | — | 3.1242 | — | — | 10.41GB | 12.07GB | 91MB | 931MB | <\|channel\|>analysis<\|message\|>We have a user who provided a  |
