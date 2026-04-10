# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-07 22:08
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
| summarization | 1024 | 1024 | no-quant | 620.6 | 68.0 | 200 | 1721ms | — | 3.2965 | — | — | 10.41GB | 12.44GB | 54MB | 268MB | <\|channel\|>analysis<\|message\|>The user has posted a long tex |
| summarization | 4096 | 4055 | no-quant | 684.2 | 65.6 | 200 | 6156ms | — | 3.0192 | — | — | 10.41GB | 13.67GB | 107MB | 931MB | <\|channel\|>analysis<\|message\|>We have a user who provided a  |
