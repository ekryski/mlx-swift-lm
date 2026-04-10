# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-07 22:19
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
| summarization | 1024 | 1024 | turbo4v2 | 626.6 | 68.2 | 200 | 1744ms | — | 2.4437 | — | — | 10.41GB | 12.44GB | 54MB | 54MB | <\|channel\|>analysis<\|message\|>We need to produce an answer.  |
| summarization | 4096 | 4055 | turbo4v2 | 668.7 | 64.2 | 200 | 6318ms | — | 2.5366 | — | — | 10.41GB | 13.67GB | 173MB | 189MB | <\|channel\|>analysis<\|message\|>We have a user request: "Summa |
