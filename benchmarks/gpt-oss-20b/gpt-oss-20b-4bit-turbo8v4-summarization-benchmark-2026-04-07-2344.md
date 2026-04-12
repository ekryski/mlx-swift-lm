# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-07 23:44
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
| summarization | 1024 | 1024 | turbo8v4 | 657.5 | 68.8 | 200 | 1658ms | — | 3.1467 | — | — | 10.41GB | 12.44GB | 49MB | 105MB | <\|channel\|>analysis<\|message\|>We have a prompt: It's a user  |
| summarization | 4096 | 4055 | turbo8v4 | 680.1 | 64.9 | 200 | 6173ms | — | 2.6820 | — | — | 10.41GB | 13.67GB | 132MB | 364MB | <\|channel\|>analysis<\|message\|>We have a user message: It's a |
| summarization | 16384 | 15955 | turbo8v4 | 630.5 | 52.9 | 200 | 25581ms | — | 2.9307 | — | — | 10.41GB | 13.69GB | 638MB | 1.35GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
