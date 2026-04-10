# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-07 23:58
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
| summarization | 1024 | 1024 | turbo8v2 | 633.7 | 68.2 | 200 | 1740ms | — | 2.6289 | — | — | 10.41GB | 12.44GB | 54MB | 88MB | <\|channel\|>analysis<\|message\|>We need to respond to user. Th |
| summarization | 4096 | 4055 | turbo8v2 | 683.9 | 65.1 | 200 | 6154ms | — | 3.3556 | — | — | 10.41GB | 13.67GB | 181MB | 305MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | turbo8v2 | 626.7 | 53.4 | 200 | 25708ms | — | 2.5867 | — | — | 10.41GB | 13.69GB | 575MB | 1.13GB | <\|channel\|>analysis<\|message\|>We have a long text: It's a no |
