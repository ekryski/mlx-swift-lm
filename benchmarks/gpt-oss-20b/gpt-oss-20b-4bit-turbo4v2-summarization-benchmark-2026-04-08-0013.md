# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-08 00:13
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
| summarization | 1024 | 1024 | turbo4v2 | 635.8 | 68.0 | 200 | 1739ms | — | 2.7467 | — | — | 10.41GB | 12.44GB | 50MB | 54MB | <\|channel\|>analysis<\|message\|>We need to analyze user query. |
| summarization | 4096 | 4055 | turbo4v2 | 667.0 | 64.3 | 200 | 6306ms | — | 3.2528 | — | — | 10.41GB | 13.67GB | 198MB | 189MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | turbo4v2 | 630.6 | 53.4 | 200 | 25535ms | — | 2.5043 | — | — | 10.41GB | 13.69GB | 766MB | 718MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
