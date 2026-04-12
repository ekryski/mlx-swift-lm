# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-08 00:33
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
| summarization | 1024 | 1024 | turbo4v2 | 641.5 | 68.6 | 179 | 1677ms | — | 2.2666 | — | — | 10.41GB | 12.44GB | 54MB | 53MB | <\|channel\|>analysis<\|message\|>The user provided a passage fr |
| summarization | 4096 | 4055 | turbo4v2 | 688.0 | 65.5 | 200 | 6173ms | — | 2.5191 | — | — | 10.41GB | 13.67GB | 181MB | 189MB | <\|channel\|>analysis<\|message\|>We have a user query: "Summari |
| summarization | 16384 | 15955 | turbo4v2 | 629.3 | 53.5 | 200 | 25613ms | — | 2.5786 | — | — | 10.41GB | 13.69GB | 766MB | 718MB | <\|channel\|>analysis<\|message\|>We have to summarize the conte |
