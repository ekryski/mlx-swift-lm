# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-06 17:09
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
| summarization | 128 | 162 | turbo3v2 | 434.2 | 72.2 | 200 | 402ms | — | — | — | — | 10.41GB | 10.95GB | 4MB | 14MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 256 | 291 | turbo3v2 | 516.2 | 70.8 | 200 | 569ms | — | — | — | — | 10.41GB | 11.32GB | 16MB | 18MB | <\|channel\|>analysis<\|message\|>The user has posted a text: "T |
| summarization | 512 | 544 | turbo3v2 | 582.1 | 68.9 | 200 | 946ms | — | — | — | — | 10.41GB | 11.91GB | 10MB | 28MB | <\|channel\|>analysis<\|message\|>We have a user query. The user |
| summarization | 1024 | 1053 | turbo3v2 | 646.6 | 70.5 | 200 | 1690ms | — | — | — | — | 10.41GB | 12.51GB | 43MB | 47MB | <\|channel\|>analysis<\|message\|>We need to produce a summary o |
| summarization | 2048 | 2061 | turbo3v2 | 704.8 | 68.2 | 200 | 2940ms | — | — | — | — | 10.41GB | 12.40GB | 85MB | 85MB | <\|channel\|>analysis<\|message\|>We have a user who posted a te |
| summarization | 4096 | 4055 | turbo3v2 | 678.3 | 64.9 | 200 | 6285ms | — | — | — | — | 10.41GB | 13.67GB | 115MB | 160MB | <\|channel\|>analysis<\|message\|>We have a user request: "Summa |
| summarization | 8192 | 8042 | turbo3v2 | 653.0 | 62.0 | 200 | 12564ms | — | — | — | — | 10.41GB | 13.71GB | 227MB | 310MB | <\|channel\|>analysis<\|message\|>We need to summarize content.  |
| summarization | 16384 | 15955 | turbo3v2 | 625.1 | 53.0 | 200 | 25792ms | — | — | — | — | 10.41GB | 13.69GB | 574MB | 607MB | <\|channel\|>analysis<\|message\|>We have a long input. The user |
| summarization | 32768 | 31717 | turbo3v2 | 553.7 | 44.0 | 200 | 57588ms | — | — | — | — | 10.41GB | 13.78GB | 1.10GB | 1.17GB | <\|channel\|>analysis<\|message\|>The user has pasted a huge chu |
| summarization | 65536 | 63299 | turbo3v2 | 420.0 | 25.8 | 200 | 153138ms | — | — | — | — | 10.41GB | 15.84GB | 2.79GB | 2.33GB | <\|channel\|>analysis<\|message\|>We have a very long text. It's |
| summarization | 131072 | 126728 | turbo3v2 | 306.3 | 15.0 | 200 | 414002ms | — | — | — | — | 10.41GB | 18.99GB | 5.33GB | 4.66GB | <\|channel\|>analysis<\|message\|>We have a huge block of text.  |
