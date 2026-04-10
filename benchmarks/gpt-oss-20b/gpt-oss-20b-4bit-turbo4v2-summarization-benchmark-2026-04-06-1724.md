# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-06 17:24
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
| summarization | 128 | 162 | turbo4v2 | 422.4 | 70.7 | 200 | 413ms | — | — | — | — | 10.41GB | 10.95GB | 0MB | 16MB | <\|channel\|>analysis<\|message\|>The user has provided a text:  |
| summarization | 256 | 291 | turbo4v2 | 508.5 | 70.5 | 200 | 579ms | — | — | — | — | 10.41GB | 11.32GB | 2MB | 22MB | <\|channel\|>analysis<\|message\|>We have a user-provided conten |
| summarization | 512 | 544 | turbo4v2 | 583.6 | 69.9 | 200 | 943ms | — | — | — | — | 10.41GB | 11.91GB | 20MB | 33MB | <\|channel\|>analysis<\|message\|>We need to respond with a summ |
| summarization | 1024 | 1053 | turbo4v2 | 671.2 | 69.3 | 200 | 1719ms | — | — | — | — | 10.41GB | 12.51GB | 41MB | 56MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 2048 | 2061 | turbo4v2 | 692.4 | 67.6 | 200 | 2991ms | — | — | — | — | 10.41GB | 12.40GB | 102MB | 100MB | <\|channel\|>analysis<\|message\|>The user provided a large chun |
| summarization | 4096 | 4055 | turbo4v2 | 666.4 | 63.4 | 200 | 6352ms | — | — | — | — | 10.41GB | 13.67GB | 116MB | 189MB | <\|channel\|>analysis<\|message\|>We need to produce a summary o |
| summarization | 8192 | 8042 | turbo4v2 | 656.2 | 60.3 | 200 | 12534ms | — | — | — | — | 10.41GB | 13.71GB | 292MB | 366MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | turbo4v2 | 614.1 | 54.0 | 200 | 26263ms | — | — | — | — | 10.41GB | 13.69GB | 766MB | 718MB | <\|channel\|>analysis<\|message\|>We have a long text. The user  |
| summarization | 32768 | 31717 | turbo4v2 | 526.8 | 43.9 | 200 | 60526ms | — | — | — | — | 10.41GB | 13.78GB | 815MB | 1.38GB | <\|channel\|>analysis<\|message\|>We have a huge text: It's a lo |
| summarization | 65536 | 63299 | turbo4v2 | 423.9 | 26.0 | 200 | 149641ms | — | — | — | — | 10.41GB | 15.91GB | 2.79GB | 2.76GB | <\|channel\|>analysis<\|message\|>We have a very long text. The  |
| summarization | 131072 | 126728 | turbo4v2 | 292.4 | 14.5 | 200 | 433793ms | — | — | — | — | 10.41GB | 18.99GB | 5.33GB | 5.51GB | <\|channel\|>analysis<\|message\|>We have a huge block of text.  |
