# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-03 23:44
- **Branch**: `ek/consolidated-benchmarks`
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
| summarization | 128 | 162 | turbo3 | 378.7 | 60.5 | 200 | 429ms | — | 2.7665 | — | 0.5161 | 10.41GB | 10.90GB | 9MB | 16MB | <\|channel\|>analysis<\|message\|>The user asks: "Summarize the  |
| summarization | 256 | 291 | turbo3 | 498.3 | 60.3 | 200 | 584ms | — | 1.6345 | — | 0.1657 | 10.41GB | 11.14GB | 3MB | 22MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 512 | 544 | turbo3 | 582.2 | 60.0 | 200 | 935ms | — | 2.5280 | — | 0.2683 | 10.41GB | 11.67GB | 28MB | 33MB | <\|channel\|>analysis<\|message\|>We have a user asking: "Summar |
| summarization | 1024 | 1053 | turbo3 | 610.4 | 59.5 | 200 | 1997ms | — | 2.5667 | — | 0.4823 | 10.41GB | 12.32GB | 0MB | 56MB | <\|channel\|>analysis<\|message\|>We need to read the provided t |
| summarization | 2048 | 2061 | turbo3 | 700.7 | 58.3 | 200 | 2944ms | — | 3.0536 | — | 0.1994 | 10.41GB | 12.21GB | 0MB | 100MB | <\|channel\|>analysis<\|message\|>We have a user who has posted  |
| summarization | 4096 | 4055 | turbo3 | 670.6 | 56.0 | 200 | 6335ms | — | 3.2787 | — | 0.4224 | 10.41GB | 13.41GB | 140MB | 189MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | turbo3 | 672.7 | 52.6 | 200 | 12268ms | — | 3.9140 | — | 0.6531 | 10.41GB | 13.46GB | 0MB | 366MB | <\|channel\|>analysis<\|message\|>We need to summarise the conte |
| summarization | 16384 | 15955 | turbo3 | 636.5 | 45.8 | 200 | 25580ms | — | 2.7025 | — | 0.2747 | 10.41GB | 13.48GB | 0MB | 718MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 32768 | 31717 | turbo3 | 556.7 | 37.5 | 200 | 57406ms | — | 2.5903 | — | 0.3588 | 10.41GB | 13.54GB | 1.22GB | 1.38GB | <\|channel\|>analysis<\|message\|>We have a huge text. The user  |
| summarization | 65536 | 63299 | turbo3 | 438.0 | 24.4 | 200 | 145015ms | — | 3.3435 | — | 0.4095 | 10.41GB | 15.76GB | 1.46GB | 2.76GB | <\|channel\|>analysis<\|message\|>The user has provided a huge b |
| summarization | 131072 | 126728 | turbo3 | 309.3 | 14.4 | 200 | 410222ms | — | 2.6031 | — | 0.4209 | 10.41GB | 18.58GB | 2.91GB | 5.51GB | <\|channel\|>analysis<\|message\|>We have a very long text that  |
