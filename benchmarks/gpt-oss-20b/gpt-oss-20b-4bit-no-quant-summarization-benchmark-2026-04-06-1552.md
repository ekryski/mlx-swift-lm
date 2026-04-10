# Inference Benchmark - GPT-OSS 20B

- **Date**: 2026-04-06 15:52
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
| summarization | 128 | 162 | no-quant | 355.7 | 60.9 | 200 | 478ms | — | 1.7941 | — | 0.3077 | 10.41GB | 10.95GB | 12MB | 79MB | <\|channel\|>analysis<\|message\|>We have a user message: "The G |
| summarization | 256 | 291 | no-quant | 468.2 | 59.6 | 200 | 622ms | — | 2.3856 | — | 0.3723 | 10.41GB | 11.32GB | 7MB | 107MB | <\|channel\|>analysis<\|message\|>We need to analyze user reques |
| summarization | 512 | 544 | no-quant | 544.3 | 59.7 | 200 | 1000ms | — | 2.1761 | — | 0.1119 | 10.41GB | 11.91GB | 30MB | 163MB | <\|channel\|>analysis<\|message\|>We have a user prompt: "The Gr |
| summarization | 1024 | 1053 | no-quant | 603.0 | 59.4 | 200 | 1844ms | — | 2.4394 | — | 0.3540 | 10.41GB | 12.51GB | 49MB | 274MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 2048 | 2061 | no-quant | 682.9 | 58.8 | 200 | 3021ms | — | 2.4623 | — | 0.2296 | 10.41GB | 12.40GB | 68MB | 495MB | <\|channel\|>analysis<\|message\|>We need to read the user's mes |
| summarization | 4096 | 4055 | no-quant | 652.0 | 55.3 | 200 | 6471ms | — | 2.6953 | — | 0.3418 | 10.41GB | 13.67GB | 132MB | 931MB | <\|channel\|>analysis<\|message\|>We have a user asking: "Summar |
| summarization | 8192 | 8042 | no-quant | 657.3 | 51.4 | 200 | 12415ms | — | 2.7293 | — | 0.2677 | 10.41GB | 13.71GB | 357MB | 1.76GB | <\|channel\|>analysis<\|message\|>We need to produce a summary.  |
| summarization | 16384 | 15955 | no-quant | 621.0 | 45.6 | 200 | 25910ms | — | 2.8886 | — | 0.2419 | 10.41GB | 13.69GB | 670MB | 3.45GB | <\|channel\|>analysis<\|message\|>We have a huge block of text:  |
| summarization | 32768 | 31717 | no-quant | 538.1 | 36.6 | 200 | 59220ms | — | 3.1033 | — | 0.3271 | 10.41GB | 13.70GB | 1.22GB | 6.82GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 65536 | 63299 | no-quant | 418.0 | 22.7 | 200 | 151750ms | — | 3.3075 | — | 0.4646 | 10.41GB | 15.84GB | 2.92GB | 13.56GB | <\|channel\|>analysis<\|message\|>We have a huge chunk of text:  |
| summarization | 131072 | 126728 | no-quant | 296.3 | 13.1 | 200 | 427974ms | — | 3.6558 | — | 0.3807 | 10.41GB | 18.67GB | 5.82GB | 27.11GB | <\|channel\|>analysis<\|message\|>We have a long text that seems |
