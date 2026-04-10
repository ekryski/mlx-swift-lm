# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-03 20:56
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `sjgdr/gpt-oss-20b-mlx-fp16`

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
| summarization | 128 | 162 | no-quant | 269.5 | 21.1 | 200 | 604ms | — | 2.3980 | — | — | 12.82GB | 15.66GB | 5MB | 79MB | <\|channel\|>analysis<\|message\|>The user provided some text. I |
| summarization | 256 | 291 | no-quant | 438.9 | 21.1 | 200 | 663ms | — | 1.8548 | — | — | 12.82GB | 15.82GB | 15MB | 107MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 512 | 544 | no-quant | 536.7 | 20.5 | 200 | 1014ms | — | 2.2424 | — | — | 12.82GB | 16.14GB | 20MB | 163MB | <\|channel\|>analysis<\|message\|>The user has provided a block  |
| summarization | 1024 | 1053 | no-quant | 591.5 | 20.5 | 200 | 1920ms | — | 2.6582 | — | — | 12.82GB | 16.85GB | 9MB | 274MB | <\|channel\|>analysis<\|message\|>The user provided a long passa |
| summarization | 2048 | 2061 | no-quant | 664.1 | 20.3 | 200 | 3106ms | — | 2.5479 | — | — | 12.82GB | 15.64GB | 89MB | 495MB | <\|channel\|>analysis<\|message\|>The user provided a long excer |
| summarization | 4096 | 4055 | no-quant | 675.7 | 20.1 | 200 | 6274ms | — | 2.7231 | — | — | 12.82GB | 17.89GB | 198MB | 931MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | no-quant | 678.5 | 19.8 | 200 | 12140ms | — | 2.4621 | — | — | 12.82GB | 17.95GB | 325MB | 1.76GB | <\|channel\|>analysis<\|message\|>The user provided a huge block |
| summarization | 16384 | 15955 | no-quant | 639.1 | 19.2 | 200 | 25251ms | — | 2.4476 | — | — | 12.82GB | 17.82GB | 191MB | 3.45GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 32768 | 31717 | no-quant | 555.8 | 18.1 | 200 | 57483ms | — | 2.2273 | — | — | 12.82GB | 18.14GB | 1.47GB | 6.82GB | <\|channel\|>analysis<\|message\|>The user provided a huge chunk |
| summarization | 65536 | 63299 | no-quant | 429.4 | 11.9 | 200 | 149567ms | — | 2.6917 | — | — | 12.82GB | 20.29GB | 2.91GB | 13.56GB | <\|channel\|>analysis<\|message\|>The user posted a huge text th |
| summarization | 131072 | 126728 | no-quant | 297.5 | 10.2 | 200 | 426532ms | — | 2.5661 | — | — | 12.82GB | 23.12GB | 4.61GB | 27.11GB | <\|channel\|>analysis<\|message\|>The user provided a massive te |
