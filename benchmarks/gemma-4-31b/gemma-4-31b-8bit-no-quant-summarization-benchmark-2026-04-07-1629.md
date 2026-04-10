# Inference Benchmark - Gemma 4 31B

**Date**: 2026-04-07 16:29
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 8bit
**Model**: `mlx-community/gemma-4-31b-it-8bit`

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
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | no-quant | 43.2 | 8.7 | 315 | 3173ms | 1.2209 | 1.0231 | — | — | 30.38GB | 31.31GB | 451MB | 94MB | <\|channel>thought  *   Input text: A snippet from the beginn |
| summarization | 1024 | 1014 | no-quant | 70.0 | 8.5 | 400 | 15288ms | 1.2047 | — | — | — | 30.38GB | 33.14GB | 832MB | 309MB | <\|channel>thought  *   Source material: An excerpt from the  |
| summarization | 4096 | 4094 | no-quant | 70.6 | 8.7 | 400 | 58835ms | 1.2643 | — | — | — | 30.38GB | 36.91GB | 1.08GB | 983MB | <\|channel>thought The provided text is the beginning of Chap |
| summarization | 32768 | 32815 | no-quant | 66.9 | 7.1 | 400 | 490925ms | 1.0000 | 1.1527 | — | — | 30.38GB | 40.17GB | 2.90GB | 7.10GB | <\|channel>thought <channel\|>The provided text comprises the  |
