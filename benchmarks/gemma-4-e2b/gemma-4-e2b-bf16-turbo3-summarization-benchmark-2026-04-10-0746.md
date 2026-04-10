# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-10 07:46
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: bf16
**Model**: `mlx-community/gemma-4-e2b-it-bf16`

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
| summarization | 128 | 116 | turbo3 | 804.1 | 45.2 | 293 | 145ms | 1.4404 | — | 1.3614 | — | 8.66GB | 8.82GB | 7MB | 18MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | turbo3 | 713.6 | 44.4 | 400 | 359ms | 1.4800 | 1.9199 | 0.4056 | 1.6262 | 8.66GB | 8.96GB | 4MB | 29MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | turbo3 | 1283.5 | 43.4 | 400 | 393ms | 1.2047 | — | 0.3606 | — | 8.66GB | 9.05GB | 11MB | 40MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | turbo3 | 1829.5 | 44.3 | 400 | 556ms | 1.2215 | — | 0.2622 | — | 8.66GB | 9.21GB | 14MB | 63MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | turbo3 | 1954.9 | 44.1 | 400 | 1043ms | 1.4024 | — | 0.2117 | — | 8.66GB | 9.25GB | 22MB | 108MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | turbo3 | 1537.5 | 43.7 | 400 | 2664ms | 1.3226 | — | 0.2407 | — | 8.66GB | 9.29GB | 38MB | 200MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | turbo3 | 1199.3 | 42.9 | 400 | 6832ms | 1.3514 | — | 0.4551 | — | 8.66GB | 9.33GB | 32MB | 382MB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 16384 | 16384 | turbo3 | 756.5 | 41.0 | 400 | 21659ms | 1.3757 | — | 0.6322 | — | 8.66GB | 9.37GB | 134MB | 746MB | <\|channel>thought Here's a thinking process to analyze the r |
| summarization | 32768 | 32768 | turbo3 | 429.6 | 38.4 | 400 | 76280ms | 1.6726 | — | 0.5212 | — | 8.66GB | 9.46GB | 262MB | 1.44GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 65536 | 65536 | turbo3 | 231.0 | 32.3 | 400 | 283752ms | 1.5544 | — | 0.3857 | — | 8.66GB | 9.65GB | 518MB | 2.86GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | turbo3 | 119.1 | 27.5 | 400 | 1096388ms | 1.5318 | — | 0.0033 | — | 8.66GB | 10.20GB | 516MB | 5.68GB | <\|channel>thought Here's a thinking process to arrive at the |
