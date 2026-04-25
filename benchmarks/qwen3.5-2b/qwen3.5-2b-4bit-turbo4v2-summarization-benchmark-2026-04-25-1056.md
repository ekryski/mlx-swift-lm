# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-25 10:56
**Branch**: `ek/tom-eric-moe-tuning`
**Commit**: `82bada9 perf: optimized bridge v2 — 49.7ms at 1024 (20.6K tok/s)`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-2B-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 108GB |
| macOS | 26.4.1 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |
| Thinking | No |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Speculative decoding | none |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | default |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 109 | turbo4v2 | 3493.9 | 297.6 | 158 | 31ms | — | — | — | — | 1010MB | 1.27GB | 12MB | 12MB | Here is a summary of the provided text:  The excerpt present |
| summarization | 1024 | 1011 | turbo4v2 | 11400.4 | 292.6 | 200 | 89ms | — | — | — | — | 1010MB | 2.04GB | 22MB | 54MB | The provided text is a **summary of the Table of Contents an |
| summarization | 4096 | 4077 | turbo4v2 | 12052.1 | 276.0 | 200 | 339ms | — | — | — | — | 1010MB | 2.69GB | 51MB | 190MB | This excerpt from **"The Great Gatsby" by F. Scott Fitzgeral |
| summarization | 32768 | 32692 | turbo4v2 | 7480.3 | 212.0 | 200 | 4371ms | — | — | — | — | 1010MB | 3.12GB | 330MB | 1.43GB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
