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
| summarization | 128 | 109 | no-quant | 3548.7 | 299.7 | 200 | 31ms | — | — | — | — | 1010MB | 1.27GB | 14MB | 68MB | The text you provided is a selection from F. Scott Fitzgeral |
| summarization | 1024 | 1011 | no-quant | 11464.9 | 292.9 | 200 | 88ms | — | — | — | — | 1010MB | 2.04GB | 19MB | 265MB | This text excerpt from *The Great Gatsby* by F. Scott Fitzge |
| summarization | 4096 | 4077 | no-quant | 12143.7 | 276.2 | 200 | 336ms | — | — | — | — | 1010MB | 2.69GB | 60MB | 936MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* b |
| summarization | 32768 | 32692 | no-quant | 7482.1 | 212.0 | 200 | 4370ms | — | — | — | — | 1010MB | 3.12GB | 265MB | 7.03GB | This text is a condensed narrative from **_The Great Gatsby_ |
