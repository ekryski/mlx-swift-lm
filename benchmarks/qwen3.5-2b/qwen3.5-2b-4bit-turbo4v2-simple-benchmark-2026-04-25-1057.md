# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-25 10:57
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
| simple | 4096 | 41 | turbo4v2 | 944.2 | 290.8 | 40 | 44ms | — | — | — | — | 1010MB | 1.14GB | 9MB | 4MB | My name is Qwen3.5, a large language model developed by Tong |
