# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 13:29
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `30ff02a cleaning up performance optimization plan and inference architecture docs`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| Thinking | Yes |
| Perplexity tracking (MLX_BENCH_PPL) | Yes |
| KL divergence (MLX_BENCH_KLD) | Yes |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Speculative decoding | none |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | default |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | turbo4v2 | 766.3 | 79.0 | 351 | 152ms | 1.5758 | — | 1.3349 | — | 2.45GB | 2.63GB | 7MB | 21MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | turbo4v2 | 1054.7 | 76.8 | 350 | 243ms | 1.3150 | — | 1.0744 | — | 2.45GB | 2.74GB | 6MB | 27MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | turbo4v2 | 1626.5 | 73.6 | 400 | 310ms | 1.2780 | — | 0.7484 | — | 2.45GB | 2.97GB | 6MB | 40MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | turbo4v2 | 1910.4 | 76.7 | 400 | 532ms | 1.3601 | — | 0.6677 | — | 2.45GB | 3.22GB | 10MB | 63MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | turbo4v2 | 1783.9 | 76.4 | 400 | 1143ms | 1.5511 | — | 0.8284 | — | 2.45GB | 3.27GB | 23MB | 108MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | turbo4v2 | 1456.1 | 75.1 | 400 | 2813ms | 1.5295 | — | 0.8483 | — | 2.45GB | 3.32GB | 37MB | 200MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | turbo4v2 | 1106.3 | 72.6 | 400 | 7406ms | 1.6903 | — | 1.3672 | — | 2.45GB | 3.34GB | 32MB | 382MB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 16384 | 16384 | turbo4v2 | 725.6 | 67.8 | 400 | 22581ms | 1.5824 | — | 1.1805 | — | 2.45GB | 3.38GB | 134MB | 746MB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 32768 | 32768 | turbo4v2 | 420.3 | 60.6 | 400 | 77966ms | 1.8046 | — | 1.1842 | — | 2.45GB | 3.44GB | 196MB | 1.44GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 65536 | 65536 | turbo4v2 | 227.2 | 46.7 | 400 | 288467ms | 1.9849 | — | 1.1416 | — | 2.45GB | 3.67GB | 518MB | 2.86GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | turbo4v2 | 114.8 | 37.2 | 400 | 1137005ms | 2.0327 | — | 0.5617 | — | 2.45GB | 4.15GB | 773MB | 5.68GB | <\|channel>thought Here's a thinking process that leads to th |
