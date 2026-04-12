# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-10 15:53
**Branch**: `ek/tom-eric-moe-tuning`
**Commit**: `a44f9f2 reformatting benchmarks a bit`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| Max Tokens | 200 |
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
| summarization | 128 | 110 | turbo4v2 | 968.4 | 107.4 | 197 | 114ms | — | — | — | — | 2.45GB | 2.62GB | 30MB | 14MB | The provided text appears to be a **fragment of a poem or li |
| summarization | 256 | 249 | turbo4v2 | 1411.6 | 99.3 | 200 | 177ms | — | — | — | — | 2.45GB | 2.73GB | 33MB | 20MB | This excerpt is a collection of fragmented pieces from or re |
| summarization | 512 | 496 | turbo4v2 | 2175.4 | 96.4 | 200 | 229ms | — | — | — | — | 2.45GB | 2.96GB | 35MB | 31MB | This excerpt appears to be a collection of fragments from or |
| summarization | 1024 | 1008 | turbo4v2 | 1954.3 | 99.8 | 200 | 517ms | — | — | — | — | 2.45GB | 3.22GB | 44MB | 54MB | The provided text is an excerpt from *The Great Gatsby* by F |
| summarization | 2048 | 2031 | turbo4v2 | 1864.4 | 99.8 | 200 | 1091ms | — | — | — | — | 2.45GB | 3.27GB | 57MB | 99MB | This text appears to be an excerpt from **The Great Gatsby** |
| summarization | 4096 | 4088 | turbo4v2 | 1463.5 | 96.9 | 200 | 2794ms | — | — | — | — | 2.45GB | 3.33GB | 77MB | 191MB | This excerpt from *The Great Gatsby* establishes a reflectiv |
| summarization | 8192 | 8192 | turbo4v2 | 1117.4 | 92.2 | 200 | 7333ms | — | — | — | — | 2.45GB | 3.34GB | 69MB | 373MB | This is a selection of text from **The Great Gatsby** by F.  |
| summarization | 16384 | 16384 | turbo4v2 | 724.0 | 85.5 | 200 | 22630ms | — | — | — | — | 2.45GB | 3.38GB | 172MB | 737MB | This excerpt from *The Great Gatsby* is dense and focuses he |
| summarization | 32768 | 32768 | turbo4v2 | 419.8 | 74.6 | 200 | 78051ms | — | — | — | — | 2.45GB | 3.44GB | 197MB | 1.43GB | This is a dense and fragmented excerpt from a novel, likely  |
