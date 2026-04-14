# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 21:27
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `aeec89d updating build system to use make so we stop having stale C and metal file issues`
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
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 2048 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Repetition penalty | nil |
| Repetition context size | 20 |
| Presence penalty | nil |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Collect per-token data | false |
| Track perplexity | false |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 0 |
| Thinking (effective) | No |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Speculative decoding | none |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 100 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 110 | turbo4v2 | 1015.1 | 106.7 | 209 | 109ms | — | — | — | — | 2.45GB | 2.63GB | 3MB | 14MB | The provided text is a very fragmented collection of element |
| summarization | 256 | 249 | turbo4v2 | 1668.7 | 99.9 | 279 | 150ms | — | — | — | — | 2.45GB | 2.75GB | 6MB | 23MB | This excerpt from *The Great Gatsby* includes several distin |
| summarization | 512 | 496 | turbo4v2 | 2108.5 | 95.9 | 400 | 236ms | — | — | — | — | 2.45GB | 3.00GB | 8MB | 40MB | The provided text consists of a few distinct pieces: a title |
| summarization | 1024 | 1008 | turbo4v2 | 1986.0 | 101.0 | 400 | 509ms | — | — | — | — | 2.45GB | 3.28GB | 11MB | 63MB | This excerpt comes from **The Great Gatsby** by F. Scott Fit |
| summarization | 2048 | 2031 | turbo4v2 | 1634.8 | 99.4 | 400 | 1244ms | — | — | — | — | 2.45GB | 3.31GB | 12MB | 108MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | turbo4v2 | 1379.7 | 96.2 | 400 | 2964ms | — | — | — | — | 2.45GB | 3.33GB | 29MB | 199MB | This text is an excerpt from **The Great Gatsby** by F. Scot |
| summarization | 8192 | 8192 | turbo4v2 | 1022.0 | 93.0 | 400 | 8017ms | — | — | — | — | 2.45GB | 3.35GB | 44MB | 382MB | This is an excerpt from **Nick Carraway's narration** in **T |
| summarization | 16384 | 16384 | turbo4v2 | 665.2 | 84.6 | 400 | 24633ms | — | — | — | — | 2.45GB | 3.40GB | 84MB | 746MB | This provided text appears to be an excerpt from **The Great |
| summarization | 32768 | 32768 | turbo4v2 | 388.6 | 74.2 | 400 | 84316ms | — | — | — | — | 2.45GB | 3.49GB | 197MB | 1.44GB | This is a rich, dense, and deeply introspective segment from |
