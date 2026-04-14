# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 21:11
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
| summarization | 128 | 110 | turbo4v2 | 924.3 | 104.2 | 193 | 120ms | — | — | — | — | 2.45GB | 2.49GB | 2MB | 13MB | The provided text is a very fragmented collection of excerpt |
| summarization | 256 | 249 | turbo4v2 | 1157.9 | 96.3 | 286 | 216ms | — | — | — | — | 2.45GB | 2.49GB | 5MB | 24MB | The provided text is a collection of fragments from or relat |
| summarization | 512 | 496 | turbo4v2 | 1859.0 | 95.4 | 366 | 268ms | — | — | — | — | 2.45GB | 2.49GB | 2MB | 38MB | This excerpt from *The Great Gatsby* introduces a section th |
| summarization | 1024 | 1008 | turbo4v2 | 2340.5 | 99.0 | 400 | 432ms | — | — | — | — | 2.45GB | 2.51GB | 7MB | 63MB | This excerpt from *The Great Gatsby* provides a glimpse into |
| summarization | 2048 | 2031 | turbo4v2 | 2505.5 | 98.6 | 400 | 812ms | — | — | — | — | 2.45GB | 2.54GB | 0MB | 108MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | turbo4v2 | 2542.3 | 98.7 | 400 | 1609ms | — | — | — | — | 2.45GB | 2.58GB | 20MB | 199MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 8192 | 8192 | turbo4v2 | 2152.3 | 92.8 | 400 | 3807ms | — | — | — | — | 2.45GB | 2.67GB | 52MB | 382MB | This appears to be an excerpt from **Nick Carraway's narrati |
| summarization | 16384 | 16384 | turbo4v2 | 1612.0 | 86.0 | 400 | 10165ms | — | — | — | — | 2.45GB | 2.86GB | 84MB | 746MB | This is a summary of the provided excerpts from F. Scott Fit |
| summarization | 32768 | 32768 | turbo4v2 | 842.7 | 71.9 | 400 | 38891ms | — | — | — | — | 2.45GB | 3.22GB | 196MB | 1.44GB | This is a substantial amount of text, clearly from a work of |
