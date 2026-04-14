# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-11 20:51
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
| KV cache strategy | None (FP16) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
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
| summarization | 128 | 110 | no-quant | 1497.2 | 103.8 | 239 | 74ms | — | — | — | — | 2.45GB | 2.49GB | 4MB | 76MB | This is a very fragmented piece of text, presenting a mix of |
| summarization | 256 | 249 | no-quant | 676.9 | 101.0 | 230 | 369ms | — | — | — | — | 2.45GB | 2.49GB | 7MB | 105MB | This excerpt appears to be a collection of fragmented pieces |
| summarization | 512 | 496 | no-quant | 1748.7 | 93.6 | 387 | 284ms | — | — | — | — | 2.45GB | 2.49GB | 8MB | 193MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 1024 | 1008 | no-quant | 2471.4 | 101.6 | 400 | 409ms | — | — | — | — | 2.45GB | 2.51GB | 10MB | 308MB | This text appears to be an excerpt from **F. Scott Fitzgeral |
| summarization | 2048 | 2031 | no-quant | 2647.4 | 101.3 | 400 | 768ms | — | — | — | — | 2.45GB | 2.54GB | 16MB | 532MB | This excerpt from *The Great Gatsby* sets the stage by intro |
| summarization | 4096 | 4088 | no-quant | 2503.4 | 97.5 | 400 | 1634ms | — | — | — | — | 2.45GB | 2.58GB | 28MB | 982MB | This text appears to be an excerpt from **The Great Gatsby** |
| summarization | 8192 | 8192 | no-quant | 2189.9 | 93.3 | 400 | 3742ms | — | — | — | — | 2.45GB | 2.67GB | 52MB | 1.84GB | This is a fascinating and dense excerpt from **Nick Carraway |
| summarization | 16384 | 16384 | no-quant | 1558.2 | 82.2 | 400 | 10516ms | — | — | — | — | 2.45GB | 2.86GB | 34MB | 3.59GB | This provided text appears to be an excerpt from **Nick Carr |
| summarization | 32768 | 32768 | no-quant | 765.2 | 74.0 | 400 | 42826ms | — | — | — | — | 2.45GB | 3.22GB | 197MB | 7.09GB | This is an excerpt from **The Great Gatsby** by **F. Scott F |
