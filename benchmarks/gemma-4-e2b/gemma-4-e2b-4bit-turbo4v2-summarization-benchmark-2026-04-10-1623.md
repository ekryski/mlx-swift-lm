# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 16:23
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `b52bbd7 updating gemma4-e2b 4bit turbo4v2 benchmark without thinking`
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
| summarization | 128 | 110 | turbo4v2 | 712.2 | 104.1 | 302 | 155ms | — | — | — | — | 2.45GB | 2.62GB | 32MB | 18MB | The provided text is a very fragmented collection of element |
| summarization | 256 | 249 | turbo4v2 | 1426.9 | 101.4 | 241 | 175ms | — | — | — | — | 2.45GB | 2.73GB | 33MB | 22MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 512 | 496 | turbo4v2 | 2080.8 | 95.1 | 400 | 239ms | — | — | — | — | 2.45GB | 2.96GB | 0MB | 40MB | This excerpt comes from **The Great Gatsby** by F. Scott Fit |
| summarization | 1024 | 1008 | turbo4v2 | 2118.7 | 100.4 | 400 | 477ms | — | — | — | — | 2.45GB | 3.22GB | 43MB | 63MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 2048 | 2031 | turbo4v2 | 1860.2 | 99.6 | 400 | 1093ms | — | — | — | — | 2.45GB | 3.27GB | 57MB | 108MB | This excerpt appears to be a collection of fragmented passag |
| summarization | 4096 | 4088 | turbo4v2 | 1481.3 | 97.3 | 400 | 2761ms | — | — | — | — | 2.45GB | 3.33GB | 77MB | 199MB | This excerpt appears to be a collection of fragmented passag |
| summarization | 8192 | 8192 | turbo4v2 | 1097.9 | 92.6 | 400 | 7462ms | — | — | — | — | 2.45GB | 3.34GB | 117MB | 382MB | This is a fascinating and dense excerpt from **Nick Carraway |
| summarization | 16384 | 16384 | turbo4v2 | 724.0 | 85.8 | 400 | 22632ms | — | — | — | — | 2.45GB | 3.38GB | 172MB | 746MB | This excerpt from *The Great Gatsby* is a rich tapestry of N |
| summarization | 32768 | 32768 | turbo4v2 | 419.5 | 75.2 | 400 | 78111ms | — | — | — | — | 2.45GB | 3.44GB | 296MB | 1.44GB | This is an excerpt from **The Great Gatsby** by **F. Scott F |
| summarization | 65536 | 65536 | turbo4v2 | 216.6 | 55.1 | 400 | 302538ms | — | — | — | — | 2.45GB | 3.67GB | 426MB | 2.86GB | This is an extensive collection of passages from **Nick Carr |
| summarization | 131072 | 130557 | turbo4v2 | 113.8 | 44.3 | 400 | 1147233ms | — | — | — | — | 2.45GB | 4.15GB | 837MB | 5.68GB | This extensive collection of excerpts from F. Scott Fitzgera |
