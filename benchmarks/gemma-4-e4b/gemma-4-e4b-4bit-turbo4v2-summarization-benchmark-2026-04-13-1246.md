# Inference Benchmark - Gemma 4 E4B

- **Date**: 2026-04-13 12:46
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `0445a6d perf: fix prefill memory bloat for SSM/GDN hybrid models`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-e4b-it-4bit`

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
| Collect per-token data | true |
| Track perplexity | true |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 0 |
| Thinking (effective) | No |
| Perplexity tracking (MLX_BENCH_PPL) | Yes |
| KL divergence (MLX_BENCH_KLD) | Yes |
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
| summarization | 128 | 110 | turbo4v2 | 532.6 | 67.0 | 44 | 207ms | — | 5.3780 | — | 1.7524 | 3.98GB | 4.33GB | 0MB | 7MB | This content is a section The provided above.  The content a |
| summarization | 512 | 496 | turbo4v2 | 679.1 | 63.7 | 226 | 732ms | — | 2.2021 | — | 0.9272 | 3.98GB | 4.63GB | 0MB | 32MB | This excerpted.  The provided is the text describes the prov |
| summarization | 1024 | 1008 | turbo4v2 | 704.8 | 59.5 | 271 | 1433ms | — | 2.1559 | — | 0.7091 | 3.98GB | 4.67GB | 0MB | 57MB | Here's a summary of the provided text. The selection consist |
| summarization | 4096 | 4088 | turbo4v2 | 622.4 | 62.5 | 400 | 6570ms | — | 2.4589 | — | 0.7666 | 3.98GB | 5.00GB | 0MB | 199MB | This text is the beginning of *The Great Gatsby*, which begi |
| summarization | 32768 | 32768 | turbo4v2 | 230.6 | 49.9 | 400 | 142085ms | — | 2.8293 | — | 0.8794 | 3.98GB | 5.54GB | 0MB | 1.44GB | This is a very substantial excerpt from *The Great Gatsby*,  |
