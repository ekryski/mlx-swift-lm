# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-10 11:35
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `3c56707 cleaning up benchmarks a bit. Updating benchmark runner to also include ops_per_buffer, thinking, kld, ppl flag states`
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
| summarization | 128 | 116 | no-quant | 669.3 | 78.3 | 328 | 174ms | 1.8528 | — | 2.0576 | — | 2.45GB | 2.63GB | 5MB | 97MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 1101.3 | 76.7 | 400 | 233ms | 1.2633 | — | 0.7215 | — | 2.45GB | 2.74GB | 8MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 1389.6 | 73.9 | 400 | 362ms | 1.2661 | — | 0.9112 | — | 2.45GB | 2.97GB | 10MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 1888.9 | 76.6 | 400 | 538ms | 1.3258 | — | 0.5029 | — | 2.45GB | 3.22GB | 15MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 1785.7 | 76.5 | 400 | 1142ms | 1.5200 | — | 0.7047 | — | 2.45GB | 3.27GB | 19MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1492.0 | 75.1 | 400 | 2745ms | 1.5314 | — | 0.8743 | — | 2.45GB | 3.32GB | 37MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8192 | no-quant | 1111.4 | 72.4 | 400 | 7372ms | 1.6069 | — | 1.2568 | — | 2.45GB | 3.34GB | 69MB | 1.84GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 16384 | 16384 | no-quant | 719.5 | 67.2 | 400 | 22774ms | 1.9889 | — | 1.2866 | — | 2.45GB | 3.38GB | 135MB | 3.59GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 32768 | 32768 | no-quant | 414.6 | 60.0 | 400 | 79040ms | 1.8403 | — | 0.9946 | — | 2.45GB | 3.44GB | 262MB | 7.09GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 65536 | 65536 | no-quant | 227.2 | 47.0 | 400 | 288493ms | 2.2205 | — | 1.0786 | — | 2.45GB | 3.67GB | 518MB | 14.09GB | <\|channel>thought Here's a thinking process that leads to th |
| summarization | 131072 | 130563 | no-quant | 117.5 | 37.8 | 400 | 1111325ms | 1.6345 | — | 0.5808 | — | 2.45GB | 4.15GB | 773MB | 27.98GB | <\|channel>thought Here's a thinking process that leads to th |
