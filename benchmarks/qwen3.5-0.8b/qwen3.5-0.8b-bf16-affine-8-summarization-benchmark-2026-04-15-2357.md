# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-15 23:57
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
- **Quantization**: bf16
- **Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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
| KV cache strategy | Affine (8-bit, group 64, start 512) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | 8 |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 512 |
| Prefill step size | 2048 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | 248068 |
| Think end token id | 248069 |
| Thinking phase prefilled | true |
| Collect per-token data | true |
| Track perplexity | true |
| N-gram size | 0 |
| Max n-gram draft tokens | 5 |
| Additional processors count | 1 |
| Thinking token budget (processor) | 200 |
| Thinking (effective) | Yes |
| Perplexity tracking (MLX_BENCH_PPL) | Yes |
| KL divergence (MLX_BENCH_KLD) | Yes |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Speculative decoding | none |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 200 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 117 | affine-8 | 225.5 | 118.1 | 400 | 519ms | 2.6558 | 1.3835 | 0.5478 | 0.1730 | 1.40GB | 1.66GB | 0MB | 64MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-8 | 233.4 | 117.1 | 400 | 1067ms | 2.6429 | 2.0428 | 0.5158 | 0.1582 | 1.40GB | 1.67GB | 0MB | 80MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-8 | 231.3 | 116.3 | 400 | 2180ms | 2.3522 | 2.4242 | 0.4623 | 0.1923 | 1.40GB | 1.70GB | 0MB | 111MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-8 | 234.9 | 116.1 | 400 | 4339ms | 2.3138 | 2.5102 | 0.4773 | 0.1575 | 1.40GB | 1.73GB | 0MB | 175MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-8 | 241.0 | 114.1 | 400 | 8472ms | 4.0356 | 3.4919 | 0.3165 | 0.1631 | 1.40GB | 1.82GB | 0MB | 300MB | The user wants a summary of the provided text, which is an e |
| summarization | 4096 | 4085 | affine-8 | 241.0 | 111.8 | 400 | 16954ms | 2.5060 | 2.1304 | 0.5273 | 0.1851 | 1.40GB | 1.81GB | 0MB | 552MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-8 | 239.9 | 106.0 | 400 | 34142ms | 2.7018 | 1.1099 | 0.4621 | 0.0399 | 1.40GB | 1.88GB | 0MB | 1.03GB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 16384 | 16361 | affine-8 | 236.8 | 98.4 | 400 | 69103ms | 2.7887 | 2.8505 | 0.5479 | 0.1305 | 1.40GB | 2.14GB | 0MB | 2.01GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | affine-8 | 228.9 | 85.6 | 400 | 142895ms | 2.6237 | 3.6068 | 0.2701 | 0.1908 | 1.40GB | 2.44GB | 0MB | 3.98GB | The user wants a summary of the text "The Great Gatsby" by F |
| summarization | 65536 | 65468 | affine-8 | 215.0 | 67.2 | 400 | 304459ms | 3.1837 | 3.3766 | 0.3311 | 0.1802 | 1.40GB | 3.19GB | 0MB | 7.91GB | The user is asking for a summary of the provided text, which |
| summarization | 131072 | 130773 | affine-8 | 191.9 | 47.9 | 400 | 681452ms | 2.7241 | 2.1905 | 0.2034 | 0.1177 | 1.40GB | 4.75GB | 0MB | 15.76GB | The user wants a summary of the provided text, which is exce |
