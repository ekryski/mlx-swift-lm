# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 06:42
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `38405e0 updating the dylib files for the prefill bridge`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-0.8B-8bit`

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
| KV cache strategy | TurboQuant (turbo3) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo3 |
| KV group size | 64 |
| Quantized KV start | 0 |
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
| summarization | 128 | 117 | turbo3 | 209.8 | 151.4 | 400 | 558ms | 3.1200 | 1.7525 | 0.4622 | 0.3533 | 763MB | 1.07GB | 0MB | 23MB | The user wants a summary of the provided excerpt from *The G |
| summarization | 256 | 249 | turbo3 | 230.7 | 156.0 | 400 | 1080ms | 2.4857 | 3.4968 | 0.4881 | 0.1648 | 763MB | 1.07GB | 0MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 238.8 | 155.4 | 400 | 2112ms | 2.9449 | 3.0976 | 0.3630 | 0.1468 | 763MB | 1.07GB | 0MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo3 | 240.7 | 157.0 | 400 | 4235ms | 2.3820 | 2.3271 | 0.4484 | 0.1658 | 763MB | 1.09GB | 0MB | 63MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 2048 | 2042 | turbo3 | 242.6 | 154.1 | 400 | 8420ms | 3.7779 | 2.9727 | 0.4535 | 0.1810 | 763MB | 1.12GB | 0MB | 109MB | The user wants a summary of the provided text. The text is a |
| summarization | 4096 | 4085 | turbo3 | 241.6 | 154.1 | 400 | 16906ms | 2.5418 | 1.9427 | 0.4827 | 0.1916 | 763MB | 1.13GB | 0MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Sour |
| summarization | 8192 | 8190 | turbo3 | 239.5 | 145.5 | 400 | 34201ms | 2.5772 | 3.0578 | 0.4880 | 0.2443 | 763MB | 1.23GB | 0MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo3 | 236.2 | 131.4 | 400 | 69274ms | 3.6046 | 3.2429 | 0.2718 | 0.2098 | 763MB | 1.43GB | 0MB | 745MB | The user wants a summary of the provided text "The Great Gat |
| summarization | 32768 | 32700 | turbo3 | 229.2 | 109.4 | 400 | 142656ms | 2.5045 | 3.6094 | 0.2985 | 0.1375 | 763MB | 1.79GB | 0MB | 1.44GB | The user wants a summary of the provided text, which is *The |
| summarization | 65536 | 65468 | turbo3 | 215.7 | 81.3 | 400 | 303580ms | 3.3828 | 2.5966 | 0.3065 | 0.1101 | 763MB | 2.53GB | 0MB | 2.86GB | The user wants a summary of the provided text.  **Content An |
| summarization | 131072 | 130773 | turbo3 | 193.3 | 54.8 | 400 | 676435ms | 3.6370 | 3.1867 | 0.2459 | 0.1479 | 763MB | 4.10GB | 0MB | 5.69GB | The user wants a summary of the text provided, which is a co |
