# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-16 02:57
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
| summarization | 128 | 117 | turbo4v2 | 238.0 | 120.1 | 400 | 492ms | 2.7889 | 1.6181 | 0.5010 | 0.3538 | 1.40GB | 1.67GB | 0MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4v2 | 221.8 | 120.5 | 400 | 1123ms | 2.7779 | 2.6556 | 0.4149 | 0.2404 | 1.40GB | 1.67GB | 0MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4v2 | 229.8 | 119.0 | 400 | 2193ms | 2.8688 | 2.0611 | 0.4919 | 0.2188 | 1.40GB | 1.68GB | 0MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 231.9 | 117.9 | 400 | 4394ms | 3.0266 | 2.9025 | 0.3372 | 0.2433 | 1.40GB | 1.76GB | 0MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4v2 | 241.5 | 116.9 | 400 | 8458ms | 3.7813 | 2.3471 | 0.2771 | 0.1500 | 1.40GB | 1.82GB | 0MB | 109MB | The user wants a summary of the provided text. I need to ext |
| summarization | 4096 | 4085 | turbo4v2 | 242.1 | 115.1 | 400 | 16872ms | 2.7052 | 3.3609 | 0.3249 | 0.1393 | 1.40GB | 1.81GB | 0MB | 199MB | The user wants a summary of the provided text. The text is a |
| summarization | 8192 | 8190 | turbo4v2 | 240.0 | 109.2 | 400 | 34134ms | 6.5090 | 2.9127 | 0.4774 | 0.1502 | 1.40GB | 1.88GB | 0MB | 382MB | 用户提供了《The Great Gatsby》的全文及其前文摘要，要求总结内容。首先需要确保分析符合中国语言规范，且需基 |
| summarization | 16384 | 16361 | turbo4v2 | 235.8 | 100.8 | 400 | 69394ms | 3.4584 | 3.1250 | 0.3808 | 0.2105 | 1.40GB | 2.13GB | 0MB | 745MB | The user wants a summary of the text provided in the documen |
| summarization | 32768 | 32700 | turbo4v2 | 230.1 | 86.9 | 400 | 142101ms | 4.0164 | 2.9218 | 0.2765 | 0.1326 | 1.40GB | 2.44GB | 0MB | 1.44GB | The user is asking for a summary of the provided text, which |
| summarization | 65536 | 65468 | turbo4v2 | 214.2 | 68.5 | 400 | 305651ms | 2.9112 | 2.6812 | 0.2732 | 0.1605 | 1.40GB | 3.31GB | 0MB | 2.86GB | The user wants a summary of the provided text, which is "The |
| summarization | 131072 | 130773 | turbo4v2 | 192.6 | 48.6 | 400 | 678831ms | 3.0209 | 3.7262 | 0.2809 | 0.0834 | 1.40GB | 4.75GB | 0MB | 5.69GB | The user wants a summary of the provided text, which appears |
