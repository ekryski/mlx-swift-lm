# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-10 21:45
- **Branch**: `ek/tom-eric-moe-tuning`
- **Commit**: `b52bbd7 updating gemma4-e2b 4bit turbo4v2 benchmark without thinking`
- **Quantization**: 4bit
- **Model**: `mlx-community/Nemotron-Cascade-2-30B-A3B-4bit`

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
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | nil |
| Repetition context size | 20 |
| Presence penalty | nil |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | 12 |
| Think end token id | 13 |
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
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 100 (hardware default, applegpu_g13s) |

## System prompt

No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 129 | no-quant | 249.3 | 27.1 | 400 | 518ms | 1.3284 | 1.3678 | 0.3602 | 0.1872 | 16.56GB | 17.71GB | 40MB | 116MB | Okay, the user shared a snippet of "The Great Gatsby" with a |
| summarization | 256 | 257 | no-quant | 53.2 | 26.8 | 400 | 4830ms | 1.6653 | 1.4078 | 0.5275 | 0.1259 | 16.56GB | 18.36GB | 56MB | 144MB | The user posted text: "The Great Gatsby by F. Scott Fitzgera |
| summarization | 512 | 513 | no-quant | 38.8 | 26.7 | 400 | 13236ms | 2.0106 | 1.2768 | 0.8019 | 0.1453 | 16.56GB | 19.78GB | 46MB | 200MB | We need to respond to user. The user posted a text: "The Gre |
| summarization | 1024 | 1025 | no-quant | 78.5 | 26.7 | 400 | 13052ms | 1.2892 | 1.2705 | 0.4444 | 0.2895 | 16.56GB | 24.74GB | 48MB | 312MB | Okay, the user has shared a lengthy excerpt from "The Great  |
| summarization | 2048 | 2049 | no-quant | 103.5 | 26.8 | 400 | 19797ms | 1.9866 | 1.5538 | 1.1696 | 0.3614 | 16.56GB | 43.98GB | 77MB | 536MB | We need to respond appropriately. The user posted a text tha |
| summarization | 4096 | 4097 | no-quant | 91.9 | 26.1 | 400 | 44559ms | 1.9795 | 1.6792 | 0.7028 | 0.2156 | 16.56GB | 44.14GB | 100MB | 984MB | The user posted a large block of text that appears to be a h |
| summarization | 8192 | 8181 | no-quant | 147.3 | 25.5 | 400 | 55534ms | 1.8020 | 2.0423 | 0.2022 | 0.2255 | 16.56GB | 44.16GB | 148MB | 1.83GB | We need to provide a summary of the content above. The conte |
| summarization | 16384 | 16344 | no-quant | 222.7 | 24.3 | 400 | 73380ms | 1.8246 | 2.0892 | 0.2948 | 0.2220 | 16.56GB | 44.25GB | 245MB | 3.58GB | We need to summarize given text. The user provided a massive |
| summarization | 32768 | 32553 | no-quant | 255.8 | 23.3 | 400 | 127243ms | 1.8588 | 1.4106 | 0.2245 | 0.0283 | 16.56GB | 44.44GB | 431MB | 7.04GB | We need to provide a summary of the above content. The user  |
| summarization | 65536 | 64945 | no-quant | 275.4 | 21.0 | 400 | 235846ms | 1.9832 | 1.5040 | 0.0642 | 0.1056 | 16.56GB | 44.82GB | 612MB | 13.96GB | We need to summarize the provided text. It's a weird mix of  |
| summarization | 131072 | 129844 | no-quant | 247.8 | 15.3 | 400 | 524051ms | 2.2642 | 3.0897 | 0.0627 | 0.0909 | 16.56GB | 45.57GB | 1.54GB | 27.82GB | We'll need to respond with a summary. The user posted the te |
