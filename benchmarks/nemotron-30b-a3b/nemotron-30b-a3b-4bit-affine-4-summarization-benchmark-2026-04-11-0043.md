# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-11 00:43
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
| KV cache strategy | Affine (4-bit, group 64, start 512) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | 4 |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 512 |
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
| summarization | 128 | 129 | affine-4 | 251.9 | 26.5 | 400 | 513ms | 1.4776 | 1.2349 | 0.4680 | 0.2232 | 16.56GB | 17.71GB | 43MB | 36MB | Okay, the user shared a partial text of "The Great Gatsby" w |
| summarization | 256 | 257 | affine-4 | 33.2 | 27.1 | 400 | 7741ms | 1.9673 | 1.3417 | 0.9523 | 0.2162 | 16.56GB | 18.42GB | 49MB | 45MB | We need to respond to user's message. The user posted a frag |
| summarization | 512 | 513 | affine-4 | 51.0 | 29.8 | 400 | 10058ms | 1.8902 | 1.4623 | 0.9083 | 0.2454 | 16.56GB | 19.78GB | 44MB | 62MB | We need to respond to the user. The user posted the beginnin |
| summarization | 1024 | 1025 | affine-4 | 156.3 | 26.9 | 400 | 6560ms | 1.9777 | 1.3674 | 0.8863 | 0.3554 | 16.56GB | 24.74GB | 17MB | 97MB | We need to respond to the user. The user posted a text that  |
| summarization | 2048 | 2049 | affine-4 | 108.1 | 26.8 | 400 | 18951ms | 1.7884 | 1.5264 | 0.9654 | 0.2562 | 16.56GB | 43.98GB | 44MB | 167MB | We need to respond to user message. The user posted a text t |
| summarization | 4096 | 4097 | affine-4 | 92.0 | 26.7 | 400 | 44511ms | 2.1344 | 1.4903 | 0.7460 | 0.2591 | 16.56GB | 44.14GB | 55MB | 307MB | The user posted a large block of text that appears to be a d |
| summarization | 8192 | 8181 | affine-4 | 154.8 | 25.8 | 400 | 52840ms | 1.6943 | 2.0111 | 0.3541 | 0.1079 | 16.56GB | 44.16GB | 62MB | 587MB | We need to provide a summary of the content above. The user  |
| summarization | 16384 | 16344 | affine-4 | 210.3 | 24.1 | 400 | 77713ms | 1.8538 | 1.7481 | 0.3616 | 0.1681 | 16.56GB | 44.25GB | 78MB | 1.12GB | We need to summarize given content. The user gave a text tha |
| summarization | 32768 | 32553 | affine-4 | 280.7 | 24.7 | 400 | 115958ms | 1.9151 | 1.5426 | 0.1695 | 0.0651 | 16.56GB | 44.44GB | 58MB | 2.20GB | We need to summarize the content above. The user gave a text |
| summarization | 65536 | 64945 | affine-4 | 292.4 | 22.7 | 400 | 222074ms | 2.0180 | 1.7399 | 0.0714 | 0.0902 | 16.56GB | 44.82GB | 153MB | 4.36GB | We need to provide a summary of the content above. The conte |
| summarization | 131072 | 129844 | affine-4 | 257.6 | 19.5 | 400 | 504020ms | 2.1959 | 2.0420 | 0.0978 | 0.0756 | 16.56GB | 45.57GB | 153MB | 8.69GB |   We need to summarize the above text, which is a mixture of |
