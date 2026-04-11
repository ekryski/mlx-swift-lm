# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-11 03:41
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
| KV cache strategy | TurboQuant (turbo3) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo3 |
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
| summarization | 128 | 129 | turbo3 | 248.6 | 19.5 | 400 | 519ms | 1.9179 | 1.3024 | 0.7807 | 0.2140 | 16.56GB | 17.71GB | 39MB | 24MB | We need to respond. The user posted an excerpt: "The Great G |
| summarization | 256 | 257 | turbo3 | 53.4 | 13.7 | 400 | 4809ms | 1.6989 | 1.3273 | 0.5394 | 0.0679 | 16.56GB | 18.36GB | 47MB | 29MB | We need to respond to the user's request. The user posted a  |
| summarization | 512 | 513 | turbo3 | 57.0 | 13.6 | 400 | 8996ms | 2.1018 | 1.1487 | 0.9522 | 0.1394 | 16.56GB | 19.81GB | 34MB | 41MB | We need to respond. The user posted the opening of The Great |
| summarization | 1024 | 1025 | turbo3 | 89.3 | 13.5 | 400 | 11478ms | 1.8335 | 1.4531 | 0.7100 | 0.2941 | 16.56GB | 24.75GB | 57MB | 63MB | The user posted a text that looks like a modified version of |
| summarization | 2048 | 2049 | turbo3 | 95.1 | 13.5 | 400 | 21549ms | 1.8558 | 1.8403 | 0.7057 | 0.3931 | 16.56GB | 43.98GB | 43MB | 109MB | We need to respond to the user's prompt. The user posted a t |
| summarization | 4096 | 4097 | turbo3 | 59.3 | 13.4 | 400 | 69107ms | 2.0468 | 1.7940 | 0.8747 | 0.2780 | 16.56GB | 44.14GB | 49MB | 200MB | We need to respond to the user's input. The user posted a lo |
| summarization | 8192 | 8181 | turbo3 | 115.2 | 13.2 | 400 | 71014ms | 2.1919 | 1.8346 | 0.3723 | 0.1683 | 16.56GB | 44.16GB | 69MB | 381MB | We need to summarize the content above, which appears to be  |
| summarization | 16384 | 16344 | turbo3 | 191.1 | 12.8 | 400 | 85538ms | 2.0835 | 2.0620 | 0.3759 | 0.1850 | 16.56GB | 44.25GB | 181MB | 744MB | We need to summarize given passage. It's a long excerpt from |
| summarization | 32768 | 32553 | turbo3 | 269.0 | 12.2 | 400 | 121023ms | 1.9586 | 1.8140 | 0.1964 | 0.0517 | 16.56GB | 44.44GB | 249MB | 1.43GB | We need to summarize the content above, which appears to be  |
| summarization | 65536 | 64945 | turbo3 | 295.4 | 10.9 | 400 | 219833ms | 2.0654 | 1.5116 | 0.0797 | 0.1038 | 16.56GB | 44.82GB | 279MB | 2.84GB | We need to summarize the content above, which is a long pass |
| summarization | 131072 | 129844 | turbo3 | 264.5 | 8.5 | 204 | 490955ms | 2.6686 | 2.5346 | 0.1127 | -0.0382 | 16.56GB | 45.57GB | 564MB | 5.64GB | The Great Gatsby.  by F. Scott Fitzgerald.   I've been tryin |
