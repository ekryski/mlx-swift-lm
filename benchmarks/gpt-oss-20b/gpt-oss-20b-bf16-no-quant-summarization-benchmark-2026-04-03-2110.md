# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-03 21:10
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `sjgdr/gpt-oss-20b-mlx-fp16`

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
| Temperature | 0.8 |
| Top P | 0.8 |
| Top K | 0 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Reasoning Effort | medium |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 162 | no-quant | 270.4 | 20.7 | 200 | 600ms | — | 2.0020 | — | — | 12.82GB | 15.66GB | 5MB | 79MB | <\|channel\|>analysis<\|message\|>The user provided a text that  |
| summarization | 256 | 291 | no-quant | 436.0 | 20.7 | 200 | 668ms | — | 2.0266 | — | — | 12.82GB | 15.82GB | 18MB | 107MB | <\|channel\|>analysis<\|message\|>The user posted a block of tex |
| summarization | 512 | 544 | no-quant | 536.4 | 20.5 | 200 | 1014ms | — | 2.2702 | — | — | 12.82GB | 16.14GB | 23MB | 163MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 1024 | 1053 | no-quant | 600.1 | 20.8 | 200 | 1909ms | — | 2.1840 | — | — | 12.82GB | 16.85GB | 54MB | 274MB | <\|channel\|>analysis<\|message\|>We need to summarize content.  |
| summarization | 2048 | 2061 | no-quant | 665.0 | 20.9 | 200 | 3102ms | — | 2.5548 | — | — | 12.82GB | 15.64GB | 85MB | 495MB | <\|channel\|>analysis<\|message\|>The user provided a large exce |
| summarization | 4096 | 4055 | no-quant | 677.1 | 20.6 | 200 | 6263ms | — | 3.1915 | — | — | 12.82GB | 17.89GB | 165MB | 931MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | no-quant | 677.2 | 20.5 | 200 | 12159ms | — | 2.7480 | — | — | 12.82GB | 17.95GB | 374MB | 1.76GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | no-quant | 639.4 | 19.6 | 200 | 25244ms | — | 2.1985 | — | — | 12.82GB | 17.82GB | 766MB | 3.45GB | <\|channel\|>analysis<\|message\|>The user has provided a huge c |
| summarization | 32768 | 31717 | no-quant | 554.8 | 18.1 | 200 | 57595ms | — | 2.4602 | — | — | 12.82GB | 18.14GB | 1.41GB | 6.82GB | <\|channel\|>analysis<\|message\|>The user provided an extremely |
| summarization | 65536 | 63299 | no-quant | 430.6 | 14.3 | 200 | 147423ms | — | 2.2083 | — | — | 12.82GB | 20.29GB | 2.91GB | 13.56GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 131072 | 126728 | no-quant | 303.7 | 10.1 | 200 | 417674ms | — | 2.4517 | — | — | 12.82GB | 23.12GB | 5.58GB | 27.11GB | <\|channel\|>analysis<\|message\|>We need to summarize this huge |
