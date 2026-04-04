# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-03 23:18
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `loan-star/gpt-oss-20b-mlx-4Bit`

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
| summarization | 128 | 162 | turbo4 | 361.4 | 60.8 | 200 | 449ms | — | 2.1172 | — | 0.2379 | 10.41GB | 10.90GB | 10MB | 21MB | <\|channel\|>analysis<\|message\|>The user has given a prompt: " |
| summarization | 256 | 291 | turbo4 | 476.5 | 60.8 | 200 | 611ms | — | 1.6628 | — | 0.2477 | 10.41GB | 11.14GB | 15MB | 29MB | <\|channel\|>analysis<\|message\|>We have a user query: "Summari |
| summarization | 512 | 544 | turbo4 | 551.8 | 60.2 | 200 | 986ms | — | 2.3151 | — | 0.3138 | 10.41GB | 11.67GB | 26MB | 43MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 1024 | 1053 | turbo4 | 605.8 | 59.2 | 200 | 1852ms | — | 2.6991 | — | 0.2810 | 10.41GB | 12.32GB | 40MB | 73MB | <\|channel\|>analysis<\|message\|>The user provided a text: It's |
| summarization | 2048 | 2061 | turbo4 | 686.3 | 58.1 | 200 | 3006ms | — | 2.8398 | — | 0.3268 | 10.41GB | 12.21GB | 102MB | 131MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 4096 | 4055 | turbo4 | 662.9 | 55.9 | 200 | 6367ms | — | 3.6195 | — | 0.4588 | 10.41GB | 13.41GB | 165MB | 247MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | turbo4 | 663.6 | 52.3 | 200 | 12385ms | — | 2.7630 | — | 0.4125 | 10.41GB | 13.46GB | 0MB | 479MB | <\|channel\|>analysis<\|message\|>We have a long text, presumabl |
| summarization | 16384 | 15955 | turbo4 | 628.7 | 45.8 | 200 | 25684ms | — | 2.5184 | — | 0.4064 | 10.41GB | 13.48GB | 766MB | 939MB | <\|channel\|>analysis<\|message\|>We have a huge chunk of text.  |
| summarization | 32768 | 31717 | turbo4 | 548.4 | 38.4 | 200 | 58249ms | — | 3.0156 | — | 0.4683 | 10.41GB | 13.54GB | 1.04GB | 1.81GB | <\|channel\|>analysis<\|message\|>We need to summarize content.  |
| summarization | 65536 | 63299 | turbo4 | 437.7 | 24.5 | 200 | 145049ms | — | 2.7213 | — | 0.3834 | 10.41GB | 15.76GB | 124MB | 3.60GB | <\|channel\|>analysis<\|message\|>We have a long stream-of-consc |
| summarization | 131072 | 126728 | turbo4 | 309.2 | 14.5 | 200 | 410289ms | — | 3.1907 | — | 0.4723 | 10.41GB | 18.58GB | 496MB | 7.20GB | <\|channel\|>analysis<\|message\|>The user provided a huge block |
