# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-06 15:18
**Branch**: `ek/tom-eric-moe-tuning`
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
| summarization | 128 | 162 | turbo4v2 | 384.9 | 62.0 | 200 | 442ms | — | 2.1756 | — | 0.5499 | 10.41GB | 10.95GB | 12MB | 16MB | <\|channel\|>analysis<\|message\|>We have a user input: "The Gre |
| summarization | 256 | 291 | turbo4v2 | 189.7 | 62.6 | 200 | 1567ms | — | 2.4746 | — | 0.6803 | 10.41GB | 11.32GB | 0MB | 22MB | <\|channel\|>analysis<\|message\|>We have a user who posted some |
| summarization | 512 | 544 | turbo4v2 | 543.1 | 60.6 | 200 | 1002ms | — | 2.2788 | — | 0.1609 | 10.41GB | 11.91GB | 30MB | 33MB | <\|channel\|>analysis<\|message\|>We have to summarize content.  |
| summarization | 1024 | 1053 | turbo4v2 | 603.5 | 59.4 | 200 | 1929ms | — | 2.3818 | — | 0.3017 | 10.41GB | 12.51GB | 43MB | 56MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 2048 | 2061 | turbo4v2 | 678.2 | 58.1 | 200 | 3042ms | — | 3.1063 | — | 0.3734 | 10.41GB | 12.40GB | 85MB | 100MB | <\|channel\|>analysis<\|message\|>The user has provided a large  |
| summarization | 4096 | 4055 | turbo4v2 | 652.5 | 56.6 | 200 | 6409ms | — | 2.6036 | — | 0.3054 | 10.41GB | 13.67GB | 165MB | 189MB | <\|channel\|>analysis<\|message\|>We have a user query: "Summari |
| summarization | 8192 | 8042 | turbo4v2 | 640.4 | 51.7 | 200 | 12799ms | — | 3.0330 | — | 0.3863 | 10.41GB | 13.71GB | 390MB | 366MB | <\|channel\|>analysis<\|message\|>We have a long text. The user  |
| summarization | 16384 | 15955 | turbo4v2 | 611.0 | 45.0 | 200 | 26389ms | — | 3.1122 | — | 0.2912 | 10.41GB | 13.69GB | 575MB | 718MB | <\|channel\|>analysis<\|message\|>We have a huge chunk of text,  |
| summarization | 32768 | 31717 | turbo4v2 | 542.5 | 36.7 | 200 | 58749ms | — | 4.4206 | — | 0.4225 | 10.41GB | 13.78GB | 1.41GB | 1.38GB | <\|channel\|>analysis<\|message\|>We have a long text: It's a ve |
| summarization | 65536 | 63299 | turbo4v2 | 426.0 | 22.2 | 200 | 148833ms | — | 3.1945 | — | 0.3102 | 10.41GB | 15.84GB | 2.19GB | 2.76GB | <\|channel\|>analysis<\|message\|>We have a huge block of text.  |
| summarization | 131072 | 126728 | turbo4v2 | 300.3 | 13.2 | 200 | 422349ms | — | 2.5811 | — | 0.4490 | 10.41GB | 18.67GB | 5.82GB | 5.51GB | <\|channel\|>analysis<\|message\|>The user posted a massive text |
