# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-04 00:10
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
| summarization | 128 | 162 | turbo4v2 | 379.5 | 61.0 | 200 | 428ms | — | 1.9781 | — | 0.5681 | 10.41GB | 10.90GB | 6MB | 16MB | <\|channel\|>analysis<\|message\|>We have a user query: They pro |
| summarization | 256 | 291 | turbo4v2 | 498.3 | 60.7 | 200 | 584ms | — | 1.5299 | — | 0.1788 | 10.41GB | 11.14GB | 4MB | 22MB | <\|channel\|>analysis<\|message\|>We need to summarize content.  |
| summarization | 512 | 544 | turbo4v2 | 582.2 | 60.2 | 200 | 935ms | — | 2.6393 | — | 0.2358 | 10.41GB | 11.67GB | 6MB | 33MB | <\|channel\|>analysis<\|message\|>The user has provided a text t |
| summarization | 1024 | 1053 | turbo4v2 | 610.5 | 59.2 | 200 | 1955ms | — | 2.4663 | — | 0.3199 | 10.41GB | 12.32GB | 45MB | 56MB | <\|channel\|>analysis<\|message\|>The user has provided a long e |
| summarization | 2048 | 2061 | turbo4v2 | 695.6 | 57.9 | 200 | 2965ms | — | 3.3143 | — | 0.4830 | 10.41GB | 12.21GB | 102MB | 100MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 4096 | 4055 | turbo4v2 | 671.2 | 56.0 | 200 | 6387ms | — | 2.7672 | — | 0.4488 | 10.41GB | 13.41GB | 115MB | 189MB | <\|channel\|>analysis<\|message\|>We have a long text. It's some |
| summarization | 8192 | 8042 | turbo4v2 | 672.6 | 52.6 | 200 | 12209ms | — | 3.0387 | — | 0.2802 | 10.41GB | 13.46GB | 179MB | 366MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | turbo4v2 | 636.3 | 46.0 | 200 | 25398ms | — | 3.2218 | — | 0.3649 | 10.41GB | 13.48GB | 670MB | 718MB | <\|channel\|>analysis<\|message\|>We need to produce a summary o |
| summarization | 32768 | 31717 | turbo4v2 | 556.3 | 38.8 | 200 | 57396ms | — | 2.7496 | — | 0.3888 | 10.41GB | 13.54GB | 1003MB | 1.38GB | <\|channel\|>analysis<\|message\|>We have a huge block of text.  |
| summarization | 65536 | 63299 | turbo4v2 | 438.2 | 24.5 | 200 | 144971ms | — | 3.4159 | — | 0.3790 | 10.41GB | 15.76GB | 0MB | 2.76GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 131072 | 126728 | turbo4v2 | 309.1 | 14.5 | 200 | 410420ms | — | 2.5212 | — | 0.3763 | 10.41GB | 18.58GB | 0MB | 5.51GB | <\|channel\|>analysis<\|message\|>We have a very long text. It a |
