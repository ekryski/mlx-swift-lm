# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-03 22:51
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
| summarization | 128 | 162 | affine-4 | 358.7 | 60.2 | 200 | 453ms | — | 2.0561 | — | 0.4612 | 10.41GB | 10.90GB | 1MB | 25MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 256 | 291 | affine-4 | 469.0 | 60.7 | 200 | 621ms | — | 2.1909 | — | 0.2508 | 10.41GB | 11.14GB | 18MB | 34MB | <\|channel\|>analysis<\|message\|>The user has given a prompt: " |
| summarization | 512 | 544 | affine-4 | 553.5 | 59.8 | 200 | 983ms | — | 2.6737 | — | 0.1927 | 10.41GB | 11.67GB | 10MB | 51MB | <\|channel\|>analysis<\|message\|>The user wants to summarize th |
| summarization | 1024 | 1053 | affine-4 | 605.6 | 59.1 | 200 | 1845ms | — | 2.4450 | — | 0.3637 | 10.41GB | 12.32GB | 0MB | 86MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 2048 | 2061 | affine-4 | 688.3 | 57.7 | 200 | 2997ms | — | 2.8543 | — | 0.3368 | 10.41GB | 12.21GB | 94MB | 155MB | <\|channel\|>analysis<\|message\|>We have a user-provided text:  |
| summarization | 4096 | 4055 | affine-4 | 662.8 | 55.9 | 200 | 6378ms | — | 2.5806 | — | 0.3691 | 10.41GB | 13.41GB | 165MB | 291MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | affine-4 | 665.4 | 52.1 | 200 | 12380ms | — | 3.0149 | — | 0.2888 | 10.41GB | 13.46GB | 325MB | 563MB | <\|channel\|>analysis<\|message\|>We have a long user-provided t |
| summarization | 16384 | 15955 | affine-4 | 628.8 | 46.2 | 200 | 25704ms | — | 2.8281 | — | 0.2565 | 10.41GB | 13.48GB | 0MB | 1.08GB | <\|channel\|>analysis<\|message\|>We have a user query: "Summari |
| summarization | 32768 | 31717 | affine-4 | 548.8 | 38.0 | 200 | 58192ms | — | 2.5928 | — | 0.2189 | 10.41GB | 13.54GB | 940MB | 2.13GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 65536 | 63299 | affine-4 | 429.7 | 24.4 | 200 | 147734ms | — | 2.8741 | — | 0.3959 | 10.41GB | 15.76GB | 1.94GB | 4.24GB | <\|channel\|>analysis<\|message\|>We have a very long user-provi |
| summarization | 131072 | 126728 | affine-4 | 300.2 | 14.4 | 200 | 422572ms | — | 2.8913 | — | 0.4837 | 10.41GB | 18.58GB | 5.82GB | 8.47GB | <\|channel\|>analysis<\|message\|>We have a huge text. The user: |
