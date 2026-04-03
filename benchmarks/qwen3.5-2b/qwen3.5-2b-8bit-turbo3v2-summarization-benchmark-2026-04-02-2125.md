# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 21:25
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |
| Repetition Penalty | 1.0 |
| Presence Penalty | 1.5 |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | turbo3v2 | 611.1 | 81.2 | 400 | 196ms | 3.1351 | 3.2333 | 0.0405 | 0.0132 | 1.86GB | 2.08GB | 19MB | 20MB | The text you've provided is a **poem** written by **Zelda Fi |
| summarization | 256 | 251 | turbo3v2 | 854.9 | 79.7 | 400 | 294ms | 4.8186 | 3.4073 | 0.0497 | 0.0183 | 1.86GB | 2.31GB | 8MB | 24MB | Here is a summary of the text you provided:  The passage beg |
| summarization | 512 | 506 | turbo3v2 | 984.8 | 79.4 | 201 | 514ms | 2.5841 | 1.7715 | 0.0389 | 0.0623 | 1.86GB | 2.71GB | 19MB | 27MB | This passage from *The Great Gatsby* (specifically "Once Aga |
| summarization | 1024 | 1021 | turbo3v2 | 1024.3 | 79.6 | 400 | 997ms | 2.7070 | 2.6352 | 0.0133 | 0.0421 | 1.86GB | 3.27GB | 20MB | 53MB | Here is a summary of the provided text, which is the **Prolo |
| summarization | 2048 | 2044 | turbo3v2 | 1075.9 | 80.0 | 400 | 1927ms | 2.3110 | 3.3884 | 0.0511 | 0.0228 | 1.86GB | 3.97GB | 38MB | 92MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 4096 | 4087 | turbo3v2 | 1261.0 | 80.0 | 201 | 3283ms | 3.3004 | 1.4945 | 0.0329 | 0.4176 | 1.86GB | 3.95GB | 61MB | 161MB | **Note:** There are errors in the source text provided (e.g. |
| summarization | 8192 | 8192 | turbo3v2 | 1343.0 | 76.8 | 201 | 6143ms | 2.2660 | 2.2403 | 0.0233 | 0.0012 | 1.86GB | 4.04GB | 108MB | 316MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* c |
| summarization | 16384 | 16363 | turbo3v2 | 1360.9 | 72.2 | 393 | 12136ms | 3.1204 | 1.2307 | 0.0130 | 0.0084 | 1.86GB | 4.39GB | 207MB | 630MB | Based on the text provided from **"The Great Gatsby" by F. S |
| summarization | 32768 | 32702 | turbo3v2 | 1296.7 | 65.2 | 201 | 25529ms | 3.2460 | 2.7768 | 0.0268 | 0.0553 | 1.86GB | 5.00GB | 396MB | 1.21GB | This excerpt from **F. Scott Fitzgerald's *The Great Gatsby* |
| summarization | 65536 | 65470 | turbo3v2 | 1055.8 | 51.0 | 212 | 67579ms | 3.0668 | 1.8500 | 0.0267 | -0.0252 | 1.86GB | 6.33GB | 585MB | 2.41GB | This is a detailed summary of F. Scott Fitzgerald's novel, * |
| summarization | 131072 | 130775 | turbo3v2 | 798.0 | 39.6 | 202 | 164307ms | 3.0511 | 7.9860 | 0.0584 | 0.3594 | 1.86GB | 8.63GB | 1.51GB | 4.81GB | This text consists of excerpts from two famous literary work |
