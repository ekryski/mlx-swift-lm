# Inference Benchmark - mlx-community/Qwen2.5-7B-Instruct-8bit

**Date**: 2026-04-05 15:33
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: custom
**Model**: `mlx-community/Qwen2.5-7B-Instruct-8bit`

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
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 125 | turbo4 | 383.0 | 40.1 | 200 | 328ms | — | 1.9011 | — | — | 7.54GB | 7.70GB | 7MB | 19MB | The content you've provided includes the table of contents f |
| summarization | 256 | 254 | turbo4 | 471.9 | 40.0 | 200 | 539ms | — | 1.5521 | — | — | 7.54GB | 7.86GB | 12MB | 26MB | The excerpt provided appears to be the beginning of F. Scott |
| summarization | 512 | 506 | turbo4 | 490.0 | 39.8 | 200 | 1033ms | — | 1.6734 | — | — | 7.54GB | 8.05GB | 25MB | 41MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 1024 | 1020 | turbo4 | 517.0 | 39.3 | 200 | 2051ms | — | 1.5816 | — | — | 7.54GB | 8.33GB | 48MB | 71MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald b |
| summarization | 2048 | 2035 | turbo4 | 516.0 | 38.6 | 200 | 4096ms | — | 1.7260 | — | — | 7.54GB | 8.73GB | 102MB | 130MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald b |
| summarization | 4096 | 4031 | turbo4 | 512.3 | 37.6 | 200 | 8035ms | — | 2.0061 | — | — | 7.54GB | 8.83GB | 184MB | 246MB | The excerpt from F. Scott Fitzgerald's "The Great Gatsby" in |
| summarization | 8192 | 8026 | turbo4 | 489.0 | 34.6 | 200 | 16604ms | — | 1.9325 | — | — | 7.54GB | 9.00GB | 384MB | 478MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald i |
| summarization | 16384 | 16003 | turbo4 | 456.6 | 32.2 | 200 | 35299ms | — | 1.7193 | — | — | 7.54GB | 9.25GB | 779MB | 941MB | The passage from "The Great Gatsby" by F. Scott Fitzgerald i |
