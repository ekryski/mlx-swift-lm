# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-08 14:31
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

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
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Turbo Recompress Interval | Dynamic |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 110 | turbo4v2 | 534.9 | 88.7 | 200 | 208ms | — | 1.7474 | — | — | 2.45GB | 2.91GB | 6MB | 14MB | This excerpt from *The Great Gatsby* is a short, evocative p |
| summarization | 256 | 249 | turbo4v2 | 1052.7 | 85.9 | 200 | 237ms | — | 1.6288 | — | — | 2.45GB | 3.28GB | 13MB | 20MB | The provided text is a collection of excerpts and titles, li |
| summarization | 512 | 496 | turbo4v2 | 1056.3 | 84.1 | 200 | 470ms | — | 1.3286 | — | — | 2.45GB | 3.92GB | 18MB | 31MB | The provided text is an excerpt from *The Great Gatsby* by F |
| summarization | 1024 | 1008 | turbo4v2 | 1097.6 | 82.5 | 200 | 919ms | — | 1.4402 | — | — | 2.45GB | 4.76GB | 26MB | 54MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 2048 | 2031 | turbo4v2 | 1065.2 | 81.5 | 200 | 1930ms | — | 1.5206 | — | — | 2.45GB | 5.81GB | 41MB | 99MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 4096 | 4088 | turbo4v2 | 1631.0 | 79.4 | 200 | 2544ms | — | 1.5517 | — | — | 2.45GB | 5.53GB | 74MB | 191MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 8192 | 8192 | turbo4v2 | 2271.2 | 75.0 | 200 | 3672ms | — | 1.7034 | — | — | 2.45GB | 5.84GB | 138MB | 373MB | This is a fascinating and dense excerpt from **Nick Carraway |
| summarization | 16384 | 16384 | turbo4v2 | 2571.1 | 70.1 | 200 | 6417ms | — | 1.4898 | — | — | 2.45GB | 6.45GB | 269MB | 737MB | This text is an excerpt from **The Great Gatsby** by F. Scot |
| summarization | 32768 | 32768 | turbo4v2 | 2501.0 | 61.0 | 200 | 13142ms | — | 1.6425 | — | — | 2.45GB | 7.67GB | 521MB | 1.43GB | This is an excerpt from **The Great Gatsby** by F. Scott Fit |
| summarization | 65536 | 65536 | turbo4v2 | 1650.9 | 44.1 | 200 | 41781ms | — | 1.8997 | — | — | 2.45GB | 10.11GB | 1.01GB | 2.85GB | This is a substantial excerpt from **F. Scott Fitzgerald's * |
| summarization | 131072 | 130557 | turbo4v2 | 1061.1 | 37.1 | 200 | 123148ms | — | 1.7199 | — | — | 2.45GB | 13.69GB | 1.51GB | 5.67GB | The provided text is a collection of excerpts from **F. Scot |
