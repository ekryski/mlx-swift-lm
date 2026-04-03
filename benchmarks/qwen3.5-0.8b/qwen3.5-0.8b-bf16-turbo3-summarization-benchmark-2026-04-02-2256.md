# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-02 22:56
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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
| summarization | 128 | 119 | turbo3 | 1099.5 | 84.3 | 400 | 109ms | 4.5673 | 3.5965 | 0.0354 | 0.0225 | 1.40GB | 1.61GB | 16MB | 23MB | This is a **poetic and slightly distorted lyric** contained  |
| summarization | 256 | 251 | turbo3 | 2155.2 | 84.9 | 334 | 117ms | 4.4113 | 4.6428 | 0.0545 | 0.0763 | 1.40GB | 1.82GB | 16MB | 26MB | Based on the text provided, here is a summary of the first c |
| summarization | 512 | 506 | turbo3 | 2691.4 | 84.8 | 400 | 188ms | 4.5827 | 3.8425 | 0.0323 | 0.0592 | 1.40GB | 2.24GB | 21MB | 40MB | Based on the text provided, here is a summary of its main th |
| summarization | 1024 | 1021 | turbo3 | 3220.9 | 87.5 | 400 | 317ms | 4.1711 | 5.2577 | 0.0351 | 0.0085 | 1.40GB | 2.76GB | 14MB | 63MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 2048 | 2044 | turbo3 | 3545.4 | 84.3 | 400 | 577ms | 3.5246 | 4.9217 | 0.0223 | 0.0287 | 1.40GB | 3.51GB | 29MB | 109MB | Based on the provided text, here is a summary of the passage |
| summarization | 4096 | 4087 | turbo3 | 4068.0 | 84.9 | 400 | 1005ms | 3.7707 | 4.4440 | 0.0345 | 0.0755 | 1.40GB | 3.43GB | 20MB | 199MB | Here is a summary of the provided text:  **Title:** *The Gre |
| summarization | 8192 | 8192 | turbo3 | 4328.5 | 81.7 | 400 | 1897ms | 4.2034 | 4.7671 | 0.0406 | 0.0191 | 1.40GB | 3.72GB | 111MB | 382MB | This text is the second section (Chapter IV) of *The Great G |
| summarization | 16384 | 16363 | turbo3 | 4051.5 | 78.4 | 400 | 4109ms | 4.6993 | 3.9839 | 0.0324 | 0.0698 | 1.40GB | 3.98GB | 173MB | 745MB | Based on the text *The Great Gatsby*, here is a summary of t |
| summarization | 32768 | 32702 | turbo3 | 3400.8 | 66.5 | 400 | 10124ms | 4.1504 | 3.9543 | 0.0311 | 0.0392 | 1.40GB | 4.82GB | 399MB | 1.44GB | Based on the text *The Great Gatsby* by F. Scott Fitzgerald, |
| summarization | 65536 | 65470 | turbo3 | 1832.1 | 44.5 | 399 | 36609ms | 4.9901 | 2.9455 | 0.0077 | 0.0283 | 1.40GB | 7.25GB | 718MB | 2.86GB | This excerpted volume from *The Great Gatsby* by F. Scott Fi |
| summarization | 131072 | 130775 | turbo3 | 1186.5 | 24.9 | 400 | 111205ms | 4.0543 | 3.1391 | 0.0400 | 0.0252 | 1.40GB | 8.29GB | 1.51GB | 5.69GB | Based on the text provided, here is a summary of its content |
