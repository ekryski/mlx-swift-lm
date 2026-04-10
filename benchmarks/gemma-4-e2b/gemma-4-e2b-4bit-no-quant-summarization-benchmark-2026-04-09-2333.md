# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-09 23:33
**Branch**: `session/all-perf-fixes`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 108GB |
| macOS | 26.3.1 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
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
| summarization | 128 | 110 | no-quant | 4138.8 | 203.7 | 200 | 27ms | — | — | — | — | 2.45GB | 2.62GB | 30MB | 68MB | The provided text is a **fragment of a poem or a piece of wr |
| summarization | 1024 | 1008 | no-quant | 8340.0 | 195.1 | 200 | 121ms | — | — | — | — | 2.45GB | 3.22GB | 44MB | 264MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 4096 | 4088 | no-quant | 7162.6 | 188.2 | 200 | 571ms | — | — | — | — | 2.45GB | 3.32GB | 77MB | 938MB | This excerpt appears to be from **The Great Gatsby** by F. S |
| summarization | 8192 | 8192 | no-quant | 5965.4 | 179.1 | 200 | 1374ms | — | — | — | — | 2.45GB | 3.34GB | 117MB | 1.79GB | This is an excerpt from **Nick Carraway's perspective** in F |
