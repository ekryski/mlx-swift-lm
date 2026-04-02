# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 15:46
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
| summarization | 128 | 119 | turbo4v2 | 612.3 | 83.6 | 264 | 196ms | 2.9589 | 3.3174 | 0.0457 | 0.0408 | 1.86GB | 2.08GB | 12MB | 17MB | This excerpt is an epigraph (an introductory quote) from F.  |
| summarization | 1024 | 1021 | turbo4v2 | 1051.9 | 81.8 | 400 | 971ms | 2.7251 | 2.4242 | 0.0259 | 0.0458 | 1.86GB | 3.27GB | 25MB | 63MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 4096 | 4087 | turbo4v2 | 1245.2 | 80.5 | 321 | 3368ms | 2.6368 | 1.0944 | 0.0088 | 0.0049 | 1.86GB | 3.95GB | 58MB | 196MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 32768 | 32702 | turbo4v2 | 1302.1 | 64.5 | 400 | 25424ms | 2.9309 | 3.8860 | 0.0230 | 0.0507 | 1.86GB | 5.00GB | 399MB | 1.44GB | This text is a complete excerpt from **The Great Gatsby** by |
