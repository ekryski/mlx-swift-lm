# Inference Benchmark - mlx-community/Qwen3.5-35B-A3B-8bit

**Date**: 2026-04-04 21:24
**Branch**: `feature/turboquant-plus-optimizations`
**Quantization**: custom
**Model**: `mlx-community/Qwen3.5-35B-A3B-8bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 115GB |
| macOS | 26.3.1 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 400 |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| niah | — | 4131 | turbo4v2 | 3676.4 | 75.7 | 400 | 1125ms | — | 1.2219 | — | — | 34.30GB | 36.27GB | 122MB | 201MB | PASS(@10%): Thinking Process:  1.  **Analyze the Request:**  |
| niah | — | 4130 | turbo4v2 | 3817.8 | 75.8 | 400 | 1082ms | — | 1.2371 | — | — | 34.30GB | 36.27GB | 121MB | 201MB | FAIL(@25%): Thinking Process:  1.  **Analyze the Request:**  |
| niah | — | 4129 | turbo4v2 | 3794.8 | 75.8 | 400 | 1088ms | — | 1.1695 | — | — | 34.30GB | 36.27GB | 121MB | 201MB | PASS(@50%): Thinking Process:  1.  **Analyze the Request:**  |
| niah | — | 4131 | turbo4v2 | 3814.7 | 75.9 | 400 | 1083ms | — | 1.2190 | — | — | 34.30GB | 36.27GB | 121MB | 201MB | PASS(@75%): Thinking Process:  1.  **Analyze the Request:**  |
| niah | — | 4130 | turbo4v2 | 3802.8 | 76.0 | 400 | 1086ms | — | 1.2567 | — | — | 34.30GB | 36.27GB | 122MB | 201MB | FAIL(@90%): Thinking Process:  1.  **Analyze the Request:**  |
