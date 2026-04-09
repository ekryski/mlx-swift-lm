# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-08 20:02
**Branch**: `tom/turboquant-fixes`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |
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
| summarization | 128 | 107 | turbo4v2 | 880.7 | 133.4 | 200 | 125ms | — | 1.3425 | — | — | 18.16GB | 18.41GB | 41MB | 14MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 239 | turbo4v2 | 1681.0 | 144.7 | 200 | 143ms | — | 1.2698 | — | — | 18.16GB | 18.67GB | 42MB | 20MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 494 | turbo4v2 | 2553.5 | 142.1 | 200 | 195ms | — | 1.3628 | — | — | 18.16GB | 19.01GB | 45MB | 31MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1009 | turbo4v2 | 3566.0 | 142.3 | 200 | 294ms | — | 1.2632 | — | — | 18.16GB | 19.70GB | 56MB | 54MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2032 | turbo4v2 | 4172.6 | 140.2 | 200 | 510ms | — | 1.3431 | — | — | 18.16GB | 20.73GB | 76MB | 99MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | turbo4v2 | 4326.9 | 136.9 | 200 | 996ms | — | 1.2767 | — | — | 18.16GB | 23.24GB | 116MB | 190MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8180 | turbo4v2 | 4383.7 | 130.5 | 200 | 1909ms | — | 1.4365 | — | — | 18.16GB | 23.83GB | 196MB | 372MB | Here's a thinking process that leads to the suggested summar |
| summarization | 16384 | 16351 | turbo4v2 | 3841.8 | 114.9 | 200 | 4292ms | — | 1.3679 | — | — | 18.16GB | 24.95GB | 355MB | 735MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32690 | turbo4v2 | 3229.7 | 111.3 | 200 | 10240ms | — | 1.2068 | — | — | 18.16GB | 27.14GB | 676MB | 1.43GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
