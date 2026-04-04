# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 10:57
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-8bit`

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
| summarization | 128 | 117 | turbo4 | 45.2 | 43.9 | 267 | 2781ms | 1.2070 | 1.4845 | -0.0056 | -0.0014 | 34.30GB | 34.51GB | 41MB | 22MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4 | 3.9 | 44.3 | 400 | 65062ms | 1.3789 | 1.7120 | -0.0069 | 0.0053 | 34.30GB | 34.70GB | 46MB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4 | 8.0 | 42.9 | 400 | 63378ms | 1.2290 | 1.5509 | 0.0270 | 0.0231 | 34.30GB | 35.07GB | 45MB | 53MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4 | 16.3 | 42.8 | 400 | 62831ms | 1.3171 | 1.6114 | -0.0122 | 0.0110 | 34.30GB | 35.78GB | 19MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 32.3 | 42.5 | 400 | 63606ms | 1.3181 | 1.6403 | 0.0030 | 0.0096 | 34.30GB | 36.86GB | 80MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4 | 56.2 | 42.0 | 400 | 73171ms | 1.2932 | 1.6442 | 0.0122 | 0.0313 | 34.30GB | 36.93GB | 70MB | 261MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4 | 115.3 | 40.6 | 400 | 71472ms | 1.4761 | 1.8401 | 0.0223 | 0.0024 | 34.30GB | 37.29GB | 140MB | 499MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
