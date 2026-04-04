# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 09:28
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
| summarization | 128 | 117 | no-quant | 29.3 | 43.5 | 400 | 4306ms | 1.2561 | 1.3045 | — | — | 34.30GB | 34.51GB | 21MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 204.2 | 43.3 | 400 | 1220ms | 1.2106 | 1.4034 | — | — | 34.30GB | 34.70GB | 40MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 334.8 | 42.9 | 389 | 1666ms | 1.2887 | 1.6123 | — | — | 34.30GB | 35.07GB | 52MB | 195MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 432.7 | 43.1 | 400 | 2643ms | 1.3642 | 1.4217 | — | — | 34.30GB | 35.78GB | 56MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 469.9 | 42.6 | 400 | 4968ms | 1.2399 | 1.6480 | — | — | 34.30GB | 36.86GB | 80MB | 534MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 519.3 | 42.0 | 400 | 8434ms | 1.2995 | 1.6767 | — | — | 34.30GB | 36.93GB | 121MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 548.8 | 40.7 | 400 | 15608ms | 1.3612 | 1.6492 | — | — | 34.30GB | 37.29GB | 201MB | 1.84GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 535.4 | 38.9 | 400 | 31115ms | 1.3139 | 1.6268 | — | — | 34.30GB | 37.88GB | 325MB | 3.58GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 497.0 | 35.7 | 400 | 66325ms | 1.6026 | 1.6736 | — | — | 34.30GB | 39.12GB | 681MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 394.5 | 30.1 | 400 | 166421ms | 1.2117 | 1.4772 | — | — | 34.30GB | 41.74GB | 1.29GB | 14.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 241.4 | 22.7 | 400 | 542053ms | 1.2596 | 1.6698 | — | — | 34.30GB | 46.43GB | 2.53GB | 28.02GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
