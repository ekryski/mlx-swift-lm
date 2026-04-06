# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-06 11:56
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| summarization | 128 | 117 | turbo4v2 | 248.5 | 52.3 | 378 | 509ms | — | — | — | — | 18.16GB | 18.43GB | 38MB | 22MB | The user wants a summary of the provided text.  1.  **Identi |
| summarization | 256 | 249 | turbo4v2 | 336.4 | 52.1 | 400 | 741ms | — | — | — | — | 18.16GB | 18.69GB | 46MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4v2 | 409.7 | 51.7 | 400 | 1231ms | — | — | — | — | 18.16GB | 19.02GB | 44MB | 40MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | turbo4v2 | 528.6 | 51.0 | 400 | 2138ms | — | — | — | — | 18.16GB | 19.72GB | 50MB | 63MB | The user wants a summary of the provided text, which is the  |
| summarization | 2048 | 2042 | turbo4v2 | 582.9 | 52.9 | 400 | 3846ms | — | — | — | — | 18.16GB | 20.74GB | 62MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 543.8 | 52.5 | 400 | 7992ms | — | — | — | — | 18.16GB | 23.25GB | 121MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo4v2 | 554.4 | 47.3 | 400 | 15263ms | — | — | — | — | 18.16GB | 23.84GB | 190MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4v2 | 545.2 | 46.8 | 400 | 30429ms | — | — | — | — | 18.16GB | 24.96GB | 287MB | 745MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | turbo4v2 | 443.1 | 38.4 | 400 | 82485ms | — | — | — | — | 18.16GB | 27.16GB | 681MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4v2 | 367.8 | 33.9 | 400 | 178398ms | — | — | — | — | 18.16GB | 31.72GB | 1.22GB | 2.86GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo4v2 | 239.9 | 24.0 | 400 | 549478ms | — | — | — | — | 18.16GB | 39.65GB | 2.54GB | 5.69GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
