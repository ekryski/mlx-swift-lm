# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 12:08
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
| summarization | 128 | 117 | turbo3 | 42.2 | 43.2 | 355 | 3009ms | 1.3317 | 1.7538 | 0.0314 | 0.0323 | 34.30GB | 34.51GB | 41MB | 21MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo3 | 3.8 | 43.8 | 400 | 65571ms | 1.2083 | 1.6070 | 0.0132 | 0.0579 | 34.30GB | 34.70GB | 46MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 7.7 | 43.4 | 400 | 66014ms | 1.2666 | 1.6244 | 0.0254 | 0.0144 | 34.30GB | 35.07GB | 52MB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo3 | 15.9 | 43.2 | 400 | 64360ms | 1.2810 | 1.3396 | 0.0028 | 0.0100 | 34.30GB | 35.78GB | 42MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo3 | 29.9 | 43.1 | 400 | 68649ms | 1.3834 | 1.5440 | 0.0320 | 0.0048 | 34.30GB | 36.86GB | 55MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo3 | 54.3 | 42.3 | 400 | 75624ms | 1.2165 | 1.5305 | 0.0120 | 0.0137 | 34.30GB | 36.93GB | 96MB | 199MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | turbo3 | 106.6 | 40.7 | 400 | 77299ms | 1.5066 | 1.5073 | 0.0069 | 0.0121 | 34.30GB | 37.29GB | 201MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo3 | 181.9 | 38.8 | 400 | 90393ms | 1.3287 | 1.5130 | 0.0314 | 0.0143 | 34.30GB | 37.88GB | 251MB | 745MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo3 | 263.1 | 35.7 | 400 | 124797ms | 1.4351 | 1.7353 | 0.0083 | 0.0547 | 34.30GB | 39.12GB | 614MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo3 | 294.2 | 30.1 | 400 | 223061ms | 1.1550 | 1.5624 | 0.0039 | 0.0186 | 34.30GB | 41.74GB | 1.16GB | 2.86GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo3 | 215.7 | 22.4 | 400 | 606702ms | 1.2675 | 1.4293 | 0.0202 | 0.0054 | 34.30GB | 46.43GB | 1.78GB | 5.69GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
