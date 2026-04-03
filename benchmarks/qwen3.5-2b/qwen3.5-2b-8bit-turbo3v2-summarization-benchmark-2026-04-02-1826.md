# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 18:26
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
| summarization | 128 | 119 | turbo3v2 | 609.6 | 83.4 | 400 | 197ms | 2.7477 | 2.3120 | -0.0048 | 0.0262 | 1.86GB | 2.08GB | 16MB | 20MB | The text you provided is **not a summary of the novel *The G |
| summarization | 256 | 251 | turbo3v2 | 858.1 | 80.1 | 201 | 293ms | 3.1341 | 4.6257 | 0.0484 | -0.2924 | 1.86GB | 2.31GB | 13MB | 17MB | The provided text excerpt is the **opening chapter of *The G |
| summarization | 512 | 506 | turbo3v2 | 979.3 | 81.8 | 348 | 517ms | 2.5470 | 4.1101 | 0.0203 | 0.0172 | 1.86GB | 2.71GB | 19MB | 32MB | This passage is a reflection on **F. Scott Fitzgerald's phil |
| summarization | 1024 | 1021 | turbo3v2 | 1056.6 | 82.1 | 400 | 967ms | 2.8255 | 3.2348 | -0.0149 | 0.0038 | 1.86GB | 3.27GB | 23MB | 53MB | The text provided is **F. Scott Fitzgerald's** prologue to * |
| summarization | 2048 | 2044 | turbo3v2 | 1091.3 | 80.8 | 201 | 1896ms | 2.7806 | 1.3915 | -0.0025 | 0.2189 | 1.86GB | 3.97GB | 37MB | 84MB | Based on the text provided, here is a summary of the content |
| summarization | 4096 | 4087 | turbo3v2 | 1257.4 | 80.2 | 400 | 3297ms | 2.4497 | 3.1538 | 0.0013 | 0.0284 | 1.86GB | 3.95GB | 63MB | 169MB | **Book: The Great Gatsby (Chapter I) - Summary**  **1. Autho |
| summarization | 8192 | 8192 | turbo3v2 | 1349.9 | 78.1 | 222 | 6113ms | 3.2375 | 2.1352 | 0.0152 | -0.0088 | 1.86GB | 4.04GB | 108MB | 316MB | Here is a summary of **Chapter One: Once Again to Zelda**, w |
| summarization | 16384 | 16363 | turbo3v2 | 1364.9 | 72.9 | 400 | 12103ms | 2.6847 | 4.0713 | 0.0371 | 0.0245 | 1.86GB | 4.39GB | 207MB | 630MB | Here is a summary of *The Great Gatsby*, based on the provid |
| summarization | 32768 | 32702 | turbo3v2 | 1306.4 | 64.6 | 400 | 25358ms | 2.6596 | 1.2124 | 0.0225 | -0.0030 | 1.86GB | 5.00GB | 333MB | 1.22GB | **The Great Gatsby (1920s American Novel) Summary**  Set dur |
| summarization | 65536 | 65470 | turbo3v2 | 1110.4 | 40.5 | 400 | 61168ms | 3.8542 | 2.5379 | 0.0201 | 0.0593 | 1.86GB | 6.34GB | 783MB | 2.42GB | Here is a summary of the narrative provided in "The Great Ga |
| summarization | 131072 | 130775 | turbo3v2 | 835.2 | 37.8 | 400 | 156975ms | 3.1729 | 2.7351 | 0.0090 | 0.0074 | 1.86GB | 8.63GB | 1.51GB | 4.82GB | Here is a summary of *The Age of Innocence* by Edith Wharton |
