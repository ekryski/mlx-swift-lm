# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 16:21
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 8bit
**Model**: `mlx-community/gemma-4-26b-a4b-it-8bit`

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
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | no-quant | 125.9 | 23.7 | 400 | 923ms | 1.3651 | — | — | — | 24.97GB | 26.04GB | 231MB | 113MB | <\|channel>thought  *   Title: *The Great Gatsby* *   Author: |
| summarization | 1024 | 1014 | no-quant | 555.8 | 23.0 | 400 | 2137ms | 1.2603 | — | — | — | 24.97GB | 27.45GB | 344MB | 309MB | <\|channel>thought  *   Text: The opening pages of *The Great |
| summarization | 4096 | 4094 | no-quant | 585.1 | 22.3 | 400 | 7638ms | 1.2833 | — | — | — | 24.97GB | 29.94GB | 528MB | 983MB | <\|channel>thought  *   Source text: The opening of F. Scott  |
| summarization | 32768 | 32815 | no-quant | 519.6 | 15.8 | 400 | 63513ms | 1.0000 | 1.2640 | — | — | 24.97GB | 32.65GB | 1.57GB | 7.10GB | <\|channel>thought <channel\|>The provided text contains the f |
