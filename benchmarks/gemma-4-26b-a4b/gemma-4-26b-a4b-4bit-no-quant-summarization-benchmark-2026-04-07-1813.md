# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 18:13
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`

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
| summarization | 128 | 116 | no-quant | 234.9 | 27.5 | 400 | 495ms | 1.3008 | 1.2106 | 0.6128 | 0.1521 | 13.48GB | 14.13GB | 236MB | 113MB | <\|channel>thought  *   Text provided:         *   Title: *Th |
| summarization | 1024 | 1014 | no-quant | 185.9 | 26.5 | 400 | 5790ms | 1.3409 | — | 0.6312 | — | 13.48GB | 15.95GB | 368MB | 309MB | <\|channel>thought  *   Text: An excerpt from the beginning o |
| summarization | 4096 | 4094 | no-quant | 483.5 | 25.4 | 400 | 8714ms | 1.3500 | — | 0.6353 | — | 13.48GB | 18.43GB | 528MB | 983MB | <\|channel>thought  *   Text: The first chapter of F. Scott F |
| summarization | 32768 | 32815 | no-quant | 522.2 | 17.0 | 400 | 63315ms | 1.0000 | 1.2895 | 2.6122 | 0.4128 | 13.48GB | 21.16GB | 1.58GB | 7.10GB | <\|channel>thought <channel\|>The provided text constitutes th |
