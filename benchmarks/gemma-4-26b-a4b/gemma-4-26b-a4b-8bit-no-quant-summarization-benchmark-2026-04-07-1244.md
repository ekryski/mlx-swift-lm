# Inference Benchmark - Gemma 4 26B A4B

**Date**: 2026-04-07 12:44
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
| summarization | 128 | 116 | no-quant | 118.9 | 23.8 | 400 | 977ms | 1.4938 | 1.2050 | — | — | 24.97GB | 26.04GB | 236MB | 113MB | <\|channel>thought  *   Title: *The Great Gatsby* *   Author: |
| summarization | 256 | 255 | no-quant | 437.1 | 23.6 | 400 | 584ms | 1.3489 | — | — | — | 24.97GB | 26.24GB | 256MB | 143MB | <\|channel>thought  *   Title: *The Great Gatsby* by F. Scott |
| summarization | 512 | 502 | no-quant | 453.0 | 23.3 | 400 | 1109ms | 1.3067 | — | — | — | 24.97GB | 26.58GB | 376MB | 197MB | <\|channel>thought  *   Source text provided: The title page, |
| summarization | 1024 | 1014 | no-quant | 548.2 | 23.0 | 400 | 2114ms | 1.4153 | — | — | — | 24.97GB | 27.45GB | 376MB | 309MB | <\|channel>thought  *   Text provided: The beginning of *The  |
| summarization | 2048 | 2037 | no-quant | 545.4 | 22.8 | 400 | 4436ms | 1.3824 | — | — | — | 24.97GB | 29.13GB | 440MB | 533MB | <\|channel>thought  *   Source material provided: The opening |
| summarization | 4096 | 4094 | no-quant | 588.8 | 22.3 | 400 | 7598ms | 1.3379 | — | — | — | 24.97GB | 29.94GB | 496MB | 983MB | <\|channel>thought  *   *Source Material:* The opening chapte |
| summarization | 8192 | 8229 | no-quant | 610.4 | 21.5 | 400 | 13570ms | 1.0000 | 1.2697 | — | — | 24.97GB | 28.98GB | 680MB | 1.84GB | <\|channel>thought <channel\|>This text comprises the complete |
| summarization | 16384 | 16395 | no-quant | 588.2 | 19.8 | 400 | 28097ms | 6.0378 | 1.2182 | — | — | 24.97GB | 30.00GB | 976MB | 3.59GB | <\|channel>�user <channel\|>The provided text contains the ope |
| summarization | 32768 | 32815 | no-quant | 518.4 | 15.8 | 400 | 63848ms | 1.1872 | 1.2798 | — | — | 24.97GB | 32.65GB | 1.59GB | 7.10GB | <\|channel>_thought <channel\|>The provided text contains the  |
| summarization | 65536 | 65896 | no-quant | 377.1 | 11.7 | 396 | 175451ms | 13.9828 | 1.2488 | — | — | 24.97GB | 38.21GB | 2.84GB | 14.16GB | <\|channel>� <channel\|>The provided text is the complete text |
| summarization | 131072 | 130563 | no-quant | 223.6 | 8.1 | 400 | 584909ms | 1.3799 | — | — | — | 24.97GB | 48.99GB | 5.36GB | 27.98GB | <\|channel>thought  *   Text 1: *The Great Gatsby* by F. Scot |
