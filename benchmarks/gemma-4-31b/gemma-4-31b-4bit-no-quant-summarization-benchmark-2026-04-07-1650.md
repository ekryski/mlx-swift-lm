# Inference Benchmark - Gemma 4 31B

**Date**: 2026-04-07 16:50
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-31b-it-4bit`

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
| summarization | 128 | 116 | no-quant | 58.6 | 13.1 | 299 | 2131ms | 1.2625 | 1.0633 | 0.5703 | 0.5197 | 16.09GB | 16.65GB | 461MB | 91MB | <\|channel>thought  *   Content: A snippet of the beginning o |
| summarization | 1024 | 1014 | no-quant | 68.2 | 12.8 | 400 | 15321ms | 1.2285 | — | 0.3593 | — | 16.09GB | 18.88GB | 824MB | 309MB | <\|channel>thought  *   Source material: The opening pages of |
| summarization | 4096 | 4094 | no-quant | 67.6 | 12.1 | 400 | 61314ms | 1.2454 | — | 0.3272 | — | 16.09GB | 22.66GB | 1.04GB | 983MB | <\|channel>thought *   Source material: The opening section ( |
| summarization | 32768 | 32815 | no-quant | 66.1 | 9.2 | 400 | 497157ms | 1.0003 | 1.1793 | -0.0001 | 0.1972 | 16.09GB | 25.87GB | 3.25GB | 7.10GB | <\|channel>thought <channel\|>The provided text comprises the  |
