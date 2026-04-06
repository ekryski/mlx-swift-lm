# Inference Benchmark - Nemotron 30B A3B

**Date**: 2026-04-06 01:54
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 8bit
**Model**: `mlx-community/Nemotron-Cascade-2-30B-A3B-8bit`

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
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
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
| summarization | 128 | 135 | no-quant | 56.0 | 26.0 | 332 | 2763ms | 1.5982 | 1.2979 | — | — | 31.26GB | 32.44GB | 50MB | 102MB | We need to summarize the content above. The content includes |
| summarization | 1024 | 1049 | no-quant | 457.4 | 25.8 | 400 | 2871ms | 1.6829 | 1.7027 | — | — | 31.26GB | 40.10GB | 62MB | 317MB | We need to summarize the content above, which presumably is  |
| summarization | 4096 | 4110 | no-quant | 108.7 | 25.6 | 400 | 38487ms | 1.6504 | 1.3531 | — | — | 31.26GB | 58.84GB | 98MB | 987MB | We need to provide a summary of the content above. The user  |
| summarization | 32768 | 32553 | no-quant | 117.6 | 23.6 | 400 | 286351ms | 1.7583 | 1.4970 | — | — | 31.26GB | 59.14GB | 434MB | 7.04GB | We need to provide a summary of the given text. It's a long  |
