# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 15:43
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
| summarization | 128 | 119 | turbo4 | 601.5 | 79.7 | 201 | 199ms | 2.6393 | 1.5750 | 0.0201 | 0.0042 | 1.86GB | 2.08GB | 16MB | 19MB | The text provided is the **Preface to "Once Again to Zelda," |
| summarization | 1024 | 1021 | turbo4 | 1049.2 | 80.4 | 400 | 974ms | 2.4916 | 2.1916 | 0.0134 | 0.0309 | 1.86GB | 3.27GB | 23MB | 83MB | Here is a summary of the provided excerpt from F. Scott Fitz |
| summarization | 4096 | 4087 | turbo4 | 1221.0 | 77.8 | 201 | 3414ms | 3.2026 | 1.2919 | 0.0566 | 0.0519 | 1.86GB | 3.95GB | 60MB | 249MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 32768 | 32702 | turbo4 | 1281.8 | 63.1 | 400 | 25911ms | 3.3791 | 1.0184 | 0.0554 | 0.0013 | 1.86GB | 5.00GB | 398MB | 1.88GB | Here is a summary of *The Great Gatsby* based on the provide |
