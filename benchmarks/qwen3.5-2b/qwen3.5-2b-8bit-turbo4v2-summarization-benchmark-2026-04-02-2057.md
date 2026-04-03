# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 20:57
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
| summarization | 128 | 119 | turbo4v2 | 605.3 | 79.6 | 400 | 198ms | 2.5790 | 2.1728 | 0.0338 | 0.0108 | 1.86GB | 2.08GB | 13MB | 23MB | Based on the text snippet you provided, there appears to be  |
| summarization | 1024 | 1021 | turbo4v2 | 1025.5 | 80.5 | 201 | 996ms | 2.8315 | 1.0682 | -0.0167 | 0.1345 | 1.86GB | 3.27GB | 15MB | 54MB | Based on the text provided, here is a summary of the content |
| summarization | 4096 | 4087 | turbo4v2 | 1216.7 | 77.3 | 201 | 3429ms | 3.0890 | 1.5188 | 0.0199 | 0.0902 | 1.86GB | 3.95GB | 35MB | 191MB | Here is a summary of the text provided, which is a prologue  |
| summarization | 32768 | 32702 | turbo4v2 | 1273.4 | 62.2 | 368 | 26048ms | 2.9184 | 3.1083 | 0.0194 | 0.0577 | 1.86GB | 5.00GB | 399MB | 1.43GB | This text is a selection of short story segments from F. Sco |
