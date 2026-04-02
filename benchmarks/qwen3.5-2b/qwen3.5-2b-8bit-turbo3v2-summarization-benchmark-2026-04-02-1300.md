# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 13:00
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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: GPU memory increase from the KV cache after generation; for KV-quantized runs this reflects the compressed cache size.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|--------|
| summarization | 128 | 119 | turbo3v2 | 614.0 | 81.0 | 201 | 195ms | 3.1555 | 1.6263 | 0.0524 | 0.0376 | 1.86GB | 2.08GB | 9MB | Based on the text provided, here is a summary of the content |
| summarization | 1024 | 1021 | turbo3v2 | 1013.1 | 78.8 | 400 | 1010ms | 2.9883 | 3.3147 | 0.0117 | 0.0203 | 1.86GB | 3.27GB | 26MB | The provided text is an excerpt from **Chapter 1** of *The G |
| summarization | 4096 | 4087 | turbo3v2 | 1221.1 | 77.0 | 400 | 3396ms | 3.4762 | 2.8729 | 0.0452 | 0.0001 | 1.86GB | 3.95GB | 63MB | Here is a summary of the text provided:  The passage begins  |
| summarization | 32768 | 32702 | turbo3v2 | 1275.2 | 63.3 | 400 | 26004ms | 3.0356 | 1.0252 | 0.0201 | 0.0066 | 1.86GB | 5.00GB | 399MB | This text is the beginning of **The Great Gatsby** by F. Sco |
