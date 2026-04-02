# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 12:28
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
| summarization | 128 | 119 | turbo4 | 611.9 | 79.9 | 201 | 196ms | 3.6103 | 1.0876 | 0.0417 | 0.1587 | 1.86GB | 2.08GB | 15MB | The text you provided appears to be a section from an early  |
| summarization | 1024 | 1021 | turbo4 | 1016.4 | 79.4 | 400 | 1006ms | 3.6719 | 2.2888 | 0.0091 | 0.0254 | 1.86GB | 3.27GB | 20MB | F. Scott Fitzgerald's first chapter of *The Great Gatsby* in |
| summarization | 4096 | 4087 | turbo4 | 1232.9 | 77.6 | 400 | 3371ms | 3.7037 | 1.0201 | 0.0510 | 0.0006 | 1.86GB | 3.95GB | 53MB | This text is an excerpt from **Chapter 1 of F. Scott Fitzger |
| summarization | 32768 | 32702 | turbo4 | 1273.3 | 63.1 | 400 | 26020ms | 3.2010 | 1.0076 | 0.0346 | 0.0001 | 1.86GB | 5.00GB | 333MB | This text is a collection of excerpts from **F. Scott Fitzge |
