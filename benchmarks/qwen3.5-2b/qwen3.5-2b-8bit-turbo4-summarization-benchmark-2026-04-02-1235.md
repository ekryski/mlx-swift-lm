# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 12:35
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
| summarization | 128 | 119 | turbo4 | 597.1 | 79.5 | 201 | 201ms | 2.8579 | 1.4879 | 0.0389 | -0.0122 | 1.86GB | 2.08GB | 16MB | The text you've shared appears to be an excerpt from **"Once |
| summarization | 1024 | 1021 | turbo4 | 1026.3 | 78.1 | 211 | 998ms | 2.1038 | 2.8253 | 0.0254 | 0.1482 | 1.86GB | 3.27GB | 25MB | Based on the excerpt from F. Scott Fitzgerald's *The Great G |
| summarization | 4096 | 4087 | turbo4 | 1239.3 | 77.2 | 201 | 3363ms | 2.7670 | 1.3667 | 0.0461 | 0.0242 | 1.86GB | 3.95GB | 60MB | Here is a summary of the provided text, which combines *The  |
| summarization | 32768 | 32702 | turbo4 | 1275.3 | 63.1 | 400 | 25992ms | 2.7945 | 3.0319 | 0.0289 | 0.0342 | 1.86GB | 5.00GB | 267MB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
