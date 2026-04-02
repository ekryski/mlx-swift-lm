# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 12:56
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
| summarization | 128 | 119 | turbo3 | 591.9 | 80.8 | 210 | 202ms | 2.6707 | 2.7924 | -0.0209 | 0.0258 | 1.86GB | 2.08GB | 16MB | This text appears to be the **first two pages of the introdu |
| summarization | 1024 | 1021 | turbo3 | 1046.2 | 82.7 | 388 | 976ms | 2.2807 | 2.8462 | 0.0132 | 0.0327 | 1.86GB | 3.27GB | 23MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 4096 | 4087 | turbo3 | 1241.8 | 76.6 | 400 | 3352ms | 2.7515 | 3.3155 | 0.0181 | 0.0388 | 1.86GB | 3.95GB | 63MB | This passage is the **prologue** to F. Scott Fitzgerald's *T |
| summarization | 32768 | 32702 | turbo3 | 1279.5 | 63.3 | 400 | 25908ms | 2.8761 | 2.4929 | 0.0417 | 0.0171 | 1.86GB | 5.00GB | 398MB | This text is the first draft of a **summary/outline** by an  |
