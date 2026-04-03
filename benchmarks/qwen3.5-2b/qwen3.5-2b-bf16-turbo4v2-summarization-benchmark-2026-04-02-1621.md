# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 16:21
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-2B-bf16`

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
| summarization | 128 | 119 | turbo4v2 | 665.8 | 55.4 | 400 | 180ms | 3.4594 | 2.1363 | 0.0135 | 0.0046 | 3.51GB | 3.75GB | 18MB | 23MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby*,  |
| summarization | 256 | 251 | turbo4v2 | 1098.0 | 54.3 | 238 | 229ms | 2.7387 | 2.9640 | -0.0008 | 0.0687 | 3.51GB | 3.94GB | 14MB | 22MB | The text you provided offers two distinct literary works.  1 |
| summarization | 512 | 506 | turbo4v2 | 1409.7 | 55.6 | 400 | 359ms | 2.6292 | 2.7479 | 0.0399 | 0.0349 | 3.51GB | 4.15GB | 21MB | 40MB | In this passage, F. Scott Fitzgerald recounts how his father |
| summarization | 1024 | 1021 | turbo4v2 | 1657.7 | 56.3 | 201 | 616ms | 2.6536 | 1.8252 | 0.0342 | 0.0155 | 3.51GB | 4.68GB | 25MB | 54MB | **Summary of Selected Texts from "The Great Gatsby"**  This  |
| summarization | 2048 | 2044 | turbo4v2 | 1749.2 | 55.8 | 201 | 1182ms | 2.9660 | 1.1215 | 0.0108 | 0.0002 | 3.51GB | 5.50GB | 35MB | 100MB | This excerpt from **Chapter I of *The Great Gatsby*** introd |
| summarization | 4096 | 4087 | turbo4v2 | 2026.1 | 54.8 | 201 | 2025ms | 2.5623 | 1.4238 | 0.0271 | 0.4255 | 3.51GB | 5.41GB | 59MB | 191MB | Based on the text provided, here is a summary of the content |
| summarization | 8192 | 8192 | turbo4v2 | 2169.6 | 53.6 | 274 | 3787ms | 3.1656 | 2.5404 | 0.0276 | 0.0020 | 3.51GB | 5.60GB | 111MB | 376MB | Here is a summary of the provided text:  The narrative begin |
| summarization | 16384 | 16363 | turbo4v2 | 2126.9 | 51.2 | 201 | 7782ms | 2.8748 | 1.5863 | 0.0520 | 0.0078 | 3.51GB | 5.89GB | 187MB | 736MB | Here is a summary of *The Great Gatsby* based on the text pr |
| summarization | 32768 | 32702 | turbo4v2 | 1975.4 | 47.4 | 201 | 16881ms | 3.0539 | 1.1632 | 0.0221 | -0.0002 | 3.51GB | 6.51GB | 331MB | 1.43GB | This summary outlines the key events, characters, and themat |
| summarization | 65536 | 65470 | turbo4v2 | 1293.5 | 39.0 | 201 | 60747ms | 2.5379 | 2.4857 | 0.0430 | 0.0002 | 3.51GB | 7.84GB | 780MB | 2.85GB | This story, titled "The Great Gatsby" by F. Scott Fitzgerald |
| summarization | 131072 | 130775 | turbo4v2 | 892.6 | 30.4 | 201 | 149915ms | 2.6073 | 1.9511 | 0.0303 | 0.0001 | 3.51GB | 10.22GB | 1.51GB | 5.68GB | The text contains two distinct short stories: **"The Great G |
