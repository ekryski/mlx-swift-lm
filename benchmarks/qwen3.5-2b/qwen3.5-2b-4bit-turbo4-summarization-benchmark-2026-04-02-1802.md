# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 18:02
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-2B-4bit`

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
| summarization | 128 | 119 | turbo4 | 611.8 | 91.2 | 400 | 196ms | 2.7628 | 2.5426 | 0.1167 | 0.1609 | 1010MB | 1.21GB | 19MB | 30MB | The text you provided appears to be a thematic excerpt and t |
| summarization | 256 | 251 | turbo4 | 868.2 | 93.2 | 400 | 290ms | 3.9513 | 1.5704 | 0.2091 | 0.0348 | 1010MB | 1.43GB | 14MB | 38MB | The text you provided is the opening chapters of **"This Sid |
| summarization | 512 | 506 | turbo4 | 951.0 | 93.5 | 400 | 533ms | 3.0369 | 2.9373 | 0.1372 | 0.0497 | 1010MB | 1.88GB | 11MB | 53MB | This passage is a **letter from Thomas Parke d'Invilliers to |
| summarization | 1024 | 1021 | turbo4 | 1031.8 | 93.6 | 400 | 990ms | 3.0630 | 1.0985 | 0.1591 | 0.0228 | 1010MB | 2.48GB | 22MB | 83MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 2048 | 2044 | turbo4 | 1084.3 | 91.2 | 273 | 1930ms | 3.5398 | 4.0748 | 0.2157 | 0.4366 | 1010MB | 3.18GB | 39MB | 135MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 4096 | 4087 | turbo4 | 1211.5 | 91.5 | 400 | 3426ms | 2.5593 | 1.7701 | 0.1186 | 0.0745 | 1010MB | 3.14GB | 52MB | 261MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* o |
| summarization | 8192 | 8192 | turbo4 | 1310.6 | 90.3 | 400 | 6294ms | 3.1910 | 3.5684 | 0.1644 | 0.2580 | 1010MB | 3.34GB | 75MB | 499MB | Here is a summary of the text provided:  **Title:** The Grea |
| summarization | 16384 | 16363 | turbo4 | 1325.7 | 84.4 | 201 | 12492ms | 2.8739 | 1.4464 | 0.1913 | -0.2167 | 1010MB | 3.62GB | 204MB | 962MB | This is a detailed summary of the provided text from *The Gr |
| summarization | 32768 | 32702 | turbo4 | 1280.8 | 75.1 | 400 | 25934ms | 2.9159 | 1.0119 | 0.1899 | -0.0010 | 1010MB | 4.18GB | 398MB | 1.88GB | Here is a summary of the novel **"The Great Gatsby"** by F.  |
| summarization | 65536 | 65470 | turbo4 | 1025.1 | 44.9 | 400 | 70070ms | 2.9821 | 1.2222 | 0.2483 | 0.0107 | 1010MB | 5.47GB | 783MB | 3.74GB | The provided text is the complete novella **"The Great Gatsb |
| summarization | 131072 | 130775 | turbo4 | 760.2 | 35.2 | 400 | 173205ms | 3.0935 | 3.0044 | 0.2514 | 0.1309 | 1011MB | 7.82GB | 1.51GB | 7.44GB | Based on the text provided, here is a summary of the narrati |
