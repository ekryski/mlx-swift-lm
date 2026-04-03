# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-02 22:48
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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
| summarization | 128 | 119 | turbo4 | 1159.2 | 89.4 | 400 | 104ms | 3.0136 | 4.7655 | 0.0243 | 0.0432 | 1.40GB | 1.61GB | 14MB | 30MB | Based on the text provided, it appears you have copied a spe |
| summarization | 256 | 251 | turbo4 | 2076.2 | 88.6 | 400 | 121ms | 4.0416 | 2.5815 | 0.0478 | 0.0127 | 1.40GB | 1.82GB | 8MB | 38MB | Based on the text provided from **"Once Again" by Zelda**, h |
| summarization | 512 | 506 | turbo4 | 2752.7 | 88.2 | 400 | 184ms | 4.6104 | 3.7883 | 0.0636 | 0.0283 | 1.40GB | 2.24GB | 22MB | 53MB | Based on the text provided, here is a summary of *The Time T |
| summarization | 1024 | 1021 | turbo4 | 3308.7 | 88.9 | 400 | 309ms | 4.2771 | 2.5414 | 0.0039 | 0.0248 | 1.40GB | 2.76GB | 22MB | 83MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 2048 | 2044 | turbo4 | 3574.4 | 87.6 | 400 | 572ms | 5.4402 | 3.4791 | 0.0804 | 0.0320 | 1.40GB | 3.51GB | 7MB | 142MB | **Summary of *The Great Gatsby* by F. Scott Fitzgerald**  In |
| summarization | 4096 | 4087 | turbo4 | 4097.5 | 86.0 | 400 | 998ms | 4.9261 | 4.4296 | 0.0194 | 0.0461 | 1.40GB | 3.43GB | 42MB | 261MB | Here is a summary of the excerpt from *The Great Gatsby* by  |
| summarization | 8192 | 8192 | turbo4 | 4348.7 | 83.1 | 400 | 1898ms | 3.7306 | 3.4716 | 0.0169 | -0.0049 | 1.40GB | 3.72GB | 74MB | 499MB | Based on the text provided, here is a summary of the charact |
| summarization | 16384 | 16363 | turbo4 | 4116.6 | 78.0 | 400 | 4037ms | 4.6747 | 3.0642 | 0.0219 | 0.0018 | 1.40GB | 3.98GB | 207MB | 974MB | Based on the text provided by F. Scott Fitzgerald from *The  |
| summarization | 32768 | 32702 | turbo4 | 3449.5 | 68.6 | 400 | 9897ms | 5.6098 | 4.3937 | 0.0111 | 0.0600 | 1.40GB | 4.82GB | 398MB | 1.88GB | Based on the provided text of *The Great Gatsby*, here is a  |
| summarization | 65536 | 65470 | turbo4 | 2225.4 | 52.0 | 400 | 29774ms | 4.6482 | 4.9434 | 0.0284 | 0.0235 | 1.40GB | 7.25GB | 784MB | 3.74GB | Based on the provided text, here is a summary of *The Great  |
| summarization | 131072 | 130775 | turbo4 | 1193.9 | 32.4 | 400 | 110990ms | 4.3583 | 3.5989 | 0.0377 | 0.0082 | 1.40GB | 8.29GB | 1.01GB | 7.44GB | Based on the text provided, here is a summary of **The Great |
