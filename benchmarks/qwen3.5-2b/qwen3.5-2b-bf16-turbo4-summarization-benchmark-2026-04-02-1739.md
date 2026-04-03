# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 17:39
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
| summarization | 128 | 119 | turbo4 | 718.3 | 55.8 | 400 | 167ms | 3.5645 | 4.3056 | 0.0384 | 0.0355 | 3.51GB | 3.75GB | 8MB | 30MB | The text you've provided is a brief selection from the prolo |
| summarization | 256 | 251 | turbo4 | 1117.7 | 54.6 | 400 | 225ms | 2.2345 | 2.6922 | 0.0432 | 0.0306 | 3.51GB | 3.94GB | 18MB | 38MB | The provided text consists of two distinct parts with differ |
| summarization | 512 | 506 | turbo4 | 1404.1 | 55.3 | 201 | 361ms | 2.0771 | 1.8171 | 0.0144 | 0.0375 | 3.51GB | 4.15GB | 13MB | 41MB | This excerpt, primarily drawn from the opening chapter of *T |
| summarization | 1024 | 1021 | turbo4 | 1662.1 | 56.4 | 400 | 615ms | 4.1368 | 2.4883 | 0.0278 | 0.0252 | 3.51GB | 4.68GB | 27MB | 83MB | F. Scott Fitzgerald's *The Great Gatsby* opens with F. Scott |
| summarization | 2048 | 2044 | turbo4 | 1761.1 | 55.8 | 201 | 1161ms | 2.7776 | 1.2902 | 0.0190 | 0.0029 | 3.51GB | 5.50GB | 36MB | 130MB | Based on the provided text from F. Scott Fitzgerald's "The G |
| summarization | 4096 | 4087 | turbo4 | 2042.3 | 55.5 | 400 | 2022ms | 2.7383 | 2.5467 | 0.0178 | 0.0413 | 3.51GB | 5.41GB | 63MB | 261MB | Based on the text provided, here is a summary organized by t |
| summarization | 8192 | 8192 | turbo4 | 2183.2 | 54.0 | 400 | 3763ms | 2.5322 | 3.0527 | 0.0364 | 0.0534 | 3.51GB | 5.60GB | 112MB | 499MB | This text is an excerpt from *The Great Gatsby* by F. Scott  |
| summarization | 16384 | 16363 | turbo4 | 2163.2 | 51.1 | 380 | 7646ms | 3.2099 | 1.4520 | 0.0399 | 0.0028 | 3.51GB | 5.89GB | 207MB | 973MB | Here is a summary of *The Great Gatsby*, based on the provid |
| summarization | 32768 | 32702 | turbo4 | 1980.2 | 47.1 | 400 | 16834ms | 3.0596 | 1.0063 | 0.0082 | -0.0000 | 3.51GB | 6.51GB | 399MB | 1.88GB | **Note:** You provided the text from "The Great Gatsby" by F |
| summarization | 65536 | 65470 | turbo4 | 1362.2 | 34.2 | 400 | 54143ms | 2.8308 | 1.5867 | 0.0070 | -0.0068 | 3.51GB | 7.90GB | 783MB | 3.74GB | Here is a summary of F. Scott Fitzgerald's novel ***The Grea |
| summarization | 131072 | 130775 | turbo4 | 980.7 | 30.1 | 400 | 134573ms | 2.8483 | 2.6150 | 0.0466 | 0.0393 | 3.51GB | 10.22GB | 1.51GB | 7.44GB | The following is a summary of the narrative provided, focusi |
