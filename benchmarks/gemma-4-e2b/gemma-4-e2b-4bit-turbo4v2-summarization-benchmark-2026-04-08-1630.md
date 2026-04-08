# Inference Benchmark - Gemma 4 E2B

**Date**: 2026-04-08 16:30
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 108GB |
| macOS | 26.3.1 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 200 |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 110 | turbo4v2 | 59.8 | 163.4 | 197 | 1846ms | — | 1.9141 | — | — | 2.45GB | 2.91GB | 8MB | 14MB | The provided text appears to be a fragmented collection of e |
| summarization | 256 | 249 | turbo4v2 | 4755.7 | 151.4 | 200 | 53ms | — | 1.4531 | — | — | 2.45GB | 3.28GB | 13MB | 20MB | The provided text is a fragmented excerpt from **F. Scott Fi |
| summarization | 512 | 496 | turbo4v2 | 1139.9 | 149.9 | 200 | 437ms | — | 1.3104 | — | — | 2.45GB | 3.92GB | 16MB | 31MB | This excerpt comes from **The Great Gatsby** by F. Scott Fit |
| summarization | 1024 | 1008 | turbo4v2 | 2820.5 | 145.8 | 200 | 363ms | — | 1.5509 | — | — | 2.45GB | 4.76GB | 29MB | 54MB | This excerpt from *The Great Gatsby* features several distin |
| summarization | 2048 | 2031 | turbo4v2 | 5326.8 | 155.3 | 200 | 409ms | — | 1.3673 | — | — | 2.45GB | 5.81GB | 42MB | 99MB | This excerpt appears to be from **The Great Gatsby** by F. S |
| summarization | 4096 | 4088 | turbo4v2 | 5968.6 | 152.9 | 200 | 719ms | — | 1.4358 | — | — | 2.45GB | 5.53GB | 76MB | 191MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 8192 | 8192 | turbo4v2 | 10600.3 | 145.6 | 200 | 813ms | — | 1.5699 | — | — | 2.45GB | 5.84GB | 140MB | 373MB | This is a fascinating and dense excerpt from **Nick Carraway |
| summarization | 16384 | 16384 | turbo4v2 | 11998.1 | 131.5 | 200 | 1407ms | — | 1.6156 | — | — | 2.45GB | 6.45GB | 266MB | 737MB | This is an excerpt from **The Great Gatsby** by F. Scott Fit |
| summarization | 32768 | 32768 | turbo4v2 | 10243.1 | 114.3 | 200 | 3223ms | — | 1.5451 | — | — | 2.45GB | 7.67GB | 524MB | 1.43GB | This is a rich and complex excerpt from **F. Scott Fitzgeral |
| summarization | 65536 | 65536 | turbo4v2 | 2901.1 | 87.5 | 200 | 22755ms | — | 1.8729 | — | — | 2.45GB | 10.80GB | 1.01GB | 2.85GB | This is a substantial excerpt from **F. Scott Fitzgerald's * |
| summarization | 131072 | 130557 | turbo4v2 | 1947.3 | 71.8 | 200 | 67125ms | — | 1.9930 | — | — | 2.45GB | 14.58GB | 1.26GB | 5.67GB | This is a highly complex and dense collection of excerpts, s |
