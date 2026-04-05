# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 17:50
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| summarization | 128 | 117 | turbo4v2 | 204.6 | 52.4 | 371 | 573ms | 1.5298 | 1.5567 | 0.1032 | 0.0110 | 18.16GB | 18.43GB | 42MB | 22MB | The user has provided the title page and dedication of F. Sc |
| summarization | 256 | 249 | turbo4v2 | 46.0 | 53.1 | 400 | 5899ms | 1.6839 | 1.6824 | 0.1138 | 0.0553 | 18.16GB | 18.60GB | 24MB | 29MB | The user wants a summary of the text provided. The text is t |
| summarization | 512 | 504 | turbo4v2 | 32.4 | 51.8 | 204 | 15971ms | 1.2768 | 2.0577 | 0.0316 | 0.3595 | 18.16GB | 18.94GB | 46MB | 31MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 70.6 | 52.3 | 400 | 14864ms | 1.3984 | 1.5173 | 0.0890 | 0.0762 | 18.16GB | 19.65GB | 37MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4v2 | 130.9 | 51.8 | 400 | 16036ms | 1.3269 | 1.6133 | 0.0996 | 0.1226 | 18.16GB | 20.72GB | 80MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo4v2 | 196.0 | 50.7 | 400 | 21296ms | 1.3845 | 1.9006 | 0.0540 | 0.0680 | 18.16GB | 20.80GB | 72MB | 199MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | turbo4v2 | 305.6 | 48.5 | 400 | 27261ms | 1.6364 | 1.2970 | 0.1123 | 0.0759 | 18.16GB | 21.15GB | 201MB | 382MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4v2 | 384.8 | 44.9 | 399 | 42973ms | 1.3897 | 1.4950 | 0.0154 | 0.0984 | 18.16GB | 21.75GB | 288MB | 745MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | turbo4v2 | 423.7 | 40.0 | 400 | 77668ms | 1.5476 | 1.6265 | 0.1029 | 0.0490 | 18.16GB | 22.98GB | 409MB | 1.44GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4v2 | 368.5 | 33.7 | 395 | 178035ms | 1.3733 | 1.5109 | 0.0575 | 0.0567 | 18.16GB | 25.60GB | 1.29GB | 2.86GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | turbo4v2 | 247.3 | 25.3 | 400 | 529175ms | 1.2103 | 1.4185 | 0.1025 | 0.0487 | 18.16GB | 30.29GB | 2.53GB | 5.69GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
