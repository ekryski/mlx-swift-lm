# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-08 20:03
**Branch**: `tom/turboquant-fixes`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |
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
| summarization | 128 | 107 | no-quant | 962.6 | 136.7 | 200 | 120ms | — | 1.3390 | — | — | 18.16GB | 18.41GB | 39MB | 67MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 239 | no-quant | 1681.2 | 136.9 | 200 | 143ms | — | 1.2513 | — | — | 18.16GB | 18.67GB | 41MB | 96MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 494 | no-quant | 2537.6 | 141.4 | 200 | 197ms | — | 1.2705 | — | — | 18.16GB | 19.01GB | 46MB | 152MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1009 | no-quant | 3469.8 | 140.4 | 200 | 297ms | — | 1.2616 | — | — | 18.16GB | 19.70GB | 55MB | 264MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2032 | no-quant | 4129.8 | 139.7 | 200 | 510ms | — | 1.3420 | — | — | 18.16GB | 20.73GB | 76MB | 488MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4075 | no-quant | 4327.9 | 135.6 | 200 | 974ms | — | 1.4776 | — | — | 18.16GB | 23.24GB | 116MB | 935MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8180 | no-quant | 4369.2 | 129.1 | 200 | 1903ms | — | 1.2283 | — | — | 18.16GB | 23.83GB | 197MB | 1.79GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16351 | no-quant | 3945.3 | 124.2 | 200 | 4213ms | — | 1.3082 | — | — | 18.16GB | 24.95GB | 357MB | 3.54GB | Here's a thinking process that leads to the suggested summar |
| summarization | 32768 | 32690 | no-quant | 3310.3 | 112.8 | 200 | 9967ms | — | 1.4918 | — | — | 18.16GB | 27.14GB | 676MB | 7.03GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
