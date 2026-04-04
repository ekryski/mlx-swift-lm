# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 04:17
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-4B-bf16`

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
| summarization | 128 | 117 | no-quant | 393.9 | 29.5 | 400 | 298ms | 1.7869 | 2.2612 | — | — | 7.83GB | 8.11GB | 47MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 653.1 | 29.3 | 400 | 382ms | 1.3586 | 1.6689 | — | — | 7.83GB | 8.21GB | 42MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 718.7 | 29.5 | 400 | 702ms | 1.6613 | 1.6992 | — | — | 7.83GB | 8.55GB | 55MB | 198MB | The user wants a summary of the provided text. The text cons |
| summarization | 1024 | 1019 | no-quant | 787.4 | 29.3 | 400 | 1329ms | 1.4104 | 1.7774 | — | — | 7.83GB | 9.17GB | 64MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 778.8 | 29.3 | 400 | 2776ms | 1.7318 | 2.2889 | — | — | 7.83GB | 10.07GB | 104MB | 534MB | The user wants a summary of the provided text from *The Grea |
| summarization | 4096 | 4085 | no-quant | 815.8 | 29.1 | 400 | 5180ms | 1.8065 | 1.9335 | — | — | 7.83GB | 10.27GB | 167MB | 981MB | The user wants a summary of the provided text. The text is t |
| summarization | 8192 | 8190 | no-quant | 838.8 | 28.5 | 400 | 9952ms | 2.0072 | 2.8239 | — | — | 7.83GB | 10.67GB | 260MB | 1.84GB | The user wants a summary of the provided text, which is Chap |
| summarization | 16384 | 16361 | no-quant | 808.7 | 27.3 | 400 | 20528ms | 1.8657 | 2.7595 | — | — | 7.83GB | 11.37GB | 551MB | 3.58GB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 724.2 | 23.5 | 400 | 45634ms | 1.7216 | 2.5938 | — | — | 7.83GB | 12.77GB | 798MB | 7.07GB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 65536 | 65468 | no-quant | 427.9 | 19.7 | 400 | 157399ms | 1.5462 | 3.6396 | — | — | 7.83GB | 15.56GB | 2.04GB | 14.07GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | no-quant | 316.4 | 17.9 | 400 | 413686ms | 1.7393 | 2.1212 | — | — | 7.83GB | 21.44GB | 4.03GB | 28.02GB | The user wants a summary of the provided text. The text cons |
