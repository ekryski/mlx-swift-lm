# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 07:14
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-4B-8bit`

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
| summarization | 128 | 117 | turbo4 | 328.7 | 47.3 | 400 | 358ms | 1.6376 | 1.6516 | 0.0217 | 0.0556 | 4.16GB | 4.48GB | 45MB | 30MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4 | 419.1 | 47.0 | 381 | 595ms | 1.8634 | 1.8179 | 0.0131 | 0.0210 | 4.16GB | 4.76GB | 47MB | 37MB | The user wants me to summarize the content provided in the i |
| summarization | 512 | 504 | turbo4 | 461.3 | 46.9 | 400 | 1093ms | 1.4623 | 1.8991 | 0.0330 | -0.0160 | 4.16GB | 5.11GB | 52MB | 53MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4 | 483.8 | 47.1 | 400 | 2162ms | 1.9211 | 1.9978 | 0.0124 | -0.0003 | 4.16GB | 5.76GB | 72MB | 82MB | The user wants a summary of the provided text, which is the  |
| summarization | 2048 | 2042 | turbo4 | 490.2 | 46.6 | 400 | 4251ms | 2.0363 | 2.7902 | 0.0283 | 0.0625 | 4.16GB | 6.52GB | 104MB | 142MB | The user wants a summary of the provided text. The text is C |
| summarization | 4096 | 4085 | turbo4 | 532.5 | 45.4 | 400 | 7799ms | 2.0004 | 1.8701 | — | — | 4.16GB | 6.72GB | 157MB | 261MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | turbo4 | 548.3 | 44.7 | 400 | 15110ms | 1.9209 | 2.8527 | 0.0484 | 0.0501 | 4.16GB | 7.05GB | 294MB | 499MB | The user wants a summary of the provided text, which is Chap |
| summarization | 16384 | 16361 | turbo4 | 539.1 | 41.7 | 400 | 30593ms | 1.5715 | 2.6151 | 0.0210 | 0.0344 | 4.16GB | 7.78GB | 484MB | 974MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 32768 | 32700 | turbo4 | 507.2 | 37.1 | 400 | 64855ms | 1.6599 | 2.6931 | 0.0134 | 0.0572 | 4.16GB | 9.19GB | 932MB | 1.88GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | turbo4 | 392.0 | 26.8 | 400 | 171153ms | 2.4026 | 2.3926 | 0.0460 | 0.0307 | 4.16GB | 12.17GB | 2.04GB | 3.74GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo4 | 297.3 | 23.6 | 400 | 440173ms | 1.6683 | 1.7600 | 0.0389 | 0.0334 | 4.16GB | 17.94GB | 4.03GB | 7.44GB | The user wants a summary of the provided text. The text cont |
