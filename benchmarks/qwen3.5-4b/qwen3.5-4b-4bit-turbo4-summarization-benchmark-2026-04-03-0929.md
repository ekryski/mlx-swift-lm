# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 09:29
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-4B-4bit`

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
| summarization | 128 | 117 | turbo4 | 339.8 | 65.1 | 400 | 346ms | 1.8393 | 1.9090 | 0.1443 | 0.1818 | 2.20GB | 2.52GB | 46MB | 30MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4 | 424.1 | 64.3 | 396 | 588ms | 1.7290 | 2.1075 | 0.1658 | 0.1203 | 2.20GB | 2.83GB | 48MB | 37MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4 | 410.1 | 63.1 | 386 | 1233ms | 1.6235 | 2.0236 | 0.1141 | 0.0373 | 2.20GB | 3.33GB | 54MB | 52MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4 | 480.9 | 64.0 | 400 | 2166ms | 1.4029 | 1.9090 | 0.0780 | 0.1238 | 2.20GB | 3.80GB | 67MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 484.7 | 63.2 | 400 | 4272ms | 1.7467 | 2.5111 | 0.1024 | 0.1033 | 2.20GB | 4.55GB | 97MB | 142MB | The user wants a summary of the provided text, which is an e |
| summarization | 4096 | 4085 | turbo4 | 522.1 | 61.8 | 400 | 7912ms | 1.7091 | 2.9713 | 0.0481 | 0.2628 | 2.20GB | 4.75GB | 166MB | 261MB | The user wants a summary of the provided text, which is Chap |
| summarization | 8192 | 8190 | turbo4 | 542.9 | 59.2 | 400 | 15201ms | 1.5037 | 1.9813 | 0.0714 | 0.1125 | 2.20GB | 5.16GB | 297MB | 499MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | turbo4 | 534.5 | 54.8 | 400 | 30826ms | 2.3270 | 1.8679 | 0.1236 | 0.0520 | 2.20GB | 5.81GB | 551MB | 974MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | turbo4 | 501.9 | 47.5 | 400 | 65484ms | 2.4401 | 2.9300 | 0.0904 | 0.1258 | 2.21GB | 7.22GB | 1.04GB | 1.88GB | The user wants a summary of the provided text, which is Chap |
| summarization | 65536 | 65468 | turbo4 | 387.0 | 32.3 | 400 | 172968ms | 2.0558 | 2.4136 | 0.0981 | 0.1755 | 2.21GB | 10.05GB | 2.04GB | 3.74GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo4 | 294.8 | 27.5 | 201 | 443905ms | 1.7894 | 2.1291 | 0.0389 | 2.2522 | 2.21GB | 15.97GB | 3.52GB | 7.43GB | The user wants a summary of the provided text, which contain |
