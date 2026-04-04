# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 04:31
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
| summarization | 128 | 117 | affine-4 | 377.0 | 28.8 | 400 | 311ms | 1.5245 | 2.1217 | 0.0129 | 0.0549 | 7.83GB | 8.07GB | 33MB | 35MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 566.1 | 29.0 | 400 | 440ms | 1.7056 | 1.9273 | 0.0295 | 0.0327 | 7.83GB | 8.21GB | 33MB | 44MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 677.1 | 28.5 | 400 | 745ms | 1.2530 | 2.3006 | 0.0196 | 0.0429 | 7.83GB | 8.55GB | 32MB | 62MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 723.7 | 28.3 | 400 | 1455ms | 1.7532 | 2.4069 | 0.0348 | 0.0453 | 7.83GB | 8.88GB | 36MB | 97MB | The user wants a summary of the provided text, which is Chap |
| summarization | 2048 | 2042 | affine-4 | 756.7 | 28.1 | 400 | 2837ms | 1.2790 | 2.0969 | 0.0062 | 0.0418 | 7.83GB | 9.95GB | 42MB | 167MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 813.5 | 27.6 | 400 | 5148ms | 2.0788 | 2.5125 | 0.0544 | 0.0120 | 7.83GB | 10.10GB | 64MB | 307MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | affine-4 | 810.5 | 26.5 | 400 | 10254ms | 1.7594 | 2.7186 | 0.0475 | -0.0053 | 7.83GB | 10.42GB | 100MB | 587MB | The user wants a summary of the provided text, which is Chap |
| summarization | 16384 | 16361 | affine-4 | 789.0 | 24.9 | 400 | 20900ms | 1.7405 | 2.3501 | 0.0133 | 0.0869 | 7.83GB | 11.00GB | 173MB | 1.12GB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | affine-4 | 691.7 | 22.4 | 400 | 47490ms | 1.6555 | 2.1942 | 0.0353 | 0.0471 | 7.83GB | 12.12GB | 316MB | 2.21GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | affine-4 | 435.6 | 18.2 | 399 | 151076ms | 1.6679 | 2.2024 | 0.0028 | 0.0382 | 7.83GB | 15.13GB | 603MB | 4.40GB | The user wants a summary of the provided text, which is F. S |
| summarization | 131072 | 130773 | affine-4 | 235.6 | 13.1 | 400 | 556910ms | 1.3560 | 1.8151 | -0.0047 | 0.0426 | 7.83GB | 21.44GB | 1.15GB | 8.76GB | The user wants a summary of the provided text. The text cont |
