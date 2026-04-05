# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 15:47
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
| summarization | 128 | 117 | affine-4 | 195.5 | 52.2 | 400 | 600ms | 1.2465 | 1.5298 | 0.0704 | 0.1099 | 18.16GB | 18.43GB | 17MB | 35MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 256 | 249 | affine-4 | 44.8 | 51.8 | 400 | 5810ms | 1.7801 | 1.7004 | 0.1121 | 0.0986 | 18.16GB | 18.60GB | 30MB | 44MB | The user wants a summary of the provided text, which is the  |
| summarization | 512 | 504 | affine-4 | 39.9 | 49.8 | 400 | 13021ms | 1.4132 | 1.7992 | 0.0925 | 0.0957 | 18.16GB | 18.94GB | 35MB | 62MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | affine-4 | 84.6 | 49.2 | 400 | 12497ms | 1.6107 | 2.1051 | 0.1204 | 0.1084 | 18.16GB | 19.38GB | 41MB | 97MB | The user wants a summary of the provided text from "The Grea |
| summarization | 2048 | 2042 | affine-4 | 139.1 | 48.5 | 400 | 15100ms | 1.2599 | 1.5790 | 0.0603 | 0.1153 | 18.16GB | 20.60GB | 33MB | 167MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 221.6 | 46.8 | 399 | 18888ms | 1.7715 | 1.0989 | 0.1162 | 0.0094 | 18.16GB | 20.74GB | 55MB | 307MB | The user wants a summary of the provided text from "The Grea |
| summarization | 8192 | 8190 | affine-4 | 329.1 | 43.3 | 400 | 25320ms | 1.4385 | 1.5405 | 0.1529 | 0.0754 | 18.16GB | 21.03GB | 70MB | 587MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-4 | 394.3 | 38.3 | 400 | 41915ms | 1.5798 | 1.9211 | 0.0769 | 0.1023 | 18.16GB | 21.57GB | 113MB | 1.12GB | The user wants a summary of the provided text.  1.  **Identi |
| summarization | 32768 | 32700 | affine-4 | 432.6 | 31.2 | 400 | 76036ms | 1.3894 | 1.7866 | 0.0931 | 0.0407 | 18.16GB | 22.61GB | 213MB | 2.21GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | affine-4 | 371.5 | 23.3 | 400 | 176797ms | 1.4548 | 1.9353 | 0.0502 | 0.0811 | 18.16GB | 24.89GB | 315MB | 4.40GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 248.4 | 15.1 | 400 | 527012ms | 1.3475 | 1.7683 | 0.0614 | 0.1092 | 18.16GB | 30.29GB | 603MB | 8.76GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
