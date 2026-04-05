# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 15:06
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
| summarization | 128 | 117 | no-quant | 210.8 | 52.5 | 400 | 556ms | 1.4836 | 1.6630 | 0.0845 | 0.0568 | 18.16GB | 18.43GB | 42MB | 113MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 42.0 | 52.6 | 400 | 6231ms | 1.4047 | 1.5130 | 0.0577 | 0.0809 | 18.16GB | 18.60GB | 24MB | 142MB | The user wants a summary of the provided text, which is the  |
| summarization | 512 | 504 | no-quant | 39.1 | 51.9 | 400 | 13286ms | 1.5190 | 1.5716 | 0.0489 | 0.0982 | 18.16GB | 18.94GB | 30MB | 198MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | no-quant | 84.4 | 51.6 | 400 | 12498ms | 1.1518 | 2.0080 | 0.0477 | 0.1020 | 18.16GB | 19.65GB | 59MB | 310MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 146.2 | 51.6 | 400 | 14381ms | 1.3079 | 1.8860 | 0.0385 | 0.0977 | 18.16GB | 20.72GB | 81MB | 534MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 216.5 | 50.2 | 400 | 19308ms | 1.2716 | 1.5110 | 0.0422 | 0.1052 | 18.16GB | 20.80GB | 122MB | 981MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 319.2 | 49.0 | 400 | 26155ms | 1.3858 | 1.8214 | 0.0884 | 0.0948 | 18.16GB | 21.15GB | 139MB | 1.84GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 380.8 | 45.5 | 400 | 43407ms | 1.3978 | 1.6674 | 0.0983 | 0.0451 | 18.16GB | 21.76GB | 251MB | 3.58GB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 429.2 | 40.0 | 400 | 76719ms | 1.3747 | 1.4119 | 0.0577 | 0.0829 | 18.16GB | 22.98GB | 680MB | 7.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 366.0 | 33.9 | 400 | 179437ms | 1.2321 | 1.5621 | 0.0338 | 0.0900 | 18.16GB | 25.60GB | 1.16GB | 14.07GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 250.4 | 25.1 | 400 | 522484ms | 1.3354 | 1.7844 | 0.0745 | 0.0844 | 18.16GB | 30.29GB | 2.54GB | 28.02GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
