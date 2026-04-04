# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-04 09:46
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-8bit`

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
| summarization | 128 | 117 | affine-4 | 40.6 | 43.7 | 400 | 3119ms | 1.1768 | 1.4696 | 0.0100 | 0.0061 | 34.30GB | 34.51GB | 22MB | 35MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 4.2 | 43.4 | 400 | 60212ms | 1.2344 | 1.1327 | 0.0040 | 0.0156 | 34.30GB | 34.70GB | 33MB | 44MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 7.5 | 41.9 | 400 | 67911ms | 1.2713 | 1.6128 | 0.0317 | 0.0387 | 34.30GB | 35.07GB | 31MB | 62MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 15.3 | 41.7 | 400 | 67141ms | 1.3012 | 1.5851 | 0.0187 | 0.0283 | 34.30GB | 35.52GB | 30MB | 97MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 29.4 | 40.8 | 400 | 69879ms | 1.2907 | 1.0308 | 0.0280 | 0.0028 | 34.30GB | 36.74GB | 43MB | 167MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 55.0 | 39.8 | 400 | 74715ms | 1.2961 | 1.5919 | -0.0019 | 0.0345 | 34.30GB | 36.88GB | 56MB | 307MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-4 | 112.3 | 37.2 | 400 | 73405ms | 1.3082 | 1.3743 | 0.0246 | 0.0160 | 34.30GB | 37.17GB | 56MB | 587MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-4 | 184.5 | 33.6 | 400 | 89159ms | 1.4034 | 1.5733 | 0.0437 | -0.0001 | 34.30GB | 37.71GB | 88MB | 1.12GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | affine-4 | 259.1 | 28.3 | 400 | 126707ms | 1.3528 | 1.7717 | -0.0009 | 0.0404 | 34.30GB | 38.75GB | 182MB | 2.21GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | affine-4 | 288.7 | 21.4 | 400 | 227193ms | 1.1760 | 1.6607 | 0.0252 | 0.0248 | 34.30GB | 41.03GB | 393MB | 4.40GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | affine-4 | 215.7 | 14.4 | 400 | 606829ms | 1.1553 | 1.5542 | 0.0050 | 0.0191 | 34.30GB | 46.43GB | 678MB | 8.76GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
