# Inference Benchmark - Qwen3.5 9B

**Date**: 2026-04-02 00:06
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-9B-4bit`

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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: GPU memory increase from the KV cache after generation; for KV-quantized runs this reflects the compressed cache size.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|--------|
| summarization | 128 | 117 | affine-4 | 200.9 | 46.7 | 386 | 584ms | 1.3538 | 1.5507 | 0.0435 | 0.1144 | 4.69GB | 5.02GB | 42MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 238.7 | 46.7 | 400 | 1044ms | 1.4117 | 1.6172 | 0.0609 | 0.1108 | 4.69GB | 5.29GB | 33MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 253.2 | 45.5 | 400 | 2051ms | 1.5156 | 2.1376 | 0.1069 | 0.0886 | 4.69GB | 5.69GB | 27MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 258.5 | 44.2 | 400 | 4041ms | 1.3925 | 1.9382 | 0.0504 | 0.0794 | 4.69GB | 5.77GB | 21MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 264.9 | 44.5 | 400 | 7859ms | 1.3306 | 2.2801 | 0.0999 | 0.1999 | 4.69GB | 6.86GB | 43MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 286.3 | 42.8 | 400 | 14440ms | 1.8053 | 1.5584 | 0.1504 | 0.1529 | 4.69GB | 7.01GB | 65MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | affine-4 | 296.5 | 41.0 | 400 | 27744ms | 1.5644 | 1.5666 | 0.1664 | 0.1209 | 4.69GB | 7.33GB | 88MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-4 | 296.0 | 37.8 | 400 | 55391ms | 1.4366 | 1.6642 | 0.1014 | 0.1214 | 4.69GB | 7.91GB | 171MB | Here's a thinking process that leads to the suggested summar |
| summarization | 32768 | 32700 | affine-4 | 286.6 | 31.4 | 400 | 114250ms | 2.3130 | 2.0546 | 0.1914 | 0.0831 | 4.69GB | 9.07GB | 296MB | The user wants a summary of the provided text, which is an e |
| summarization | 65536 | 65468 | affine-4 | 249.7 | 24.7 | 400 | 262301ms | 1.4009 | 2.0821 | 0.1336 | 0.1442 | 4.69GB | 12.07GB | 456MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | affine-4 | 203.9 | 17.1 | 400 | 641530ms | 1.4146 | 2.3747 | 0.1133 | 0.1047 | 4.69GB | 18.37GB | 594MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
