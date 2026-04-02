# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-01 19:03
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-2B-bf16`

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
| summarization | 128 | 119 | affine-4 | 720.5 | 57.8 | 400 | 166ms | 3.5567 | 2.8132 | 0.0341 | 0.0500 | 3.51GB | 3.75GB | 12MB | The text provided is a **letter from F. Scott Fitzgerald to  |
| summarization | 256 | 251 | affine-4 | 1145.1 | 58.4 | 201 | 220ms | 2.6637 | 2.1846 | 0.0410 | 0.0002 | 3.51GB | 3.94GB | 5MB | Here is a summary of the text provided:  **1. A Life-Changin |
| summarization | 512 | 506 | affine-4 | 1452.4 | 56.1 | 397 | 349ms | 3.4118 | 4.0848 | 0.0643 | 0.0557 | 3.51GB | 4.15GB | 10MB | Here is a summary of the provided text:  The excerpt outline |
| summarization | 1024 | 1021 | affine-4 | 1576.6 | 56.2 | 400 | 648ms | 2.3231 | 2.6409 | 0.0643 | 0.0125 | 3.51GB | 4.31GB | 13MB | Here is a summary of the provided text from **F. Scott Fitzg |
| summarization | 2048 | 2044 | affine-4 | 1678.8 | 55.3 | 201 | 1229ms | 2.6206 | 1.9607 | 0.0670 | -0.1289 | 3.51GB | 5.12GB | 14MB | Here is a summary of the provided text, which outlines F. Sc |
| summarization | 4096 | 4087 | affine-4 | 1994.1 | 55.0 | 400 | 2071ms | 3.1367 | 2.2626 | 0.0726 | 0.0269 | 3.51GB | 5.19GB | 24MB | This excerpt, primarily from Chapter 1 of *The Great Gatsby* |
| summarization | 8192 | 8192 | affine-4 | 2155.2 | 53.9 | 400 | 3807ms | 4.1456 | 3.6374 | 0.0530 | 0.0623 | 3.51GB | 5.35GB | 13MB | Here is a summary of *The Great Gatsby* as presented in the  |
| summarization | 16384 | 16363 | affine-4 | 2122.5 | 51.5 | 201 | 7727ms | 3.0512 | 1.6451 | 0.0548 | 0.1227 | 3.51GB | 5.64GB | 31MB | This text is a summary of F. Scott Fitzgerald's novel **"The |
| summarization | 32768 | 32702 | affine-4 | 1942.9 | 47.3 | 400 | 16870ms | 3.4327 | 1.0539 | 0.0515 | -0.0055 | 3.51GB | 6.19GB | 100MB | **F. Scott Fitzgerald's "The Great Gatsby" (Chapter Summary) |
| summarization | 65536 | 65470 | affine-4 | 1521.5 | 40.8 | 400 | 43101ms | 4.6460 | 1.0110 | 0.0359 | -0.0002 | 3.51GB | 7.39GB | 114MB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | affine-4 | 971.7 | 32.6 | 400 | 134662ms | 2.8690 | 2.5991 | 0.0093 | 0.0383 | 3.51GB | 10.22GB | 146MB | Here is a summary of the content provided in the text excerp |
