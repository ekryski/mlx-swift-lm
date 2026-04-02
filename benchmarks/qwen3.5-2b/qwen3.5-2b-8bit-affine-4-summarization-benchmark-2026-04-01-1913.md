# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-01 19:13
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | affine-4 | 634.9 | 88.7 | 400 | 189ms | 2.4831 | 2.5562 | 0.0205 | 0.0414 | 1.86GB | 2.08GB | 9MB | The text you provided consists of a specific excerpt from ** |
| summarization | 256 | 251 | affine-4 | 887.8 | 87.3 | 390 | 283ms | 2.6047 | 2.8047 | 0.0148 | 0.0295 | 1.86GB | 2.31GB | 3MB | Here is a summary of the provided text:  The passage begins  |
| summarization | 512 | 506 | affine-4 | 993.0 | 84.1 | 378 | 510ms | 4.6845 | 3.2478 | 0.0683 | 0.0350 | 1.86GB | 2.71GB | 7MB | This excerpt from F. Scott Fitzgerald's **"Once again to Zel |
| summarization | 1024 | 1021 | affine-4 | 1043.4 | 83.4 | 380 | 979ms | 3.9014 | 3.2046 | 0.0465 | 0.0567 | 1.86GB | 2.67GB | 15MB | This text presents **Gatsby, the Great Gatsby**.  It consist |
| summarization | 2048 | 2044 | affine-4 | 1065.9 | 84.0 | 400 | 1940ms | 2.5787 | 2.8458 | 0.0242 | 0.0163 | 1.86GB | 3.48GB | 7MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 4096 | 4087 | affine-4 | 1238.3 | 82.0 | 201 | 3351ms | 3.5605 | 1.8772 | 0.0522 | -0.0867 | 1.86GB | 3.55GB | 18MB | Here is a summary of the provided text:  **Introduction and  |
| summarization | 8192 | 8192 | affine-4 | 1342.2 | 78.1 | 201 | 6133ms | 3.2086 | 2.1492 | 0.0550 | -0.0048 | 1.86GB | 3.71GB | 24MB | Here is a summary of the content provided:  **F. Scott Fitzg |
| summarization | 16384 | 16363 | affine-4 | 1352.5 | 73.4 | 400 | 12129ms | 2.6833 | 2.5801 | 0.0345 | 0.0481 | 1.86GB | 4.00GB | 60MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 32768 | 32702 | affine-4 | 1288.9 | 64.1 | 201 | 25414ms | 3.3332 | 2.0358 | 0.0306 | 0.7572 | 1.86GB | 4.55GB | 98MB | Here is a summary of the provided text, which begins with th |
| summarization | 65536 | 65470 | affine-4 | 1097.8 | 52.7 | 400 | 59688ms | 3.4533 | 3.9634 | 0.0544 | 0.0723 | 1.86GB | 5.82GB | 190MB | This text is the concluding chapter of F. Scott Fitzgerald's |
| summarization | 131072 | 130775 | affine-4 | 791.9 | 39.6 | 400 | 165216ms | 3.0966 | 3.4137 | 0.0555 | 0.0546 | 1.86GB | 8.63GB | 368MB | This is a summary of F. Scott Fitzgerald's **The Great Gatsb |
