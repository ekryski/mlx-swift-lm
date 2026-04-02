# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-01 18:27
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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
| summarization | 128 | 119 | affine-4 | 1184.4 | 96.7 | 400 | 101ms | 3.7145 | 3.9347 | 0.0461 | 0.0379 | 1.40GB | 1.61GB | 10MB | Based on the text provided, which appears to be a **typo or  |
| summarization | 256 | 251 | affine-4 | 2156.7 | 94.4 | 213 | 117ms | 4.4893 | 3.4696 | 0.0221 | 0.1071 | 1.40GB | 1.82GB | 10MB | This text is a passage from **The Great Gatsby** by F. Scott |
| summarization | 512 | 506 | affine-4 | 2693.4 | 92.7 | 400 | 188ms | 4.2187 | 4.5511 | 0.0440 | 0.0256 | 1.40GB | 2.24GB | 0MB | Based on the provided text, here is a summary of **The Great |
| summarization | 1024 | 1021 | affine-4 | 3013.3 | 94.0 | 399 | 339ms | 4.2575 | 1.0931 | 0.0392 | -0.0001 | 1.40GB | 2.19GB | 3MB | Based on the excerpt provided, here is a summary of the cont |
| summarization | 2048 | 2044 | affine-4 | 3215.6 | 91.0 | 201 | 636ms | 4.7037 | 3.0352 | 0.0953 | -0.0777 | 1.40GB | 2.98GB | 7MB | Here is a summary of the provided text *Once again to Zelda* |
| summarization | 4096 | 4087 | affine-4 | 3910.6 | 90.0 | 400 | 1045ms | 3.6129 | 4.4174 | 0.0612 | 0.0728 | 1.40GB | 3.05GB | 20MB | This excerpt is a narrative passage written by **Thomas Park |
| summarization | 8192 | 8192 | affine-4 | 4146.9 | 84.3 | 400 | 1976ms | 3.3581 | 4.5240 | 0.0167 | 0.0697 | 1.40GB | 3.21GB | 31MB | Based on the provided excerpts from *The Great Gatsby*, here |
| summarization | 16384 | 16363 | affine-4 | 3998.5 | 77.0 | 400 | 4093ms | 4.8318 | 3.4733 | 0.0410 | 0.0523 | 1.40GB | 3.51GB | 60MB | Based on the provided text, here is a summary of *The Great  |
| summarization | 32768 | 32702 | affine-4 | 3348.0 | 66.6 | 400 | 9775ms | 4.8221 | 4.1193 | 0.0378 | -0.0000 | 1.40GB | 4.76GB | 99MB | Based on the text *The Great Gatsby* by F. Scott Fitzgerald, |
| summarization | 65536 | 65470 | affine-4 | 2286.7 | 52.8 | 400 | 28671ms | 3.8106 | 5.5769 | 0.0471 | 0.0490 | 1.40GB | 5.70GB | 189MB | Based on the text provided (*The Great Gatsby*), here is a s |
| summarization | 131072 | 130775 | affine-4 | 1315.8 | 36.8 | 205 | 99446ms | 3.7980 | 7.0810 | 0.0542 | 0.3661 | 1.40GB | 8.29GB | 368MB | Based on the text provided, this book is **F. Scott Fitzgera |
