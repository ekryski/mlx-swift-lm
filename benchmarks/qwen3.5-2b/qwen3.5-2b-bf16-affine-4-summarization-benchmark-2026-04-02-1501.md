# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 15:01
**Branch**: `ek/turbo-opt-0-fix-default-path`
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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | affine-4 | 719.4 | 55.9 | 371 | 166ms | 3.3916 | 3.7840 | 0.0304 | 0.0632 | 3.51GB | 3.75GB | 16MB | 33MB | The text provided is the opening portion of an early poem by |
| summarization | 256 | 251 | affine-4 | 1157.3 | 54.9 | 400 | 217ms | 2.6209 | 2.4982 | 0.0441 | 0.0521 | 3.51GB | 3.94GB | 11MB | 45MB | The text you provided offers two distinct short stories with |
| summarization | 512 | 506 | affine-4 | 1430.8 | 54.3 | 400 | 354ms | 3.3881 | 1.0125 | 0.0369 | 0.0055 | 3.51GB | 4.15GB | 13MB | 62MB | **F. Scott Fitzgerald's Essay: A Defense of Judgment and the |
| summarization | 1024 | 1021 | affine-4 | 1543.2 | 54.1 | 201 | 662ms | 3.2084 | 1.3075 | 0.0494 | -0.0382 | 3.51GB | 4.31GB | 13MB | 84MB | Here is a summary of the provided text:  The excerpt consist |
| summarization | 2048 | 2044 | affine-4 | 1692.6 | 52.1 | 400 | 1214ms | 3.9213 | 3.1049 | 0.0307 | -0.0106 | 3.51GB | 5.12GB | 15MB | 167MB | This text, from "The Great Gatsby," is a chapter-by-chapter  |
| summarization | 4096 | 4087 | affine-4 | 1917.2 | 52.6 | 400 | 2147ms | 2.7572 | 1.1150 | 0.0413 | -0.0028 | 3.51GB | 5.19GB | 24MB | 307MB | Here is a summary of the provided text from **"The Great Gat |
| summarization | 8192 | 8192 | affine-4 | 2107.8 | 50.6 | 400 | 3890ms | 2.3940 | 2.4404 | 0.0334 | 0.0536 | 3.51GB | 5.35GB | 38MB | 587MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 16384 | 16363 | affine-4 | 2070.0 | 48.3 | 378 | 7922ms | 3.2812 | 1.0402 | 0.0131 | -0.0018 | 3.51GB | 5.64GB | 66MB | 1.12GB | Based on the text provided, here is a summary of *The Great  |
| summarization | 32768 | 32702 | affine-4 | 1899.4 | 44.6 | 201 | 17270ms | 3.2083 | 2.9035 | 0.0054 | 0.1678 | 3.51GB | 6.19GB | 118MB | 2.20GB | This text is a collection of excerpts from F. Scott Fitzgera |
| summarization | 65536 | 65470 | affine-4 | 1479.7 | 39.3 | 400 | 44348ms | 3.4021 | 2.5614 | 0.0406 | 0.0188 | 3.51GB | 7.39GB | 227MB | 4.40GB | Here is a summary of the narrative structure and themes in * |
| summarization | 131072 | 130775 | affine-4 | 977.4 | 30.5 | 400 | 133923ms | 3.2060 | 2.3422 | 0.0715 | 0.0134 | 3.51GB | 10.22GB | 442MB | 8.76GB | **Summary of *The Age of Innocence* by Edith Wharton**  *Age |
