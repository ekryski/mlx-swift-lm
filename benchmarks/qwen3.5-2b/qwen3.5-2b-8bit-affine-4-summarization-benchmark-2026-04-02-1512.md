# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 15:12
**Branch**: `ek/turbo-opt-0-fix-default-path`
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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | affine-4 | 600.4 | 82.2 | 400 | 199ms | 2.7243 | 2.4723 | 0.0370 | 0.0074 | 1.86GB | 2.08GB | 11MB | 35MB | The text you've shared is **not** a summary of the book *The |
| summarization | 256 | 251 | affine-4 | 863.2 | 80.4 | 201 | 291ms | 2.8382 | 1.1538 | 0.0623 | 0.0174 | 1.86GB | 2.31GB | 16MB | 31MB | The text you provided contains **two completely different wo |
| summarization | 512 | 506 | affine-4 | 966.2 | 78.0 | 400 | 524ms | 2.9282 | 2.1510 | 0.0311 | 0.0508 | 1.86GB | 2.71GB | 13MB | 62MB | The excerpt **does not summarize the work of *The Great Gats |
| summarization | 1024 | 1021 | affine-4 | 1013.8 | 77.2 | 400 | 1008ms | 2.7357 | 1.2437 | 0.0397 | 0.0125 | 1.86GB | 2.67GB | 15MB | 97MB | Based on the text provided from *The Great Gatsby* by F. Sco |
| summarization | 2048 | 2044 | affine-4 | 1060.0 | 77.1 | 400 | 1957ms | 2.9824 | 3.5426 | 0.0252 | 0.0782 | 1.86GB | 3.48GB | 17MB | 167MB | This text, "Once again to Zelda" (though it is actually the  |
| summarization | 4096 | 4087 | affine-4 | 1212.3 | 74.1 | 201 | 3404ms | 3.2148 | 1.6661 | 0.0543 | -0.0053 | 1.86GB | 3.55GB | 25MB | 293MB | Here is a summary of *The Great Gatsby* based on the provide |
| summarization | 8192 | 8192 | affine-4 | 1338.9 | 72.9 | 201 | 6151ms | 3.1980 | 1.6209 | 0.0356 | 0.0039 | 1.86GB | 3.71GB | 24MB | 574MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 16384 | 16363 | affine-4 | 1331.7 | 68.7 | 400 | 12320ms | 2.6690 | 1.1763 | 0.0410 | 0.0205 | 1.86GB | 4.00GB | 55MB | 1.12GB | Here is a summary of the provided text, which serves as the  |
| summarization | 32768 | 32702 | affine-4 | 1276.9 | 61.0 | 400 | 25658ms | 3.2533 | 3.2090 | 0.0317 | 0.0691 | 1.86GB | 4.55GB | 119MB | 2.21GB | This text is the complete text of **F. Scott Fitzgerald's *T |
| summarization | 65536 | 65470 | affine-4 | 1047.5 | 49.8 | 400 | 62579ms | 3.1424 | 3.4014 | 0.0404 | 0.0656 | 1.86GB | 5.82GB | 227MB | 4.40GB | This is a summary of the first volume of **The Great Gatsby* |
| summarization | 131072 | 130775 | affine-4 | 705.3 | 37.1 | 400 | 185492ms | 3.1099 | 2.3044 | 0.0100 | 0.0184 | 1.86GB | 8.63GB | 442MB | 8.76GB | The following is a summary of the provided texts:  *   **F.  |
