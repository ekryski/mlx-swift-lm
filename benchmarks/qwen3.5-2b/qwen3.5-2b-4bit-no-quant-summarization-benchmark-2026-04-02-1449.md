# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 14:49
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-2B-4bit`

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
| summarization | 128 | 119 | no-quant | 628.7 | 95.5 | 400 | 191ms | 2.1156 | 1.6118 | 0.0582 | 0.0851 | 1010MB | 1.21GB | 19MB | 114MB | The text you provided is the title page of **The Great Gatsb |
| summarization | 256 | 251 | no-quant | 877.4 | 88.3 | 201 | 287ms | 3.4211 | 3.3233 | 0.1273 | -0.4895 | 1010MB | 1.43GB | 16MB | 99MB | The excerpt from *The Great Gatsby* by F. Scott Fitzgerald p |
| summarization | 512 | 506 | no-quant | 963.8 | 94.2 | 201 | 526ms | 3.8031 | 1.5570 | 0.2157 | -0.2300 | 1010MB | 1.88GB | 19MB | 155MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 1024 | 1021 | no-quant | 1032.6 | 92.8 | 400 | 989ms | 3.1073 | 2.6954 | 0.1136 | 0.0681 | 1010MB | 2.48GB | 27MB | 311MB | Here is a summary of the provided text from **"The Great Gat |
| summarization | 2048 | 2044 | no-quant | 1057.5 | 93.9 | 400 | 1953ms | 2.9427 | 2.3412 | 0.1449 | 0.0487 | 1010MB | 3.18GB | 39MB | 535MB | This text presents a two-part work by F. Scott Fitzgerald: f |
| summarization | 4096 | 4087 | no-quant | 1231.6 | 96.3 | 201 | 3362ms | 3.4383 | 1.3723 | 0.1596 | -0.0086 | 1010MB | 3.14GB | 10MB | 938MB | This text is a literary excerpt from *The Great Gatsby* by F |
| summarization | 8192 | 8192 | no-quant | 1344.7 | 94.8 | 400 | 6127ms | 2.6503 | 3.9282 | 0.2291 | 0.1569 | 1010MB | 3.34GB | 112MB | 1.84GB | Here is a summary of the text:  ### **Overview** *   **The W |
| summarization | 16384 | 16363 | no-quant | 1353.4 | 87.9 | 201 | 12220ms | 3.1272 | 1.5395 | 0.1782 | -0.0097 | 1010MB | 3.62GB | 153MB | 3.54GB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
| summarization | 32768 | 32702 | no-quant | 1278.9 | 74.5 | 400 | 25921ms | 3.4783 | 1.1054 | 0.1809 | 0.0064 | 1010MB | 4.18GB | 399MB | 7.07GB | Here is a summary of the novella *The Great Gatsby* by F. Sc |
| summarization | 65536 | 65470 | no-quant | 992.9 | 41.8 | 400 | 70621ms | 6.4846 | 3.6939 | 0.2001 | 0.1864 | 1010MB | 5.46GB | 783MB | 14.07GB | # Summary of The Great Gatsby  ## Introduction F. Scott Fitz |
| summarization | 131072 | 130775 | no-quant | 771.6 | 38.9 | 201 | 171981ms | 3.2718 | 1.6255 | 0.1577 | 0.2558 | 1011MB | 7.82GB | 1.26GB | 27.98GB | The text provided is a collection of short stories. Here is  |
