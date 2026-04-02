# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 15:24
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
| summarization | 128 | 119 | affine-4 | 623.6 | 95.3 | 201 | 192ms | 3.6897 | 1.2591 | 0.1225 | 0.0543 | 1010MB | 1.21GB | 16MB | 22MB | The text you provided appears to be a brief introduction and |
| summarization | 256 | 251 | affine-4 | 885.1 | 91.7 | 400 | 284ms | 2.6843 | 3.0334 | 0.1354 | 0.1035 | 1010MB | 1.43GB | 13MB | 45MB | The text provided is the opening and first paragraph of F. S |
| summarization | 512 | 506 | affine-4 | 981.6 | 89.2 | 242 | 516ms | 3.2834 | 6.7246 | 0.1552 | 0.4495 | 1010MB | 1.88GB | 12MB | 51MB | Here is a summary of the provided text:  The passage introdu |
| summarization | 1024 | 1021 | affine-4 | 1028.1 | 88.9 | 400 | 994ms | 2.6801 | 4.8721 | 0.1884 | 0.2141 | 1010MB | 1.79GB | 15MB | 97MB | Here is a summary of *The Great Gatsby*, focusing on its the |
| summarization | 2048 | 2044 | affine-4 | 1067.3 | 91.6 | 400 | 1932ms | 3.9420 | 5.4132 | 0.2745 | 0.2114 | 1010MB | 2.60GB | 18MB | 167MB | This passage from F. Scott Fitzgerald's *The Great Gatsby* i |
| summarization | 4096 | 4087 | affine-4 | 1201.4 | 89.3 | 202 | 3422ms | 3.0793 | 8.0350 | 0.2108 | 1.0054 | 1010MB | 2.67GB | 25MB | 293MB | This text presents a fictionalized account of Nick Carraway' |
| summarization | 8192 | 8192 | affine-4 | 1327.3 | 84.3 | 400 | 6193ms | 3.6779 | 3.8797 | 0.2019 | 0.2606 | 1010MB | 2.83GB | 32MB | 587MB | Here is a summary of the text provided from *The Great Gatsb |
| summarization | 16384 | 16363 | affine-4 | 1339.9 | 80.9 | 400 | 12234ms | 2.9437 | 4.4421 | 0.2182 | 0.3450 | 1010MB | 3.24GB | 65MB | 1.12GB | Based on the text provided, here is a summary organized by t |
| summarization | 32768 | 32702 | affine-4 | 1271.0 | 71.0 | 400 | 25758ms | 3.0729 | 2.7017 | 0.1653 | 0.1760 | 1010MB | 3.67GB | 119MB | 2.21GB | Here is a summary of the novel ***The Great Gatsby*** by F.  |
| summarization | 65536 | 65470 | affine-4 | 983.0 | 55.9 | 400 | 66661ms | 3.3286 | 3.0671 | 0.2248 | 0.1726 | 1010MB | 5.00GB | 210MB | 4.40GB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
| summarization | 131072 | 130775 | affine-4 | 760.1 | 40.5 | 201 | 172113ms | 2.7779 | 1.2804 | 0.2029 | 0.5164 | 1011MB | 7.82GB | 442MB | 8.74GB | The text provided is a collection of short story segments, p |
