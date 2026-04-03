# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 16:59
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
| summarization | 128 | 119 | turbo3 | 652.9 | 55.2 | 400 | 183ms | 3.5408 | 1.0173 | 0.0456 | 0.0011 | 3.51GB | 3.75GB | 16MB | 23MB | The text provided is not from *The Great Gatsby* by F. Scott |
| summarization | 256 | 251 | turbo3 | 1161.7 | 55.3 | 400 | 216ms | 2.7047 | 2.1620 | 0.0020 | 0.0366 | 3.51GB | 3.94GB | 16MB | 29MB | The text you provided contains **two completely unrelated te |
| summarization | 512 | 506 | turbo3 | 1373.3 | 54.2 | 387 | 369ms | 3.1199 | 4.1776 | 0.0305 | 0.0214 | 3.51GB | 4.15GB | 21MB | 40MB | This excerpt from *The Great Gatsby* recounts author F. Scot |
| summarization | 1024 | 1021 | turbo3 | 1644.6 | 54.9 | 201 | 621ms | 3.0794 | 1.1294 | 0.0233 | 0.2387 | 3.51GB | 4.68GB | 25MB | 54MB | **F. Scott Fitzgerald's "The Great Gatsby"** explores the co |
| summarization | 2048 | 2044 | turbo3 | 1655.9 | 54.8 | 201 | 1235ms | 2.6399 | 1.7886 | 0.0106 | 0.0002 | 3.51GB | 5.50GB | 36MB | 100MB | Based on the provided text from F. Scott Fitzgerald's "The G |
| summarization | 4096 | 4087 | turbo3 | 2006.1 | 53.7 | 400 | 2045ms | 2.9213 | 2.1697 | 0.0175 | 0.0404 | 3.51GB | 5.41GB | 62MB | 199MB | The text you provided is not from a book titled *Once again  |
| summarization | 8192 | 8192 | turbo3 | 2132.3 | 53.3 | 400 | 3872ms | 2.8974 | 3.0958 | 0.0401 | 0.0350 | 3.51GB | 5.60GB | 111MB | 382MB | This summary of F. Scott Fitzgerald's novella *The Great Gat |
| summarization | 16384 | 16363 | turbo3 | 2146.3 | 51.1 | 400 | 7723ms | 2.1796 | 2.1857 | 0.0098 | 0.0368 | 3.51GB | 5.89GB | 207MB | 745MB | Here is a summary of the text provided, which consists of ch |
| summarization | 32768 | 32702 | turbo3 | 1950.8 | 44.6 | 400 | 17123ms | 2.5564 | 4.2866 | 0.0385 | 0.0811 | 3.51GB | 6.51GB | 399MB | 1.44GB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* ( |
| summarization | 65536 | 65470 | turbo3 | 1412.2 | 39.0 | 201 | 52188ms | 3.0183 | 1.3830 | 0.0124 | 0.4799 | 3.51GB | 7.84GB | 781MB | 2.85GB | **West Egg: A Summary of "The Great Gatsby"**  *The Great Ga |
| summarization | 131072 | 130775 | turbo3 | 1015.3 | 26.2 | 400 | 130233ms | 2.7045 | 2.8929 | 0.0146 | 0.0305 | 3.51GB | 10.22GB | 1.51GB | 5.69GB | Here is a summary of the two major works presented in your t |
