# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-01 14:37
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-0.8B-4bit`

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
| summarization | 128 | 119 | no-quant | 1023.3 | 116.9 | 230 | 117ms | 5.9011 | 3.4468 | 0.2666 | 0.3319 | 404MB | 602MB | 14MB | Based on the text you provided (which appears to be a modern |
| summarization | 256 | 251 | no-quant | 1867.2 | 138.1 | 400 | 135ms | 4.0290 | 5.0337 | 0.2372 | 0.1603 | 404MB | 803MB | 16MB | ### Summary of *The Great Gatsby*  Based on the text you pro |
| summarization | 512 | 506 | no-quant | 2118.9 | 117.6 | 400 | 239ms | 4.9152 | 4.4767 | 0.2808 | 0.1904 | 404MB | 1.19GB | 17MB | This text is the introduction to the book ***Zelda*** by F.  |
| summarization | 1024 | 1021 | no-quant | 2375.3 | 144.5 | 400 | 430ms | 4.6693 | 4.6352 | 0.2365 | 0.1951 | 404MB | 1.90GB | 18MB | Based on the provided excerpt from *The Great Gatsby* by F.  |
| summarization | 2048 | 2044 | no-quant | 2408.1 | 132.6 | 400 | 849ms | 4.9332 | 4.5558 | 0.2990 | 0.2650 | 404MB | 2.66GB | 39MB | Here is a summary of the book *The Great Gatsby*.  ### Summa |
| summarization | 4096 | 4087 | no-quant | 2827.0 | 139.6 | 201 | 1448ms | 5.6439 | 3.5885 | 0.2857 | 0.0216 | 404MB | 2.63GB | 61MB | Based on the provided text from *The Great Gatsby* (Fitzgera |
| summarization | 8192 | 8192 | no-quant | 2998.8 | 117.8 | 400 | 2746ms | 4.7405 | 3.9441 | 0.1644 | 0.2342 | 404MB | 2.80GB | 93MB | This narrative excerpt from *The Great Gatsby* is effectivel |
| summarization | 16384 | 16363 | no-quant | 2901.5 | 114.0 | 249 | 5727ms | 3.8716 | 4.4423 | 0.2128 | 0.1542 | 404MB | 3.07GB | 171MB | Based on the provided text *Once Again to Zelda* by F. Scott |
| summarization | 32768 | 32702 | no-quant | 2549.0 | 93.7 | 400 | 13219ms | 3.5509 | 3.8073 | 0.1523 | 0.1501 | 404MB | 3.81GB | 333MB | This book, *The Great Gatsby*, by F. Scott Fitzgerald, prese |
| summarization | 65536 | 65470 | no-quant | 1866.9 | 67.7 | 400 | 35519ms | 4.0804 | 3.7025 | 0.1600 | 0.1855 | 405MB | 6.36GB | 783MB | Based on the text you provided, here is a summary of the *Th |
| summarization | 131072 | 130775 | no-quant | 1125.0 | 45.4 | 340 | 116727ms | 5.5360 | 4.8837 | 0.2829 | 0.3220 | 405MB | 9.06GB | 1.51GB | Based on the provided text, here is a summary of **Ride the  |
