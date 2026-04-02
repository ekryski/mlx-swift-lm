# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-01 16:43
**Branch**: `ek/consolidated-benchmarks`
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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: GPU memory increase from the KV cache after generation; for KV-quantized runs this reflects the compressed cache size.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|--------|
| summarization | 128 | 119 | no-quant | 635.6 | 109.8 | 261 | 189ms | 3.4204 | 2.2864 | 0.1776 | 0.2816 | 1010MB | 1.21GB | 16MB | The text you provided is the first stanza of a famous **epis |
| summarization | 256 | 251 | no-quant | 907.3 | 110.8 | 201 | 277ms | 3.6514 | 1.7788 | 0.1623 | 0.2644 | 1010MB | 1.43GB | 0MB | This excerpt provides a concise summary of the core themes a |
| summarization | 512 | 506 | no-quant | 1005.1 | 110.9 | 400 | 504ms | 5.0872 | 3.8142 | 0.1719 | 0.1392 | 1010MB | 1.88GB | 4MB | **Summary**  The provided text excerpt introduces **Thomas P |
| summarization | 1024 | 1021 | no-quant | 1057.8 | 111.0 | 400 | 966ms | 3.5435 | 2.8148 | 0.2449 | 0.1875 | 1010MB | 2.48GB | 0MB | Here is a summary of the text provided from F. Scott Fitzger |
| summarization | 2048 | 2044 | no-quant | 1081.7 | 108.5 | 400 | 1905ms | 2.4966 | 1.0223 | 0.0802 | -0.0095 | 1010MB | 3.18GB | 39MB | This passage serves as a prologue to *The Great Gatsby* by * |
| summarization | 4096 | 4087 | no-quant | 1242.7 | 105.8 | 400 | 3333ms | 3.2418 | 1.0900 | 0.0572 | 0.0040 | 1010MB | 3.14GB | 53MB | ### **Summary of Selected Text from "The Great Gatsby" by F. |
| summarization | 8192 | 8192 | no-quant | 1338.4 | 100.8 | 400 | 6161ms | 2.6325 | 4.3227 | 0.2502 | 0.1823 | 1010MB | 3.34GB | 55MB | Based on the text provided from "The Great Gatsby" by F. Sco |
| summarization | 16384 | 16363 | no-quant | 1352.3 | 94.1 | 224 | 12231ms | 3.7000 | 3.5733 | 0.1523 | 0.1379 | 1010MB | 3.62GB | 205MB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
| summarization | 32768 | 32702 | no-quant | 1292.1 | 81.3 | 400 | 25662ms | 3.5168 | 1.0491 | 0.2568 | 0.0069 | 1010MB | 4.18GB | 400MB | ### The Great Gatsby (Summary)  F. Scott Fitzgerald's **The  |
| summarization | 65536 | 65470 | no-quant | 1106.5 | 61.8 | 400 | 59524ms | 3.8884 | 3.7864 | 0.1350 | 0.2424 | 1010MB | 5.47GB | 0MB | The text provided is a collection of excerpts from the novel |
| summarization | 131072 | 130775 | no-quant | 802.0 | 47.1 | 201 | 163557ms | 4.4741 | 1.1758 | 0.2380 | 0.9673 | 1011MB | 7.82GB | 643MB | This text contains two distinct novels written by famous aut |
