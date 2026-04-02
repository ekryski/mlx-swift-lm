# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-01 14:25
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
| summarization | 128 | 119 | no-quant | 810.0 | 99.2 | 378 | 148ms | 3.5813 | 2.7922 | — | — | 1.40GB | 1.66GB | 14MB | Based on the text you provided, here is a summary of the sce |
| summarization | 256 | 251 | no-quant | 2545.5 | 99.3 | 400 | 99ms | 4.3458 | 3.5956 | — | — | 1.40GB | 1.82GB | 12MB | Here is a summary of the provided text:  The passage is an o |
| summarization | 512 | 506 | no-quant | 3283.9 | 98.4 | 400 | 154ms | 5.5213 | 3.4721 | — | — | 1.40GB | 2.24GB | 18MB | Based on the excerpt from *The Great Gatsby*, this text is a |
| summarization | 1024 | 1021 | no-quant | 3697.8 | 96.9 | 400 | 277ms | 4.2793 | 4.3251 | — | — | 1.40GB | 2.76GB | 23MB | Here is a summary of the text provided:  **F. Scott Fitzgera |
| summarization | 2048 | 2044 | no-quant | 3748.4 | 95.3 | 264 | 546ms | 4.9303 | 3.6147 | — | — | 1.40GB | 3.51GB | 39MB | Here is a summary of the provided text by F. Scott Fitzgeral |
| summarization | 4096 | 4087 | no-quant | 4213.2 | 94.9 | 400 | 971ms | 4.7341 | 4.4954 | — | — | 1.40GB | 3.43GB | 64MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 8192 | 8192 | no-quant | 4449.6 | 91.0 | 400 | 1850ms | 2.8126 | 4.1549 | — | — | 1.40GB | 3.72GB | 102MB | This text is an intimate narrative, a short story excerpt fr |
| summarization | 16384 | 16363 | no-quant | 4127.8 | 83.6 | 400 | 4079ms | 3.9829 | 3.4259 | — | — | 1.40GB | 3.98GB | 208MB | Here is a summary of the novel *The Great Gatsby* by F. Scot |
| summarization | 32768 | 32702 | no-quant | 3468.4 | 71.6 | 400 | 9795ms | 4.4159 | 4.1125 | — | — | 1.40GB | 4.82GB | 332MB | Based on the text provided by F. Scott Fitzgerald, here is a |
| summarization | 65536 | 65470 | no-quant | 1779.7 | 44.4 | 400 | 42869ms | 4.6618 | 4.2894 | — | — | 1.40GB | 7.25GB | 783MB | Here is a summary of the provided text excerpt from *The Gre |
| summarization | 131072 | 130775 | no-quant | 1235.7 | 29.2 | 400 | 109395ms | 3.9004 | 3.6341 | — | — | 1.40GB | 8.29GB | 1.01GB | Based on the text provided (which appears to be excerpts fro |
