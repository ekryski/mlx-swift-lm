# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-06 16:52
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `loan-star/gpt-oss-20b-mlx-4Bit`

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
| Temperature | 0.8 |
| Top P | 0.8 |
| Top K | 0 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Reasoning Effort | medium |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 162 | no-quant | 478.9 | 72.4 | 200 | 398ms | — | — | — | — | 10.41GB | 10.95GB | 6MB | 79MB | <\|channel\|>analysis<\|message\|>We have a prompt. The user has |
| summarization | 256 | 291 | no-quant | 543.4 | 72.1 | 200 | 542ms | — | — | — | — | 10.41GB | 11.32GB | 0MB | 107MB | <\|channel\|>analysis<\|message\|>The user provided a text excer |
| summarization | 512 | 544 | no-quant | 612.0 | 66.2 | 200 | 907ms | — | — | — | — | 10.41GB | 11.91GB | 17MB | 163MB | <\|channel\|>analysis<\|message\|>We have to summarize the conte |
| summarization | 1024 | 1053 | no-quant | 601.4 | 64.7 | 200 | 1851ms | — | — | — | — | 10.41GB | 12.51GB | 54MB | 274MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 2048 | 2061 | no-quant | 688.1 | 67.3 | 200 | 3010ms | — | — | — | — | 10.41GB | 12.40GB | 94MB | 495MB | <\|channel\|>analysis<\|message\|>The user has provided a piece  |
| summarization | 4096 | 4055 | no-quant | 680.9 | 64.9 | 200 | 6179ms | — | — | — | — | 10.41GB | 13.67GB | 173MB | 931MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | no-quant | 666.8 | 61.5 | 200 | 12291ms | — | — | — | — | 10.41GB | 13.71GB | 357MB | 1.76GB | <\|channel\|>analysis<\|message\|>We have a long text that appea |
| summarization | 16384 | 15955 | no-quant | 635.7 | 54.6 | 200 | 25299ms | — | — | — | — | 10.41GB | 13.69GB | 447MB | 3.45GB | <\|channel\|>analysis<\|message\|>We have a huge block of text t |
| summarization | 32768 | 31717 | no-quant | 553.2 | 44.3 | 200 | 57624ms | — | — | — | — | 10.41GB | 13.78GB | 1.10GB | 6.82GB | <\|channel\|>analysis<\|message\|>We have a long text: It's a ve |
| summarization | 65536 | 63299 | no-quant | 414.3 | 26.0 | 200 | 154114ms | — | — | — | — | 10.41GB | 15.84GB | 2.79GB | 13.56GB | <\|channel\|>analysis<\|message\|>We have a huge block of text.  |
| summarization | 131072 | 126728 | no-quant | 292.6 | 15.3 | 200 | 433416ms | — | — | — | — | 10.41GB | 18.99GB | 3.39GB | 27.11GB | <\|channel\|>analysis<\|message\|>We have a very long text: basi |
