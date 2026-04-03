# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-02 10:30
**Branch**: `ek/consolidated-benchmarks`
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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: GPU memory increase from the KV cache after generation; for KV-quantized runs this reflects the compressed cache size.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|--------|
| summarization | 128 | 162 | no-quant | 358.4 | 60.8 | 200 | 453ms | — | 2.0767 | — | — | 10.41GB | 10.90GB | 11MB | <\|channel\|>analysis<\|message\|>We have a user input: They pro |
| summarization | 256 | 291 | no-quant | 505.1 | 61.0 | 200 | 577ms | — | 2.2279 | — | — | 10.41GB | 11.14GB | 13MB | <\|channel\|>analysis<\|message\|>We have a user prompt: The use |
| summarization | 512 | 544 | no-quant | 577.1 | 60.2 | 200 | 943ms | — | 2.8081 | — | — | 10.41GB | 11.67GB | 20MB | <\|channel\|>analysis<\|message\|>The user provided a text: appe |
| summarization | 1024 | 1053 | no-quant | 620.3 | 59.4 | 200 | 1795ms | — | 2.3512 | — | — | 10.41GB | 12.32GB | 14MB | <\|channel\|>analysis<\|message\|>The user has posted a long exc |
| summarization | 2048 | 2061 | no-quant | 694.7 | 58.8 | 200 | 2970ms | — | 1.9006 | — | — | 10.41GB | 12.21GB | 68MB | <\|channel\|>analysis<\|message\|>We have a user request: Summar |
| summarization | 4096 | 4055 | no-quant | 669.6 | 56.4 | 200 | 6288ms | — | 2.4356 | — | — | 10.41GB | 13.41GB | 140MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | no-quant | 664.8 | 52.6 | 200 | 12384ms | — | 2.7376 | — | — | 10.41GB | 13.46GB | 358MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 16384 | 15955 | no-quant | 627.6 | 46.5 | 200 | 25748ms | — | 2.5968 | — | — | 10.41GB | 13.48GB | 606MB | <\|channel\|>analysis<\|message\|>We have a huge text: It's a no |
| summarization | 32768 | 31717 | no-quant | 545.4 | 37.2 | 200 | 58634ms | — | 3.0791 | — | — | 10.41GB | 13.54GB | 1.47GB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
