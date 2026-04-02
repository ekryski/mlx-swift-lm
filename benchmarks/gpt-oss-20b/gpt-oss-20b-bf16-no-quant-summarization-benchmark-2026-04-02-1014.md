# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-02 10:14
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `sjgdr/gpt-oss-20b-mlx-fp16`

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
| summarization | 128 | 162 | no-quant | 289.3 | 20.9 | 200 | 561ms | — | 2.0335 | — | — | 12.82GB | 15.66GB | 11MB | <\|channel\|>analysis<\|message\|>The user provided a text that  |
| summarization | 256 | 291 | no-quant | 433.5 | 20.9 | 200 | 672ms | — | 2.4316 | — | — | 12.82GB | 15.82GB | 18MB | <\|channel\|>analysis<\|message\|>The user provided a block of t |
| summarization | 512 | 544 | no-quant | 536.9 | 20.7 | 200 | 1014ms | — | 2.1580 | — | — | 12.82GB | 16.14GB | 15MB | <\|channel\|>analysis<\|message\|>The user provided a block of t |
| summarization | 1024 | 1053 | no-quant | 597.7 | 20.7 | 200 | 1927ms | — | 2.3715 | — | — | 12.82GB | 16.85GB | 52MB | <\|channel\|>analysis<\|message\|>The user wants a summary of th |
| summarization | 2048 | 2061 | no-quant | 674.3 | 20.6 | 200 | 3059ms | — | 2.1654 | — | — | 12.82GB | 15.64GB | 93MB | <\|channel\|>analysis<\|message\|>The user provided a large chun |
| summarization | 4096 | 4055 | no-quant | 676.5 | 20.5 | 200 | 6239ms | — | 2.7604 | — | — | 12.82GB | 17.89GB | 148MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 8192 | 8042 | no-quant | 681.0 | 20.3 | 200 | 12060ms | — | 2.4885 | — | — | 12.82GB | 17.95GB | 358MB | <\|channel\|>analysis<\|message\|>The user posted a large chunk  |
| summarization | 16384 | 15955 | no-quant | 640.9 | 19.5 | 200 | 25216ms | — | 2.3213 | — | — | 12.82GB | 17.82GB | 766MB | <\|channel\|>analysis<\|message\|>We have a huge chunk of text.  |
| summarization | 32768 | 31717 | no-quant | 557.4 | 18.1 | 200 | 57257ms | — | 2.4727 | — | — | 12.82GB | 18.14GB | 1003MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 65536 | 63299 | no-quant | 431.5 | 13.9 | 200 | 147077ms | — | 2.1132 | — | — | 12.82GB | 20.29GB | 1.70GB | <\|channel\|>analysis<\|message\|>The user pasted a huge block o |
| summarization | 131072 | 126728 | no-quant | 300.0 | 10.2 | 200 | 422864ms | — | 2.2746 | — | — | 12.82GB | 23.12GB | 5.58GB | <\|channel\|>analysis<\|message\|>We have a huge text: first is  |
