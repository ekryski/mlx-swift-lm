# Inference Benchmark - GPT-OSS 20B

**Date**: 2026-04-02 10:34
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
| summarization | 128 | 162 | no-quant | 352.3 | 60.0 | 200 | 461ms | — | 2.0504 | — | 0.4000 | 10.41GB | 10.90GB | 4MB | <\|channel\|>analysis<\|message\|>We have a user who has posted  |
| summarization | 256 | 291 | no-quant | 350.4 | 60.5 | 200 | 831ms | — | 2.2427 | — | 0.3629 | 10.41GB | 11.14GB | 0MB | <\|channel\|>analysis<\|message\|>The user provided a piece of t |
| summarization | 512 | 544 | no-quant | 563.6 | 59.9 | 200 | 966ms | — | 2.0696 | — | 0.1467 | 10.41GB | 11.67GB | 26MB | <\|channel\|>analysis<\|message\|>The user has provided a long p |
| summarization | 1024 | 1053 | no-quant | 609.6 | 59.2 | 200 | 1812ms | — | 2.4782 | — | 0.2450 | 10.41GB | 12.32GB | 18MB | <\|channel\|>analysis<\|message\|>We need to summarize the conte |
| summarization | 2048 | 2061 | no-quant | 685.9 | 57.5 | 200 | 3007ms | — | 2.8329 | — | 0.3536 | 10.41GB | 12.21GB | 43MB | <\|channel\|>analysis<\|message\|>We have to summarize content.  |
| summarization | 4096 | 4055 | no-quant | 667.5 | 55.8 | 200 | 6302ms | — | 3.3289 | — | 0.3528 | 10.41GB | 13.41GB | 148MB | <\|channel\|>analysis<\|message\|>We have a long text, presumabl |
| summarization | 8192 | 8042 | no-quant | 669.8 | 51.7 | 200 | 12282ms | — | 2.9811 | — | 0.3144 | 10.41GB | 13.46GB | 292MB | <\|channel\|>analysis<\|message\|>We have a long block of text.  |
| summarization | 16384 | 15955 | no-quant | 632.7 | 44.9 | 200 | 25581ms | — | 2.8230 | — | 0.3038 | 10.41GB | 13.48GB | 511MB | <\|channel\|>analysis<\|message\|>We have a long text, apparentl |
| summarization | 32768 | 31717 | no-quant | 540.7 | 35.5 | 200 | 59154ms | — | 2.9118 | — | 0.4175 | 10.41GB | 13.54GB | 1.47GB | <\|channel\|>analysis<\|message\|>We have a massive text: it's e |
| summarization | 65536 | 63299 | no-quant | 398.4 | 21.9 | 200 | 163742ms | — | 3.2566 | — | 0.4061 | 10.41GB | 15.76GB | 2.43GB | <\|channel\|>analysis<\|message\|>We have a very long user input |
| summarization | 131072 | 126728 | no-quant | 299.6 | 13.0 | 200 | 423501ms | — | 3.0341 | — | 0.4601 | 10.41GB | 18.58GB | 5.82GB | <\|channel\|>analysis<\|message\|>We have a long text with many  |
