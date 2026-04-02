# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-01 14:46
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-0.8B-8bit`

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
| summarization | 128 | 119 | no-quant | 1044.3 | 117.3 | 400 | 115ms | 4.5526 | 1.8243 | 0.0285 | 0.0221 | 763MB | 960MB | 19MB | This is a collection of modern poems, specifically from the  |
| summarization | 256 | 251 | no-quant | 1768.8 | 119.5 | 370 | 142ms | 3.8379 | 4.2943 | 0.0371 | 0.0178 | 763MB | 1.13GB | 15MB | The provided text is an excerpt from F. Scott Fitzgerald's * |
| summarization | 512 | 506 | no-quant | 2073.5 | 114.3 | 400 | 244ms | 4.0739 | 3.0504 | 0.0381 | 0.0508 | 763MB | 1.52GB | 7MB | Based on the excerpt from **The Great Gatsby**, here is a su |
| summarization | 1024 | 1021 | no-quant | 2297.6 | 107.0 | 400 | 445ms | 3.2947 | 3.1847 | 0.0403 | 0.0340 | 763MB | 2.23GB | 28MB | Based on the text provided from *The Great Gatsby*, here is  |
| summarization | 2048 | 2044 | no-quant | 2400.0 | 106.6 | 400 | 852ms | 4.1542 | 3.8792 | 0.0141 | 0.0158 | 763MB | 3.01GB | 40MB | Based on the text provided, here is a summary of F. Scott Fi |
| summarization | 4096 | 4087 | no-quant | 2783.6 | 105.4 | 386 | 1469ms | 4.4720 | 1.0734 | 0.0370 | -0.0004 | 763MB | 2.90GB | 0MB | Based on the provided text *The Great Gatsby* by F. Scott Fi |
| summarization | 8192 | 8192 | no-quant | 3002.0 | 106.3 | 400 | 2740ms | 2.9594 | 2.6938 | 0.0375 | 0.0335 | 763MB | 3.15GB | 93MB | This text is an excerpt from *The Great Gatsby* by F. Scott  |
| summarization | 16384 | 16363 | no-quant | 2902.3 | 101.8 | 283 | 5751ms | 4.0090 | 5.5064 | 0.0305 | 0.0685 | 763MB | 3.42GB | 208MB | Here is a summary of the passage "Once again to Zelda" from  |
| summarization | 32768 | 32702 | no-quant | 2549.9 | 85.8 | 400 | 13245ms | 4.9476 | 3.5943 | 0.0732 | 0.0048 | 763MB | 4.16GB | 265MB | Based on the text provided by F. Scott Fitzgerald, here is a |
| summarization | 65536 | 65470 | no-quant | 1818.1 | 61.2 | 400 | 36402ms | 3.8309 | 1.2313 | 0.0428 | 0.0157 | 763MB | 6.60GB | 523MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 131072 | 130775 | no-quant | 1072.5 | 46.8 | 400 | 122395ms | 3.2821 | 3.4918 | 0.0167 | 0.0349 | 763MB | 9.41GB | 516MB | Based on the text provided, here is a summary of the charact |
