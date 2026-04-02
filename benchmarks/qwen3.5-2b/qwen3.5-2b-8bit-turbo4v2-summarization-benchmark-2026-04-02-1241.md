# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 12:41
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | turbo4v2 | 602.5 | 80.1 | 201 | 199ms | 2.3984 | 2.2451 | 0.0239 | -0.0022 | 1.86GB | 2.08GB | 16MB | The text you provided is **not** a summary of *The Great Gat |
| summarization | 1024 | 1021 | turbo4v2 | 1026.9 | 79.2 | 400 | 995ms | 2.6579 | 2.3402 | 0.0390 | 0.0500 | 1.86GB | 3.27GB | 28MB | This excerpt is from the opening chapter of **"The Great Gat |
| summarization | 4096 | 4087 | turbo4v2 | 1232.2 | 77.6 | 400 | 3388ms | 3.1924 | 3.6013 | 0.0586 | 0.0802 | 1.86GB | 3.95GB | 52MB | This text is an excerpt from F. Scott Fitzgerald's **"The Gr |
| summarization | 32768 | 32702 | turbo4v2 | 1273.4 | 63.2 | 400 | 26019ms | 3.1593 | 3.2321 | 0.0321 | 0.0167 | 1.86GB | 5.00GB | 366MB | Here is a summary of the provided text, *The Great Gatsby*,  |
