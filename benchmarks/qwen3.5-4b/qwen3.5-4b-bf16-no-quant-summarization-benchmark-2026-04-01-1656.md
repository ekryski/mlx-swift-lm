# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-01 16:56
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-4B-bf16`

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
| summarization | 128 | 117 | no-quant | 404.3 | 30.4 | 400 | 290ms | 1.8324 | 1.9833 | — | — | 7.83GB | 8.13GB | 48MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 655.1 | 29.3 | 400 | 380ms | 1.5630 | 1.7136 | — | — | 7.83GB | 8.21GB | 30MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 738.2 | 29.3 | 400 | 683ms | 1.5375 | 1.6493 | — | — | 7.83GB | 8.55GB | 0MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 792.0 | 29.6 | 400 | 1313ms | 1.5166 | 2.0687 | — | — | 7.83GB | 9.17GB | 64MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 795.7 | 30.0 | 400 | 2758ms | 2.0137 | 2.0611 | — | — | 7.83GB | 10.07GB | 66MB | The user wants a summary of the provided text, which is Chap |
| summarization | 4096 | 4085 | no-quant | 844.9 | 29.6 | 400 | 5107ms | 1.9736 | 2.1286 | — | — | 7.83GB | 10.27GB | 167MB | The user wants a summary of the provided text, which is Chap |
| summarization | 8192 | 8190 | no-quant | 855.1 | 30.1 | 201 | 9813ms | 2.3205 | 4.2261 | — | — | 7.83GB | 10.67GB | 252MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | no-quant | 823.9 | 28.5 | 400 | 20144ms | 1.7345 | 2.2541 | — | — | 7.83GB | 11.37GB | 554MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 734.9 | 25.5 | 400 | 44847ms | 1.8986 | 3.3677 | — | — | 7.83GB | 12.78GB | 1.04GB | The user wants a summary of the provided text, which is Chap |
| summarization | 65536 | 65468 | no-quant | 578.4 | 22.7 | 400 | 113615ms | 2.1222 | 2.0357 | — | — | 7.83GB | 15.70GB | 1.78GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | no-quant | 363.0 | 19.0 | 400 | 360590ms | 1.8052 | 2.3077 | — | — | 7.83GB | 21.44GB | 4.03GB | The user wants a summary of the provided text, which consist |
