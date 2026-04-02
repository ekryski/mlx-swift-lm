# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-01 17:36
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-4B-4bit`

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
| summarization | 128 | 117 | no-quant | 335.9 | 66.4 | 400 | 351ms | 2.0876 | 1.9486 | 0.1310 | 0.0852 | 2.20GB | 2.52GB | 46MB | The user is asking for a summary of the provided text. The t |
| summarization | 256 | 249 | no-quant | 424.1 | 66.3 | 400 | 589ms | 1.3109 | 1.8209 | 0.0351 | 0.0993 | 2.20GB | 2.83GB | 41MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 455.8 | 65.6 | 400 | 1106ms | 1.4398 | 1.8029 | 0.0646 | 0.0863 | 2.20GB | 3.33GB | 45MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 476.5 | 65.8 | 400 | 2200ms | 1.9350 | 2.9090 | 0.1006 | 0.0527 | 2.20GB | 3.80GB | 61MB | The user wants a summary of the provided text, which is the  |
| summarization | 2048 | 2042 | no-quant | 485.3 | 64.5 | 400 | 4273ms | 1.8638 | 2.1866 | 0.1295 | 0.0658 | 2.20GB | 4.55GB | 89MB | The user wants a summary of the provided text, which is Part |
| summarization | 4096 | 4085 | no-quant | 526.0 | 63.7 | 400 | 7855ms | 1.8234 | 3.0909 | 0.1244 | 0.1190 | 2.20GB | 4.75GB | 147MB | The user wants a summary of the provided text, which is Chap |
| summarization | 8192 | 8190 | no-quant | 542.5 | 60.8 | 400 | 15332ms | 1.5889 | 2.4960 | 0.0228 | 0.2091 | 2.20GB | 5.16GB | 260MB | Here's a thinking process that leads to the suggested summar |
| summarization | 16384 | 16361 | no-quant | 534.3 | 56.1 | 400 | 30824ms | 2.2602 | 3.0767 | 0.1236 | 0.1014 | 2.20GB | 5.81GB | 481MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 502.2 | 48.0 | 400 | 65474ms | 2.5468 | 2.5611 | 0.1885 | 0.1550 | 2.21GB | 7.22GB | 931MB | The user wants a summary of the provided text. The text is t |
| summarization | 65536 | 65468 | no-quant | 426.3 | 38.6 | 400 | 153932ms | 2.3679 | 2.4238 | 0.1622 | 0.1483 | 2.21GB | 10.07GB | 1.53GB | The user wants a summary of the provided text, which is a fu |
| summarization | 131072 | 130773 | no-quant | 305.0 | 29.0 | 400 | 429172ms | 1.6568 | 2.2988 | 0.0862 | 0.0760 | 2.21GB | 15.97GB | 4.03GB | The user wants a summary of the provided text, which consist |
