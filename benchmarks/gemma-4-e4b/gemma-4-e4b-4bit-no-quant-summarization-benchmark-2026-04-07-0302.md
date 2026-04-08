# Inference Benchmark - Gemma 4 E4B

**Date**: 2026-04-07 03:02
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: 4bit
**Model**: `mlx-community/gemma-4-e4b-it-4bit`

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
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

- **Scenario**: Benchmark method — `simple` (basic chat), `summarization` (context-scaling), `wikitext2` (forced-decode LM perplexity), `niah` (needle-in-a-haystack retrieval), `multi-turn`, `tool-calling`.
- **Think PPL / Gen PPL**: Perplexity (exp of mean negative log-probability) over thinking and generation phase tokens respectively. Lower is better. For `wikitext2`, Gen PPL is the standard LM perplexity via forced decode on WikiText-2 test data.
- **Think KLD / Gen KLD**: KL divergence of the target configuration vs the highest-fidelity baseline for this model family (bf16, or 8-bit if bf16 exceeds GPU memory). Computed by forced-decoding the target's generated tokens through the baseline model without KV cache compression. Higher values indicate greater divergence from the gold-standard model. Values near 0 mean the deployment config introduces negligible quality loss.
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | no-quant | 315.6 | 44.9 | 400 | 371ms | 2.2685 | — | 1.5610 | — | 3.98GB | 4.46GB | 18MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 463.1 | 43.8 | 379 | 554ms | 1.9698 | 1.2615 | 1.1906 | 4.9029 | 3.98GB | 4.88GB | 32MB | 139MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 551.6 | 42.9 | 400 | 912ms | 2.2309 | — | 0.9457 | — | 3.98GB | 5.53GB | 38MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 563.2 | 42.4 | 400 | 1848ms | 2.2217 | — | 1.0566 | — | 3.98GB | 6.21GB | 45MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 549.1 | 41.1 | 400 | 3884ms | 2.1009 | — | 0.9507 | — | 3.98GB | 7.53GB | 72MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 568.5 | 39.0 | 326 | 7372ms | 1.9282 | 3.0957 | 0.7638 | 1.8959 | 3.98GB | 7.46GB | 163MB | 967MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8229 | no-quant | 964.4 | 34.1 | 19 | 8555ms | — | 4.1079 | 3.0045 | — | 3.98GB | 6.14GB | 290MB | 1.76GB | This is a fascinating collection of pieces.  The provided te |
| summarization | 16384 | 16395 | no-quant | 1066.6 | 30.1 | 59 | 15426ms | — | 2.8379 | 1.0350 | — | 3.98GB | 6.87GB | 547MB | 3.51GB | The provided texts are different versions of the opening to  |
| summarization | 32768 | 32815 | no-quant | 1023.0 | 20.4 | 15 | 32113ms | — | 3.8050 | 2.0085 | — | 3.98GB | 8.40GB | 1.04GB | 7.01GB | This is a very long piece of text, and it presents several p |
| summarization | 65536 | 65896 | no-quant | 806.1 | 13.9 | 40 | 81822ms | — | 3.1164 | 1.0949 | — | 3.98GB | 11.71GB | 2.03GB | 14.09GB | The provided text is a very long, detailed, and extended ver |
| summarization | 131072 | 130563 | no-quant | 502.5 | 10.2 | 290 | 259996ms | — | 3.1627 | 0.8016 | — | 3.98GB | 18.11GB | 3.03GB | 27.95GB | This is a highly complex literary novel, and the sheer volum |
