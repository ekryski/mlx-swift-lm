# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-01 17:09
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-4B-8bit`

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
| summarization | 128 | 117 | no-quant | 338.3 | 49.0 | 400 | 347ms | 1.8059 | 1.7986 | 0.0196 | 0.0378 | 4.16GB | 4.48GB | 49MB | The user wants a summary of the text provided.  1.  **Analyz |
| summarization | 256 | 249 | no-quant | 417.6 | 48.6 | 400 | 597ms | 1.8546 | 1.7570 | -0.0175 | 0.0263 | 4.16GB | 4.76GB | 41MB | The user wants a summary of the provided text. The text prov |
| summarization | 512 | 504 | no-quant | 455.2 | 48.4 | 400 | 1108ms | 1.4342 | 1.8764 | 0.0039 | 0.0179 | 4.16GB | 5.11GB | 56MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 483.0 | 48.6 | 400 | 2161ms | 1.8876 | 2.4419 | 0.0270 | 0.0279 | 4.16GB | 5.76GB | 62MB | The user wants a summary of the provided text, which is the  |
| summarization | 2048 | 2042 | no-quant | 488.7 | 48.2 | 400 | 4272ms | 2.4848 | 1.9051 | -0.0032 | 0.0515 | 4.16GB | 6.52GB | 105MB | The user wants a summary of the provided text from *The Grea |
| summarization | 4096 | 4085 | no-quant | 529.7 | 47.4 | 400 | 7846ms | 1.9750 | 2.0464 | 0.0646 | 0.0736 | 4.16GB | 6.72GB | 146MB | The user wants a summary of the provided text. The text is C |
| summarization | 8192 | 8190 | no-quant | 546.0 | 46.1 | 400 | 15143ms | 2.0180 | 2.6993 | 0.0334 | 0.0218 | 4.16GB | 7.05GB | 260MB | The user wants a summary of the provided text. The text is C |
| summarization | 16384 | 16361 | no-quant | 537.1 | 43.3 | 400 | 30712ms | 2.0226 | 2.0266 | 0.0228 | 0.0204 | 4.16GB | 7.78GB | 552MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | no-quant | 506.7 | 36.9 | 400 | 64935ms | 1.9395 | 2.1381 | 0.0399 | 0.0574 | 4.16GB | 9.19GB | 799MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 65536 | 65468 | no-quant | 429.8 | 32.0 | 400 | 152685ms | 1.7334 | 2.1138 | 0.0114 | 0.0085 | 4.16GB | 12.17GB | 1.79GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | no-quant | 285.8 | 24.6 | 400 | 458032ms | 1.4008 | 2.0682 | 0.0087 | 0.0220 | 4.16GB | 17.94GB | 4.03GB | The user wants a summary of the text provided. The text cons |
