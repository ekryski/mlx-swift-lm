# Inference Benchmark - Qwen3.5 9B

**Date**: 2026-04-01 21:59
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-9B-4bit`

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
| summarization | 128 | 117 | no-quant | 204.3 | 46.8 | 369 | 574ms | 1.3413 | 1.9292 | 0.0511 | 0.0332 | 4.69GB | 5.02GB | 42MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 239.2 | 46.8 | 336 | 1041ms | 1.6019 | 1.9338 | 0.1272 | 0.1132 | 4.69GB | 5.29GB | 39MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 252.8 | 46.8 | 400 | 2048ms | 1.3508 | 1.9542 | 0.0943 | 0.0680 | 4.69GB | 5.69GB | 48MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 263.6 | 46.7 | 400 | 3961ms | 1.4282 | 1.7059 | 0.0625 | 0.0601 | 4.69GB | 6.26GB | 55MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 266.1 | 46.3 | 400 | 7822ms | 1.3214 | 2.4261 | 0.1083 | 0.2026 | 4.69GB | 7.02GB | 64MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 286.8 | 45.4 | 400 | 14443ms | 1.3800 | 1.5341 | 0.0795 | 0.0883 | 4.69GB | 7.21GB | 147MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 296.7 | 44.4 | 400 | 27792ms | 1.8462 | 1.5321 | 0.1474 | 0.0962 | 4.69GB | 7.61GB | 296MB | The user wants a summary of the provided text from *The Grea |
| summarization | 16384 | 16361 | no-quant | 296.5 | 41.9 | 400 | 55527ms | 1.9006 | 1.9064 | 0.1055 | 0.0912 | 4.69GB | 8.28GB | 346MB | The user wants a summary of the provided text from *The Grea |
| summarization | 32768 | 32700 | no-quant | 287.2 | 37.4 | 400 | 114247ms | 1.3739 | 1.6097 | 0.1376 | 0.0865 | 4.69GB | 9.68GB | 931MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 258.0 | 31.6 | 400 | 254239ms | 1.3940 | 2.2440 | 0.0375 | 0.1110 | 4.69GB | 12.73GB | 2.04GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 201.8 | 24.5 | 400 | 648328ms | 1.3470 | 1.7843 | 0.0899 | 0.1386 | 4.69GB | 18.37GB | 3.53GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
