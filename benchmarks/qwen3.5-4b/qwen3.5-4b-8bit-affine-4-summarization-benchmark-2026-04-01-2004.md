# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-01 20:04
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
| summarization | 128 | 117 | affine-4 | 332.8 | 48.6 | 400 | 353ms | 1.5364 | 1.7805 | 0.0225 | 0.0200 | 4.16GB | 4.48GB | 33MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 423.9 | 48.2 | 400 | 588ms | 1.4671 | 2.2560 | 0.0247 | 0.0265 | 4.16GB | 4.76GB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 461.3 | 47.0 | 400 | 1093ms | 1.9142 | 1.0843 | 0.0217 | -0.0060 | 4.16GB | 5.11GB | 31MB | The user wants a summary of the provided text from "The Grea |
| summarization | 1024 | 1019 | affine-4 | 477.5 | 46.8 | 400 | 2202ms | 1.5431 | 2.1998 | -0.0004 | 0.0407 | 4.16GB | 5.21GB | 34MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 483.6 | 46.1 | 400 | 4311ms | 1.4131 | 1.9196 | 0.0366 | 0.0549 | 4.16GB | 6.28GB | 42MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 524.6 | 44.8 | 400 | 7891ms | 2.0350 | 2.5603 | 0.0260 | 0.0287 | 4.16GB | 6.43GB | 58MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | affine-4 | 543.4 | 41.8 | 400 | 15170ms | 2.0753 | 3.9581 | 0.0387 | 0.0402 | 4.16GB | 6.75GB | 100MB | The user wants a summary of the provided text, which is Chap |
| summarization | 16384 | 16361 | affine-4 | 533.6 | 38.1 | 400 | 30775ms | 1.8762 | 1.9079 | 0.0194 | 0.0407 | 4.16GB | 7.33GB | 153MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | affine-4 | 499.1 | 32.3 | 400 | 65632ms | 1.7022 | 2.7768 | 0.0168 | 0.0752 | 4.16GB | 8.54GB | 297MB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | affine-4 | 417.8 | 25.3 | 400 | 156825ms | 1.9956 | 2.4799 | 0.0204 | 0.0606 | 4.16GB | 11.56GB | 605MB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 287.0 | 17.4 | 400 | 455796ms | 1.6447 | 1.8395 | 0.0577 | 0.0219 | 4.16GB | 17.94GB | 1.15GB | The user wants a summary of the provided text. The text cons |
