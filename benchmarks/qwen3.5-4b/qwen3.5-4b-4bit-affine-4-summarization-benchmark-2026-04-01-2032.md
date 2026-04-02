# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-01 20:32
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
| summarization | 128 | 117 | affine-4 | 338.8 | 68.2 | 354 | 347ms | 1.8582 | 2.6071 | 0.0553 | 0.0713 | 2.20GB | 2.52GB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 422.8 | 67.6 | 306 | 589ms | 1.3626 | 2.5707 | 0.0674 | 0.1202 | 2.20GB | 2.83GB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 461.0 | 64.8 | 400 | 1094ms | 1.5210 | 1.6220 | 0.0423 | 0.1404 | 2.20GB | 3.33GB | 31MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 472.2 | 64.0 | 400 | 2196ms | 2.0910 | 2.3524 | 0.0789 | 0.0660 | 2.20GB | 3.25GB | 32MB | The user wants a summary of the provided text, which is Chap |
| summarization | 2048 | 2042 | affine-4 | 480.2 | 62.4 | 400 | 4313ms | 2.0562 | 1.9000 | 0.0814 | 0.1169 | 2.20GB | 4.32GB | 44MB | The user wants a summary of the provided text, which is the  |
| summarization | 4096 | 4085 | affine-4 | 517.5 | 59.7 | 400 | 7951ms | 1.7581 | 2.5171 | 0.1360 | 0.1135 | 2.20GB | 4.47GB | 58MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-4 | 537.3 | 54.8 | 400 | 15308ms | 1.7742 | 2.3847 | 0.0760 | 0.1254 | 2.20GB | 4.79GB | 90MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 16384 | 16361 | affine-4 | 527.7 | 48.3 | 400 | 31074ms | 1.9422 | 3.4385 | 0.0737 | 0.1470 | 2.20GB | 5.38GB | 152MB | The user wants a summary of the provided text, which consist |
| summarization | 32768 | 32700 | affine-4 | 494.6 | 39.4 | 400 | 66191ms | 2.3483 | 2.8483 | 0.1619 | 0.0774 | 2.21GB | 6.59GB | 278MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 65536 | 65468 | affine-4 | 413.5 | 29.0 | 400 | 158394ms | 1.4533 | 2.7241 | 0.0634 | 0.1573 | 2.21GB | 9.64GB | 570MB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 131072 | 130773 | affine-4 | 285.5 | 19.1 | 400 | 458170ms | 2.1042 | 2.2368 | 0.1314 | 0.1596 | 2.21GB | 15.97GB | 598MB | The user wants a summary of the provided text. The text cont |
