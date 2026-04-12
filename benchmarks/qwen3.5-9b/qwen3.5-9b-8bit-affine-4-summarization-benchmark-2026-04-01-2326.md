# Inference Benchmark - Qwen3.5 9B

- **Date**: 2026-04-01 23:26
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-9B-8bit`

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
| summarization | 128 | 117 | affine-4 | 195.2 | 31.7 | 400 | 601ms | 1.4391 | 1.4997 | 0.0306 | 0.0318 | 8.86GB | 9.13GB | 12MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 235.3 | 31.9 | 400 | 1059ms | 1.4646 | 1.8250 | 0.0175 | 0.0357 | 8.86GB | 9.35GB | 13MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 253.2 | 31.5 | 400 | 2091ms | 1.4085 | 1.8105 | 0.0518 | 0.0251 | 8.86GB | 9.72GB | 20MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 263.1 | 31.1 | 400 | 4051ms | 1.4213 | 1.9827 | 0.0202 | 0.0474 | 8.86GB | 9.93GB | 33MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 267.5 | 30.8 | 400 | 7861ms | 1.5845 | 1.9387 | 0.0005 | 0.0361 | 8.86GB | 11.02GB | 40MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 287.8 | 30.0 | 400 | 14442ms | 1.4278 | 1.9482 | 0.0056 | 0.0347 | 8.86GB | 11.17GB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | affine-4 | 299.1 | 29.0 | 400 | 27610ms | 1.4022 | 2.3202 | 0.0236 | 0.0253 | 8.86GB | 11.49GB | 49MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-4 | 298.0 | 27.1 | 400 | 55168ms | 1.6318 | 2.0491 | 0.0234 | -0.0057 | 8.86GB | 12.07GB | 153MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | affine-4 | 288.6 | 23.6 | 400 | 113598ms | 1.5522 | 1.9745 | 0.0209 | 0.0499 | 8.86GB | 13.24GB | 316MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | affine-4 | 259.9 | 19.8 | 400 | 252299ms | 1.4105 | 1.7829 | 0.0297 | 0.0349 | 8.86GB | 16.24GB | 529MB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 203.5 | 14.4 | 400 | 642952ms | 1.6246 | 2.0381 | 0.0573 | 0.0577 | 8.86GB | 22.54GB | 1.15GB | The user wants a summary of the provided text. The input tex |
