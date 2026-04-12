# Inference Benchmark - Qwen3.5 9B

- **Date**: 2026-04-01 21:19
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
| summarization | 128 | 117 | no-quant | 198.4 | 31.7 | 342 | 591ms | 1.5485 | 1.8272 | 0.0237 | 0.0129 | 8.86GB | 9.13GB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 236.4 | 32.0 | 400 | 1054ms | 1.3197 | 2.0256 | 0.0008 | 0.0298 | 8.86GB | 9.35GB | 48MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 253.3 | 32.2 | 363 | 2131ms | 1.1999 | 1.6174 | 0.0195 | 0.0272 | 8.86GB | 9.72GB | 41MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 264.8 | 32.3 | 400 | 4065ms | 1.3163 | 1.7732 | 0.0181 | 0.0291 | 8.86GB | 10.28GB | 74MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 268.0 | 31.8 | 400 | 7857ms | 1.2854 | 1.9305 | 0.0248 | 0.0535 | 8.86GB | 11.17GB | 102MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 289.7 | 31.4 | 400 | 14395ms | 1.3545 | 2.0512 | 0.0423 | 0.0371 | 8.86GB | 11.28GB | 157MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 299.4 | 31.0 | 400 | 27636ms | 1.4267 | 2.3473 | 0.0288 | 0.0306 | 8.86GB | 11.69GB | 260MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 299.1 | 29.7 | 400 | 55122ms | 1.3741 | 2.1232 | 0.0126 | 0.0507 | 8.86GB | 12.43GB | 241MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 289.4 | 27.1 | 400 | 113439ms | 1.2506 | 1.7528 | -0.0023 | 0.0389 | 8.86GB | 13.85GB | 1.04GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 261.8 | 23.7 | 400 | 250619ms | 1.5174 | 1.8401 | -0.0034 | 0.0280 | 8.86GB | 16.90GB | 2.04GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | no-quant | 202.9 | 19.6 | 400 | 645164ms | 1.3539 | 1.7159 | 0.0098 | 0.0486 | 8.86GB | 22.54GB | 3.02GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
