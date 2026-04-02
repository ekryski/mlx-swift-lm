# Inference Benchmark - Qwen3.5 35B A3B

**Date**: 2026-04-02 09:09
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-35B-A3B-4bit`

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
| summarization | 128 | 117 | no-quant | 214.5 | 52.4 | 373 | 547ms | 1.5164 | 1.5853 | 0.1178 | 0.0178 | 18.16GB | 18.43GB | 40MB | The user wants a summary of the provided text.  1.  **Analyz |
| summarization | 256 | 249 | no-quant | 48.7 | 52.7 | 400 | 5387ms | 1.2404 | 1.5392 | 0.0424 | 0.0435 | 18.16GB | 18.60GB | 47MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 38.6 | 51.7 | 400 | 13450ms | 1.5361 | 1.8741 | 0.0649 | 0.1175 | 18.16GB | 18.94GB | 36MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | no-quant | 77.9 | 52.0 | 399 | 13535ms | 1.6400 | 1.4365 | 0.1270 | 0.1023 | 18.16GB | 19.65GB | 41MB | The user wants a summary of the provided text, which is the  |
| summarization | 2048 | 2042 | no-quant | 117.4 | 51.4 | 400 | 17813ms | 1.3939 | 1.8393 | 0.0729 | 0.1000 | 18.16GB | 20.72GB | 74MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 198.4 | 48.9 | 400 | 21036ms | 1.5345 | 1.0901 | 0.1110 | -0.0234 | 18.16GB | 20.80GB | 122MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 291.2 | 47.7 | 400 | 28619ms | 1.2590 | 1.5537 | 0.0568 | 0.1746 | 18.16GB | 21.15GB | 190MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 393.4 | 44.2 | 400 | 42086ms | 1.3248 | 1.8352 | 0.0719 | 0.1336 | 18.16GB | 21.75GB | 289MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 413.7 | 38.9 | 399 | 79559ms | 1.3738 | 1.3370 | 0.0679 | 0.0421 | 18.16GB | 22.98GB | 680MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 356.2 | 32.4 | 400 | 184422ms | 1.1637 | 1.5098 | 0.0442 | 0.1298 | 18.16GB | 25.60GB | 1.29GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 230.2 | 24.4 | 400 | 568448ms | 1.3364 | 1.6659 | 0.0576 | 0.0650 | 18.16GB | 30.29GB | 2.28GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
