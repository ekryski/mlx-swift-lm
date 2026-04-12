# Inference Benchmark - Qwen3.5 9B

- **Date**: 2026-04-01 21:01
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: bf16
- **Model**: `mlx-community/Qwen3.5-9B-MLX-bf16`

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
| summarization | 128 | 117 | no-quant | 202.8 | 18.3 | 400 | 578ms | 1.3776 | 1.8568 | — | — | 16.68GB | 16.98GB | 46MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 371.1 | 18.5 | 400 | 671ms | 1.2178 | 2.1208 | — | — | 16.68GB | 17.02GB | 41MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 421.3 | 18.6 | 400 | 1261ms | 1.3720 | 1.4542 | — | — | 16.68GB | 17.33GB | 58MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 437.3 | 18.6 | 400 | 2580ms | 1.4220 | 1.7770 | — | — | 16.68GB | 17.91GB | 62MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 443.0 | 18.5 | 400 | 4946ms | 1.3805 | 2.1014 | — | — | 16.68GB | 18.90GB | 105MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 468.3 | 18.4 | 400 | 9116ms | 1.2066 | 1.9109 | — | — | 16.68GB | 19.10GB | 167MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 484.3 | 18.1 | 400 | 17304ms | 1.7443 | 1.9178 | — | — | 16.68GB | 19.51GB | 258MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 475.9 | 17.8 | 400 | 34802ms | 1.6286 | 1.7133 | — | — | 16.68GB | 20.26GB | 554MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 447.6 | 16.9 | 400 | 73522ms | 1.6834 | 1.7905 | — | — | 16.68GB | 21.59GB | 1.04GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 380.6 | 15.6 | 400 | 172455ms | 1.3242 | 2.1337 | — | — | 16.68GB | 24.59GB | 2.04GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 267.7 | 13.6 | 400 | 489000ms | 1.9713 | 1.6311 | — | — | 16.68GB | 30.34GB | 4.03GB | The user wants a summary of the provided text, which consist |
