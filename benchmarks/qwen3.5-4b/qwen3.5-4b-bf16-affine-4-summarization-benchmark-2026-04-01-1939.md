# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-01 19:39
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-4B-bf16`

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
| summarization | 128 | 117 | affine-4 | 407.7 | 30.7 | 379 | 288ms | 1.6534 | 2.0986 | 0.0099 | 0.0258 | 7.83GB | 8.07GB | 34MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 583.8 | 30.4 | 380 | 427ms | 1.8960 | 1.9175 | 0.0582 | 0.0279 | 7.83GB | 8.21GB | 18MB | The user wants a summary of the provided text, which is an e |
| summarization | 512 | 504 | affine-4 | 694.6 | 29.8 | 400 | 726ms | 1.3783 | 1.6069 | 0.0224 | 0.0345 | 7.83GB | 8.55GB | 32MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 738.5 | 29.6 | 400 | 1416ms | 1.4095 | 1.8964 | 0.0430 | 0.0296 | 7.83GB | 8.88GB | 39MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 755.5 | 29.4 | 400 | 2837ms | 1.4330 | 2.2493 | 0.0399 | 0.0338 | 7.83GB | 9.95GB | 48MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 827.4 | 28.7 | 400 | 5074ms | 1.8227 | 1.9289 | 0.0001 | 0.0095 | 7.83GB | 10.10GB | 39MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | affine-4 | 842.3 | 27.6 | 400 | 9865ms | 2.0209 | 2.4952 | 0.0864 | 0.0467 | 7.83GB | 10.42GB | 102MB | The user wants a summary of the provided text, which is Chap |
| summarization | 16384 | 16361 | affine-4 | 812.9 | 25.9 | 400 | 20273ms | 2.2090 | 2.6737 | 0.0584 | 0.0585 | 7.83GB | 11.00GB | 151MB | The user wants a summary of the provided text, which is Chap |
| summarization | 32768 | 32700 | affine-4 | 726.2 | 23.3 | 400 | 45260ms | 2.0281 | 2.4901 | 0.0615 | 0.0601 | 7.83GB | 12.12GB | 317MB | The user wants a summary of the provided text, which is Chap |
| summarization | 65536 | 65468 | affine-4 | 558.5 | 19.3 | 400 | 117418ms | 2.1510 | 3.2327 | 0.0459 | 0.0505 | 7.83GB | 15.13GB | 458MB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 344.7 | 14.3 | 400 | 379658ms | 1.7262 | 1.7320 | 0.0247 | 0.0226 | 7.83GB | 21.44GB | 1.15GB | The user wants a summary of the text provided. The text cont |
