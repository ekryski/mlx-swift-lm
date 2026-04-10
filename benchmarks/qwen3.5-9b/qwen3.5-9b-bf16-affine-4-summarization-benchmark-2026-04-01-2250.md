# Inference Benchmark - Qwen3.5 9B

- **Date**: 2026-04-01 22:50
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
| summarization | 128 | 117 | affine-4 | 216.1 | 18.5 | 351 | 543ms | 1.2934 | 2.0126 | 0.0469 | 0.0320 | 16.68GB | 16.88GB | 7MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | affine-4 | 317.2 | 18.4 | 370 | 785ms | 1.2262 | 1.6995 | 0.0128 | 0.0491 | 16.68GB | 17.02GB | 33MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | affine-4 | 382.6 | 18.3 | 400 | 1318ms | 1.2428 | 2.0949 | 0.0197 | 0.0356 | 16.68GB | 17.33GB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | affine-4 | 407.0 | 18.2 | 400 | 2770ms | 1.3629 | 1.8300 | 0.0449 | 0.0286 | 16.68GB | 17.75GB | 20MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | affine-4 | 423.7 | 17.8 | 400 | 5193ms | 1.3245 | 2.1978 | 0.0109 | 0.0387 | 16.68GB | 18.84GB | 36MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | affine-4 | 466.7 | 17.8 | 400 | 9123ms | 1.9060 | 1.9517 | 0.0375 | 0.0128 | 16.68GB | 18.99GB | 47MB | The user wants a summary of the provided text, which is the  |
| summarization | 8192 | 8190 | affine-4 | 480.0 | 17.5 | 400 | 17436ms | 1.5264 | 2.0337 | -0.0006 | 0.0434 | 16.68GB | 19.30GB | 75MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | affine-4 | 472.9 | 16.8 | 400 | 35013ms | 1.4738 | 2.5182 | 0.0289 | 0.0284 | 16.68GB | 19.88GB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | affine-4 | 443.6 | 15.6 | 400 | 74090ms | 1.6859 | 1.7583 | 0.0489 | 0.0494 | 16.68GB | 21.01GB | 316MB | The user wants a summary of the provided text. The text is C |
| summarization | 65536 | 65468 | affine-4 | 378.3 | 13.6 | 400 | 173437ms | 1.3587 | 1.8302 | -0.0102 | 0.0291 | 16.68GB | 23.97GB | 604MB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | affine-4 | 271.4 | 10.8 | 400 | 482023ms | 1.4494 | 1.5482 | 0.0526 | 0.0490 | 16.68GB | 30.34GB | 1.01GB | The user wants a summary of the provided text. The text cont |
