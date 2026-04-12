# Inference Benchmark - Qwen3.5 35B A3B

- **Date**: 2026-04-02 08:44
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-35B-A3B-8bit`

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
| summarization | 128 | 117 | no-quant | 43.8 | 44.0 | 400 | 2848ms | 1.1875 | 1.6269 | — | — | 34.30GB | 34.61GB | 47MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | no-quant | 323.2 | 43.3 | 400 | 771ms | 1.2174 | 1.3709 | — | — | 34.30GB | 34.70GB | 45MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | no-quant | 415.3 | 42.8 | 400 | 1253ms | 1.3249 | 1.5980 | — | — | 34.30GB | 35.07GB | 52MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | no-quant | 433.9 | 43.6 | 400 | 2725ms | 1.2995 | 1.6386 | — | — | 34.30GB | 35.78GB | 60MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | no-quant | 461.4 | 43.3 | 400 | 4939ms | 1.3287 | 1.4041 | — | — | 34.30GB | 36.86GB | 56MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | no-quant | 520.0 | 42.5 | 399 | 8458ms | 1.3601 | 1.5459 | — | — | 34.30GB | 36.93GB | 122MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 8192 | 8190 | no-quant | 550.8 | 41.4 | 400 | 15389ms | 1.2164 | 1.6427 | — | — | 34.30GB | 37.29GB | 182MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 16384 | 16361 | no-quant | 539.2 | 39.4 | 400 | 30843ms | 1.2635 | 1.4233 | — | — | 34.30GB | 37.88GB | 326MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 32768 | 32700 | no-quant | 498.5 | 36.1 | 400 | 66178ms | 1.4937 | 1.7937 | — | — | 34.30GB | 39.12GB | 646MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 65536 | 65468 | no-quant | 398.5 | 30.6 | 400 | 164738ms | 1.2919 | 1.5584 | — | — | 34.30GB | 41.74GB | 1.29GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 131072 | 130773 | no-quant | 243.8 | 22.8 | 400 | 536796ms | 1.3647 | 1.4579 | — | — | 34.30GB | 46.43GB | 1.52GB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
