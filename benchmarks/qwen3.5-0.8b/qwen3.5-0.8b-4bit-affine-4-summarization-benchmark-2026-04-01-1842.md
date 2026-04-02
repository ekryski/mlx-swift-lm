# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-01 18:42
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-0.8B-4bit`

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
| summarization | 128 | 119 | affine-4 | 1068.5 | 141.6 | 201 | 113ms | 6.3083 | 1.3813 | 0.2127 | 1.0629 | 404MB | 602MB | 13MB | The summary you've provided appears to be a specific excerpt |
| summarization | 256 | 251 | affine-4 | 1835.3 | 140.1 | 400 | 137ms | 4.8061 | 3.0389 | 0.2562 | 0.1952 | 404MB | 803MB | 11MB | Based on the provided text, here is a summary of the content |
| summarization | 512 | 506 | affine-4 | 2148.7 | 127.7 | 400 | 236ms | 4.3654 | 2.7680 | 0.2756 | 0.2132 | 404MB | 1.19GB | 12MB | ### Summary  This text is an anonymous letter written by F.  |
| summarization | 1024 | 1021 | affine-4 | 2127.3 | 117.8 | 400 | 480ms | 3.4577 | 4.4846 | 0.2774 | 0.2899 | 404MB | 1.18GB | 6MB | Based on the text provided from *The Great Gatsby* by F. Sco |
| summarization | 2048 | 2044 | affine-4 | 2263.0 | 122.0 | 352 | 904ms | 4.0463 | 6.4621 | 0.3660 | 0.1848 | 404MB | 1.97GB | 9MB | Here is a summary of the text *The Great Gatsby*:  **Summary |
| summarization | 4096 | 4087 | affine-4 | 2692.8 | 118.5 | 400 | 1518ms | 4.3829 | 5.0621 | 0.2289 | 0.2505 | 404MB | 2.05GB | 4MB | Based on the text provided by F. Scott Fitzgerald, here is a |
| summarization | 8192 | 8192 | affine-4 | 2917.0 | 109.7 | 227 | 2809ms | 4.7890 | 3.7887 | 0.3064 | 0.2690 | 404MB | 2.20GB | 38MB | Based on the provided text *Once Again*, Zradya Lazarus pres |
| summarization | 16384 | 16363 | affine-4 | 2841.2 | 101.8 | 201 | 5761ms | 3.6678 | 1.7964 | 0.2614 | 0.4434 | 404MB | 2.53GB | 33MB | Based on the text *Once Again to Zelda* by F. Scott Fitzgera |
| summarization | 32768 | 32702 | affine-4 | 2496.9 | 87.3 | 222 | 13108ms | 3.6727 | 3.3995 | 0.2969 | 0.3750 | 404MB | 3.75GB | 118MB | Here is a summary of the excerpts provided:  **Summary by Au |
| summarization | 65536 | 65470 | affine-4 | 1835.3 | 68.7 | 400 | 35699ms | 3.9278 | 3.4876 | 0.2922 | 0.2716 | 405MB | 6.25GB | 227MB | Based on the text provided, which is Fitzgerald's essay "Onc |
| summarization | 131072 | 130775 | affine-4 | 1132.2 | 43.1 | 400 | 115540ms | 3.0519 | 3.3284 | 0.1577 | 0.1612 | 405MB | 9.06GB | 409MB | Based on the text provided, which is an excerpt from F. Scot |
