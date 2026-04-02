# Inference Benchmark - Qwen3.5 0.8B

**Date**: 2026-04-01 18:34
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-0.8B-8bit`

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
| summarization | 128 | 119 | affine-4 | 1062.4 | 116.6 | 270 | 113ms | 4.2555 | 4.6154 | 0.0309 | 0.0000 | 763MB | 960MB | 14MB | Based on the text you provided, here is a summary of the sce |
| summarization | 256 | 251 | affine-4 | 1778.5 | 114.7 | 400 | 142ms | 4.0932 | 2.4107 | 0.0469 | 0.0570 | 763MB | 1.13GB | 0MB | Based on the text provided, here is a summary:  The excerpt  |
| summarization | 512 | 506 | affine-4 | 2122.8 | 113.5 | 201 | 239ms | 3.5026 | 4.8322 | 0.0517 | 0.2309 | 763MB | 1.52GB | 0MB | The text you provided is a **metaphorical and psychological  |
| summarization | 1024 | 1021 | affine-4 | 2160.3 | 108.4 | 400 | 473ms | 3.7114 | 3.8289 | 0.0420 | 0.0637 | 763MB | 1.53GB | 12MB | Based on the provided text from *The Great Gatsby*, here is  |
| summarization | 2048 | 2044 | affine-4 | 2266.7 | 109.4 | 201 | 902ms | 4.2915 | 3.9883 | 0.0479 | 0.4804 | 763MB | 2.32GB | 14MB | This text is an excerpt from *The Great Gatsby* by F. Scott  |
| summarization | 4096 | 4087 | affine-4 | 2709.4 | 106.7 | 400 | 1509ms | 5.3652 | 4.5695 | 0.0356 | 0.0555 | 763MB | 2.40GB | 20MB | Here is a summary of the provided text, *The Great Gatsby*:  |
| summarization | 8192 | 8192 | affine-4 | 2945.0 | 100.4 | 400 | 2782ms | 4.7323 | 5.9078 | 0.0320 | 0.0874 | 763MB | 2.55GB | 26MB | Based on the provided text from *The Great Gatsby*, here is  |
| summarization | 16384 | 16363 | affine-4 | 2897.7 | 97.4 | 400 | 5647ms | 4.5370 | 1.0301 | 0.0586 | 0.0030 | 763MB | 2.88GB | 53MB | This is a dramatic, character-driven account of the events f |
| summarization | 32768 | 32702 | affine-4 | 2534.6 | 78.6 | 400 | 12923ms | 4.0214 | 5.5090 | 0.0730 | 0.0276 | 763MB | 4.10GB | 99MB | Based on *The Great Gatsby*, the provided text offers a comp |
| summarization | 65536 | 65470 | affine-4 | 1917.3 | 60.9 | 400 | 34179ms | 4.3748 | 5.1171 | 0.1114 | 0.0954 | 763MB | 6.60GB | 189MB | Based on the text provided from F. Scott Fitzgerald's *The G |
| summarization | 131072 | 130775 | affine-4 | 1189.7 | 42.9 | 400 | 109959ms | 4.3067 | 4.9833 | 0.0506 | 0.0353 | 763MB | 9.41GB | 373MB | This text is a long, fragmented essay by **F. Scott Fitzgera |
