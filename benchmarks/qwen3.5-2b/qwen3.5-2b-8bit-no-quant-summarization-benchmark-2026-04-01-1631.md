# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-01 16:31
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | no-quant | 624.4 | 88.7 | 201 | 192ms | 3.1765 | 1.2985 | 0.0388 | 0.0028 | 1.86GB | 2.08GB | 15MB | The text provided is the opening scene and first stanza of F |
| summarization | 256 | 251 | no-quant | 893.3 | 88.2 | 400 | 281ms | 4.5337 | 2.9455 | 0.0405 | 0.0187 | 1.86GB | 2.31GB | 6MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 512 | 506 | no-quant | 995.2 | 88.1 | 400 | 509ms | 3.3722 | 3.1291 | -0.0049 | 0.0136 | 1.86GB | 2.71GB | 10MB | This excerpt from *The Great Gatsby*, specifically the openi |
| summarization | 1024 | 1021 | no-quant | 1057.8 | 89.0 | 400 | 966ms | 2.2763 | 1.1216 | 0.0084 | 0.0198 | 1.86GB | 3.27GB | 9MB | Based on the provided text from F. Scott Fitzgerald's *The G |
| summarization | 2048 | 2044 | no-quant | 1096.2 | 87.1 | 400 | 1890ms | 2.8682 | 2.3456 | 0.0377 | -0.0029 | 1.86GB | 3.97GB | 0MB | Here is a summary of the provided text, *The Great Gatsby* b |
| summarization | 4096 | 4087 | no-quant | 1263.3 | 85.3 | 201 | 3295ms | 2.9327 | 1.9470 | 0.0167 | 0.0937 | 1.86GB | 3.95GB | 39MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 8192 | 8192 | no-quant | 1354.0 | 82.6 | 400 | 6116ms | 3.0880 | 3.2780 | 0.0188 | 0.0336 | 1.86GB | 4.04GB | 55MB | This section of *The Great Gatsby* serves as the narrative f |
| summarization | 16384 | 16363 | no-quant | 1365.1 | 77.6 | 400 | 12109ms | 3.2458 | 2.8284 | 0.0472 | -0.0024 | 1.86GB | 4.39GB | 172MB | This summary captures the narrative arc and key events of F. |
| summarization | 32768 | 32702 | no-quant | 1302.1 | 68.6 | 400 | 25431ms | 2.2085 | 1.0284 | 0.0160 | 0.0065 | 1.86GB | 5.00GB | 333MB | Here is a summary of the provided text, which consists of ch |
| summarization | 65536 | 65470 | no-quant | 1111.2 | 52.5 | 400 | 59377ms | 2.5334 | 1.0351 | 0.0272 | 0.0004 | 1.86GB | 6.34GB | 718MB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
| summarization | 131072 | 130775 | no-quant | 791.8 | 40.6 | 400 | 165619ms | 3.4322 | 1.0081 | 0.0346 | -0.0000 | 1.86GB | 8.63GB | 1.26GB | This summary outlines the narrative arc and key thematic shi |
