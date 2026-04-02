# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-01 16:27
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-2B-bf16`

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
| summarization | 128 | 119 | no-quant | 731.8 | 59.1 | 400 | 164ms | 3.0585 | 2.6634 | — | — | 3.51GB | 3.78GB | 18MB | Based on the text you provided, here is a summary of its con |
| summarization | 256 | 251 | no-quant | 1424.4 | 58.8 | 201 | 177ms | 3.4977 | 1.7164 | — | — | 3.51GB | 3.94GB | 12MB | The text provided offers an excerpt from F. Scott Fitzgerald |
| summarization | 512 | 506 | no-quant | 1618.7 | 58.9 | 400 | 313ms | 2.8162 | 3.6807 | — | — | 3.51GB | 4.15GB | 21MB | This passage from *The Great Gatsby* begins with a reflectio |
| summarization | 1024 | 1021 | no-quant | 1746.1 | 59.9 | 400 | 585ms | 3.1785 | 3.1568 | — | — | 3.51GB | 4.68GB | 25MB | Here is a summary of the provided text, which serves as the  |
| summarization | 2048 | 2044 | no-quant | 1804.0 | 59.2 | 201 | 1139ms | 3.3808 | 1.5226 | — | — | 3.51GB | 5.50GB | 24MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* i |
| summarization | 4096 | 4087 | no-quant | 2067.9 | 58.3 | 208 | 1990ms | 3.0334 | 2.6541 | — | — | 3.51GB | 5.41GB | 51MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 8192 | 8192 | no-quant | 2196.7 | 57.0 | 400 | 3750ms | 3.1921 | 1.0120 | — | — | 3.51GB | 5.60GB | 111MB | **"The Great Gatsby" Summary**  Set in the opulent, isolated |
| summarization | 16384 | 16363 | no-quant | 2161.5 | 54.1 | 201 | 7659ms | 2.9155 | 1.2575 | — | — | 3.51GB | 5.89GB | 170MB | Here is a summary of the provided text, which consists of ** |
| summarization | 32768 | 32702 | no-quant | 1976.2 | 50.1 | 400 | 16836ms | 4.0047 | 2.4573 | — | — | 3.51GB | 6.51GB | 400MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 65536 | 65470 | no-quant | 1546.2 | 41.6 | 400 | 42730ms | 2.7341 | 3.4252 | — | — | 3.51GB | 7.84GB | 391MB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
| summarization | 131072 | 130775 | no-quant | 976.2 | 33.4 | 400 | 134373ms | 2.7258 | 3.1640 | — | — | 3.51GB | 10.22GB | 1.51GB | Based on the text provided, here is a summary of *The Great  |
