# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 17:50
**Branch**: `ek/turbo-opt-0-fix-default-path`
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
- **GPU Baseline**: GPU memory with model weights loaded, before generation. **GPU Peak**: High-water mark during the run (includes transient computation tensors from prefill — attention scores, projections, activations — which dominate peak usage). **KV Delta**: MLX active memory increase after generation (noisy — affected by memory pool behavior, lazy evaluation, and allocation patterns). **KV Cache**: Deterministic KV cache size computed from token count, quantization config, and model dimensions — the true compressed footprint for reliable cross-config comparison.

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | turbo4 | 606.7 | 82.2 | 400 | 197ms | 2.9085 | 3.4766 | -0.0092 | 0.0351 | 1.86GB | 2.08GB | 14MB | 30MB | **Note on your input:** The text you provided appears to be  |
| summarization | 256 | 251 | turbo4 | 750.6 | 77.4 | 400 | 335ms | 3.0973 | 2.6593 | 0.0396 | 0.0285 | 1.86GB | 2.31GB | 15MB | 38MB | The text provided is the Table of Contents from F. Scott Fit |
| summarization | 512 | 506 | turbo4 | 948.8 | 76.9 | 400 | 534ms | 2.8387 | 2.3935 | 0.0202 | 0.0365 | 1.86GB | 2.71GB | 18MB | 53MB | This excerpt from F. Scott Fitzgerald's essay **"Once Again  |
| summarization | 1024 | 1021 | turbo4 | 1046.4 | 81.4 | 201 | 976ms | 3.0115 | 1.0510 | 0.0393 | 0.1203 | 1.86GB | 3.27GB | 17MB | 71MB | Here is a summary of the provided text, which includes F. Sc |
| summarization | 2048 | 2044 | turbo4 | 762.5 | 78.9 | 400 | 2751ms | 4.1861 | 3.1384 | 0.0391 | 0.0259 | 1.86GB | 3.97GB | 33MB | 142MB | Here is a summary of the provided excerpt from F. Scott Fitz |
| summarization | 4096 | 4087 | turbo4 | 1257.1 | 78.6 | 201 | 3305ms | 3.2100 | 1.7171 | 0.0272 | 0.0417 | 1.86GB | 3.95GB | 61MB | 249MB | Here is a summary of the excerpt from *The Great Gatsby*:  T |
| summarization | 8192 | 8192 | turbo4 | 1134.9 | 74.6 | 400 | 7271ms | 2.9847 | 1.0123 | 0.0270 | -0.0012 | 1.86GB | 4.04GB | 111MB | 499MB | Here is a summary of the provided text, which is **Chapter O |
| summarization | 16384 | 16363 | turbo4 | 1358.4 | 72.1 | 384 | 12152ms | 3.4687 | 1.1741 | 0.0043 | 0.0133 | 1.86GB | 4.39GB | 208MB | 973MB | This excerpt, "The Great Gatsby" (specifically the prologue, |
| summarization | 32768 | 32702 | turbo4 | 1201.9 | 63.8 | 400 | 27520ms | 3.5086 | 3.2009 | 0.0655 | 0.0316 | 1.86GB | 5.00GB | 399MB | 1.88GB | Here is a summary of the *Great Gatsby* excerpt provided:  T |
| summarization | 65536 | 65470 | turbo4 | 921.5 | 38.1 | 400 | 76297ms | 2.5879 | 1.4763 | 0.0354 | 0.0103 | 1.86GB | 6.34GB | 783MB | 3.74GB | Here is a summary of the provided text, *The Great Gatsby* b |
| summarization | 131072 | 130775 | turbo4 | 713.4 | 32.4 | 400 | 184460ms | 3.2306 | 2.5866 | 0.0155 | 0.0193 | 1.86GB | 8.63GB | 1.39GB | 7.44GB | Here is a summary of the key events and themes in "The Age o |
