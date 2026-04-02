# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-01 19:24
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: 4bit
**Model**: `mlx-community/Qwen3.5-2B-4bit`

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
| summarization | 128 | 119 | affine-4 | 633.9 | 111.3 | 400 | 189ms | 4.6689 | 1.0947 | 0.2540 | 0.0299 | 1010MB | 1.21GB | 11MB | This text presents the opening chapters of **"The Great Gats |
| summarization | 256 | 251 | affine-4 | 914.2 | 108.9 | 270 | 275ms | 4.5909 | 1.0739 | 0.2492 | 0.0716 | 1010MB | 1.43GB | 13MB | ### 📜 **Summary of the Provided Text**  The passage is a sam |
| summarization | 512 | 506 | affine-4 | 981.0 | 105.8 | 333 | 516ms | 3.2638 | 2.3875 | 0.1906 | 0.0960 | 1010MB | 1.88GB | 13MB | This text presents a personal reflection on the necessity of |
| summarization | 1024 | 1021 | affine-4 | 1041.0 | 105.7 | 400 | 981ms | 2.5602 | 1.8206 | 0.1587 | 0.1040 | 1010MB | 1.79GB | 15MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 2048 | 2044 | affine-4 | 1070.1 | 104.5 | 205 | 1929ms | 3.4307 | 5.8278 | 0.1232 | -0.3287 | 1010MB | 2.60GB | 9MB | Here is a summary of the provided text:  The text is the fir |
| summarization | 4096 | 4087 | affine-4 | 1231.1 | 101.9 | 400 | 3346ms | 3.6813 | 3.5303 | 0.2549 | 0.1932 | 1010MB | 2.67GB | 21MB | Here is a summary of the provided text from F. Scott Fitzger |
| summarization | 8192 | 8192 | affine-4 | 1327.6 | 96.4 | 400 | 6191ms | 3.2655 | 4.0255 | 0.1616 | 0.1688 | 1010MB | 2.83GB | 38MB | Here is a summary of *The Great Gatsby* by F. Scott Fitzgera |
| summarization | 16384 | 16363 | affine-4 | 1337.3 | 88.9 | 400 | 12259ms | 3.6474 | 1.0902 | 0.2346 | 0.0639 | 1010MB | 3.24GB | 44MB | Here is a summary of the narrative "The Great Gatsby" by F.  |
| summarization | 32768 | 32702 | affine-4 | 1275.7 | 76.8 | 206 | 25663ms | 3.6345 | 5.4059 | 0.2322 | 1.0379 | 1010MB | 3.67GB | 119MB | Here is a summary of **The Great Gatsby** by F. Scott Fitzge |
| summarization | 65536 | 65470 | affine-4 | 1090.9 | 59.7 | 400 | 60050ms | 3.0557 | 2.6669 | 0.2041 | 0.1032 | 1010MB | 5.00GB | 192MB | Based on the text provided, here is a summary of **F. Scott  |
| summarization | 131072 | 130775 | affine-4 | 793.8 | 43.3 | 400 | 164798ms | 4.0505 | 3.3603 | 0.3187 | 0.1751 | 1011MB | 7.82GB | 371MB | The provided text contains two of Edgar Allan Poe's famous w |
