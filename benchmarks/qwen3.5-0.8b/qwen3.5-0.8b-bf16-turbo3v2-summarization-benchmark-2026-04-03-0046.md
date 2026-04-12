# Inference Benchmark - Qwen3.5 0.8B

- **Date**: 2026-04-03 00:46
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: bf16
- **Model**: `mlx-community/Qwen3.5-0.8B-bf16`

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

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | turbo3v2 | 496.1 | 85.1 | 365 | 241ms | 3.9211 | 4.6003 | 0.0329 | 0.0828 | 1.40GB | 1.61GB | 16MB | 18MB | Based on the text provided, this is a **poem** by Zelda (pre |
| summarization | 256 | 251 | turbo3v2 | 2085.5 | 84.4 | 400 | 121ms | 4.4546 | 5.0395 | 0.0346 | 0.0124 | 1.40GB | 1.82GB | 19MB | 24MB | This text is an extract from **F. Scott Fitzgerald's** class |
| summarization | 512 | 506 | turbo3v2 | 2677.4 | 82.3 | 331 | 189ms | 3.8309 | 3.1418 | 0.0169 | 0.0430 | 1.40GB | 2.24GB | 22MB | 31MB | Based on the provided text, here is a summary of the content |
| summarization | 1024 | 1021 | turbo3v2 | 3214.4 | 84.0 | 400 | 318ms | 4.3647 | 4.2002 | 0.0197 | 0.0139 | 1.40GB | 2.76GB | 28MB | 53MB | Based on the provided text, here is a summary of *The Great  |
| summarization | 2048 | 2044 | turbo3v2 | 3554.9 | 84.7 | 201 | 575ms | 3.3384 | 2.0022 | 0.0334 | 0.2394 | 1.40GB | 3.51GB | 23MB | 84MB | Based on the provided text, here is a summary of **Once Agai |
| summarization | 4096 | 4087 | turbo3v2 | 4124.0 | 85.7 | 400 | 992ms | 5.1067 | 4.5528 | 0.0374 | 0.0617 | 1.40GB | 3.43GB | 52MB | 169MB | Based on the text provided from **Once Again to Zelda** by F |
| summarization | 8192 | 8192 | turbo3v2 | 4337.6 | 77.5 | 400 | 1902ms | 3.9882 | 3.3738 | 0.0208 | 0.0441 | 1.40GB | 3.72GB | 112MB | 323MB | This excerpt is from **Table 1** of F. Scott Fitzgerald's *T |
| summarization | 16384 | 16363 | turbo3v2 | 4062.6 | 78.1 | 201 | 4106ms | 4.0202 | 5.8952 | 0.0198 | 0.0042 | 1.40GB | 3.98GB | 205MB | 623MB | Based on the excerpt from *The Great Gatsby* by F. Scott Fit |
| summarization | 32768 | 32702 | turbo3v2 | 3354.4 | 66.9 | 400 | 10335ms | 3.7558 | 3.7111 | 0.0413 | 0.0222 | 1.40GB | 4.82GB | 366MB | 1.22GB | Based on the provided text of *The Great Gatsby* by F. Scott |
| summarization | 65536 | 65470 | turbo3v2 | 1691.0 | 40.8 | 400 | 43708ms | 3.9700 | 4.1682 | 0.0263 | 0.0171 | 1.40GB | 7.25GB | 782MB | 2.42GB | Based on the detailed excerpts provided, here is a summary o |
| summarization | 131072 | 130775 | turbo3v2 | 978.2 | 31.9 | 400 | 135214ms | 5.0563 | 3.7151 | -0.0326 | 0.0485 | 1.40GB | 8.29GB | 1.51GB | 4.82GB | Based on the text provided, which appears to be a reconstruc |
