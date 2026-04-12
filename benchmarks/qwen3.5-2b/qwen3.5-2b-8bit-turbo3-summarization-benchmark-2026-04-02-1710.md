# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 17:10
- **Branch**: `ek/turbo-opt-0-fix-default-path`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | turbo3 | 601.4 | 81.6 | 398 | 199ms | 3.2394 | 3.0774 | 0.0349 | 0.0532 | 1.86GB | 2.08GB | 15MB | 23MB | The text you've provided is a fragment of **Will Smith's son |
| summarization | 256 | 251 | turbo3 | 849.7 | 80.2 | 288 | 296ms | 2.8657 | 3.9532 | 0.0211 | 0.0146 | 1.86GB | 2.31GB | 18MB | 24MB | The text provided is the **opening chapter** of *The Great G |
| summarization | 512 | 506 | turbo3 | 968.6 | 81.0 | 201 | 523ms | 2.7357 | 1.3376 | -0.0095 | 0.0827 | 1.86GB | 2.71GB | 19MB | 31MB | This passage from F. Scott Fitzgerald's **Table of Contents* |
| summarization | 1024 | 1021 | turbo3 | 1049.9 | 82.0 | 400 | 973ms | 3.0949 | 3.2568 | 0.0220 | 0.0285 | 1.86GB | 3.27GB | 27MB | 63MB | Based on the provided excerpt from *The Great Gatsby* by F.  |
| summarization | 2048 | 2044 | turbo3 | 1083.2 | 80.0 | 201 | 1912ms | 3.6377 | 1.3474 | 0.0458 | 0.0038 | 1.86GB | 3.97GB | 24MB | 100MB | Based on the text provided, here is a summary of the content |
| summarization | 4096 | 4087 | turbo3 | 1256.6 | 79.7 | 400 | 3301ms | 3.5197 | 2.0695 | 0.0263 | 0.0435 | 1.86GB | 3.95GB | 52MB | 199MB | This passage from *The Great Gatsby* is divided into two dis |
| summarization | 8192 | 8192 | turbo3 | 1353.4 | 76.4 | 201 | 6105ms | 3.4731 | 1.4865 | 0.0493 | 0.0342 | 1.86GB | 4.04GB | 109MB | 373MB | **Note:** The text provided contains significant formatting  |
| summarization | 16384 | 16363 | turbo3 | 1353.6 | 72.4 | 400 | 12216ms | 3.0639 | 3.0852 | 0.0213 | 0.0540 | 1.86GB | 4.39GB | 139MB | 745MB | Here is a summary of the provided text, which is primarily C |
| summarization | 32768 | 32702 | turbo3 | 1267.0 | 62.8 | 201 | 26158ms | 2.5310 | 1.7844 | 0.0237 | 0.0448 | 1.86GB | 5.00GB | 395MB | 1.43GB | Here is a summary of the provided text, *The Great Gatsby* b |
| summarization | 65536 | 65470 | turbo3 | 1086.8 | 40.4 | 400 | 63337ms | 3.4881 | 2.4945 | 0.0048 | 0.0135 | 1.86GB | 6.34GB | 784MB | 2.86GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | turbo3 | 752.2 | 30.5 | 400 | 175346ms | 3.4461 | 3.1820 | 0.0586 | 0.0379 | 1.86GB | 8.63GB | 1.51GB | 5.69GB | Here is a summary of the two novellas presented in your text |
