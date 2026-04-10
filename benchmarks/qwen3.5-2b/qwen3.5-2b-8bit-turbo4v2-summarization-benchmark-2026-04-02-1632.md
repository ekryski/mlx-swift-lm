# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 16:32
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

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | turbo4v2 | 610.2 | 78.1 | 390 | 196ms | 2.9328 | 1.9969 | 0.0333 | 0.0296 | 1.86GB | 2.08GB | 15MB | 23MB | The text provided is a short note titled "**Once again to Ze |
| summarization | 256 | 251 | turbo4v2 | 847.1 | 78.0 | 292 | 297ms | 3.6858 | 4.1104 | 0.0221 | 0.0249 | 1.86GB | 2.31GB | 18MB | 24MB | The excerpt from **"Once Again to Zelda"** by F. Scott Fitzg |
| summarization | 512 | 506 | turbo4v2 | 965.4 | 77.3 | 400 | 525ms | 3.0184 | 3.9683 | 0.0482 | 0.0283 | 1.86GB | 2.71GB | 18MB | 40MB | F. Scott Fitzgerald's excerpt from the opening chapter of *T |
| summarization | 1024 | 1021 | turbo4v2 | 1010.4 | 78.5 | 400 | 1011ms | 3.0339 | 2.9321 | 0.0186 | 0.0433 | 1.86GB | 3.27GB | 27MB | 63MB | Here is a summary of the provided text, which serves as a ta |
| summarization | 2048 | 2044 | turbo4v2 | 1081.8 | 80.8 | 400 | 1957ms | 2.6994 | 2.8315 | 0.0253 | 0.0378 | 1.86GB | 3.97GB | 40MB | 109MB | **Zelda Fitzgerald's Tribute to "The Great Gatsby"**  **Tabl |
| summarization | 4096 | 4087 | turbo4v2 | 1236.6 | 79.3 | 201 | 3357ms | 3.4057 | 1.3863 | -0.0133 | 0.0897 | 1.86GB | 3.95GB | 41MB | 191MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 8192 | 8192 | turbo4v2 | 1340.0 | 77.0 | 400 | 6162ms | 2.1210 | 2.4234 | 0.0189 | 0.0366 | 1.86GB | 4.04GB | 92MB | 382MB | Here is a summary of the provided text, which includes the e |
| summarization | 16384 | 16363 | turbo4v2 | 1357.0 | 73.1 | 400 | 12172ms | 2.5724 | 2.5548 | 0.0108 | 0.0279 | 1.86GB | 4.39GB | 207MB | 745MB | Here is a summary of the provided text, structured by the ke |
| summarization | 32768 | 32702 | turbo4v2 | 1293.7 | 63.1 | 400 | 25666ms | 3.2566 | 1.0076 | 0.0216 | 0.0004 | 1.86GB | 5.00GB | 333MB | 1.44GB | This summary covers the narrative arc of F. Scott Fitzgerald |
| summarization | 65536 | 65470 | turbo4v2 | 1083.3 | 47.0 | 400 | 62104ms | 3.0880 | 1.0091 | 0.0053 | 0.0004 | 1.86GB | 6.34GB | 784MB | 2.86GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | turbo4v2 | 733.5 | 31.5 | 400 | 181340ms | 2.7845 | 1.7736 | 0.0414 | 0.0161 | 1.86GB | 8.63GB | 1.51GB | 5.69GB | Here is a summary of the plot from F. Scott Fitzgerald's *Th |
