# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 20:23
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
| summarization | 128 | 119 | turbo4v2-p2 | 598.0 | 79.4 | 400 | 200ms | 3.4332 | 2.4008 | 0.0285 | 0.0339 | 1.86GB | 2.08GB | 11MB | 23MB | The text provided is a **Table of Contents** from the novel  |
| summarization | 256 | 251 | turbo4v2-p2 | 870.6 | 78.4 | 201 | 289ms | 3.4638 | 1.8499 | 0.0492 | 0.0356 | 1.86GB | 2.31GB | 10MB | 20MB | The text provided is an introduction to F. Scott Fitzgerald' |
| summarization | 512 | 506 | turbo4v2-p2 | 955.3 | 78.2 | 201 | 530ms | 3.4855 | 1.2451 | 0.0407 | 0.4224 | 1.86GB | 2.71GB | 16MB | 31MB | This passage is an early reflection by F. Scott Fitzgerald o |
| summarization | 1024 | 1021 | turbo4v2-p2 | 1039.2 | 80.4 | 400 | 984ms | 2.5940 | 1.1301 | 0.0090 | 0.0048 | 1.86GB | 3.27GB | 23MB | 63MB | Based on the provided text from *The Great Gatsby* by F. Sco |
| summarization | 2048 | 2044 | turbo4v2-p2 | 1069.7 | 76.6 | 201 | 1939ms | 3.5114 | 1.2507 | 0.0314 | 0.5338 | 1.86GB | 3.97GB | 35MB | 100MB | This excerpt from *The Great Gatsby*, specifically chapters  |
| summarization | 4096 | 4087 | turbo4v2-p2 | 1237.2 | 78.9 | 201 | 3354ms | 3.2716 | 1.4972 | 0.0510 | 0.0380 | 1.86GB | 3.95GB | 61MB | 191MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 8192 | 8192 | turbo4v2-p2 | 1341.7 | 77.5 | 400 | 6151ms | 2.3227 | 3.2407 | 0.0060 | 0.0282 | 1.86GB | 4.04GB | 112MB | 382MB | Here is a summary of the provided text, **"The Great Gatsby" |
| summarization | 16384 | 16363 | turbo4v2-p2 | 1343.3 | 72.4 | 400 | 12292ms | 4.0977 | 3.5954 | 0.0198 | 0.0366 | 1.86GB | 4.39GB | 207MB | 745MB | This story, **"Once Again to Zelda"**, written by F. Scott F |
| summarization | 32768 | 32702 | turbo4v2-p2 | 1298.2 | 64.4 | 400 | 25512ms | 2.3497 | 3.2842 | 0.0432 | 0.0375 | 1.86GB | 5.00GB | 333MB | 1.44GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 65536 | 65470 | turbo4v2-p2 | 1046.8 | 40.1 | 400 | 67862ms | 3.3546 | 1.0761 | 0.0063 | 0.0023 | 1.86GB | 6.34GB | 783MB | 2.86GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | turbo4v2-p2 | 741.9 | 29.8 | 400 | 176939ms | 3.4852 | 2.3651 | 0.0146 | 0.0246 | 1.86GB | 8.63GB | 1.51GB | 5.69GB | This text is not a real book; it combines elements of **F. S |
