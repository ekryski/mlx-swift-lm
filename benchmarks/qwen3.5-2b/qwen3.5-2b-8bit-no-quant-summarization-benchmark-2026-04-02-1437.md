# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 14:37
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
| summarization | 128 | 119 | no-quant | 605.0 | 81.3 | 400 | 198ms | 2.6743 | 2.5010 | 0.0310 | 0.0249 | 1.86GB | 2.08GB | 19MB | 114MB | **Important Correction and Clarification**  The text you hav |
| summarization | 256 | 251 | no-quant | 825.3 | 77.9 | 400 | 305ms | 2.8792 | 2.3217 | 0.0021 | 0.0299 | 1.86GB | 2.31GB | 11MB | 142MB | The text you provided is a short passage titled **"Once Agai |
| summarization | 512 | 506 | no-quant | 961.9 | 78.8 | 400 | 527ms | 3.3010 | 2.3050 | 0.0340 | 0.0216 | 1.86GB | 2.71GB | 21MB | 198MB | This passage from **"The Great Gatsby"** (Section I) outline |
| summarization | 1024 | 1021 | no-quant | 1034.4 | 79.3 | 201 | 988ms | 2.7928 | 1.4320 | 0.0211 | 0.0695 | 1.86GB | 3.27GB | 24MB | 267MB | **1. The Core Message of My Father's Advice** The narrator r |
| summarization | 2048 | 2044 | no-quant | 1062.5 | 74.7 | 218 | 1959ms | 3.5798 | 2.6260 | 0.0589 | 0.0678 | 1.86GB | 3.97GB | 37MB | 495MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* s |
| summarization | 4096 | 4087 | no-quant | 1242.0 | 72.3 | 201 | 3356ms | 3.6081 | 1.1510 | 0.0191 | 0.3572 | 1.86GB | 3.95GB | 60MB | 938MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 8192 | 8192 | no-quant | 1194.4 | 75.1 | 335 | 6901ms | 2.9618 | 1.0363 | 0.0155 | 0.0071 | 1.86GB | 4.04GB | 112MB | 1.82GB | Here is a summary of the provided excerpt from F. Scott Fitz |
| summarization | 16384 | 16363 | no-quant | 1366.1 | 72.8 | 400 | 12096ms | 2.3026 | 1.2092 | 0.0104 | 0.0082 | 1.86GB | 4.39GB | 207MB | 3.58GB | This text is a chapter-by-chapter summary of F. Scott Fitzge |
| summarization | 32768 | 32702 | no-quant | 1252.0 | 64.6 | 211 | 26456ms | 2.6774 | 1.5687 | 0.0425 | 0.0596 | 1.86GB | 5.00GB | 397MB | 7.03GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 65536 | 65470 | no-quant | 977.8 | 43.0 | 400 | 74346ms | 2.6048 | 2.8569 | 0.0038 | 0.0339 | 1.86GB | 6.20GB | 783MB | 14.07GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | no-quant | 754.8 | 31.5 | 400 | 173834ms | 3.0999 | 2.8715 | 0.0135 | 0.0223 | 1.86GB | 8.63GB | 1.51GB | 28.02GB | Based on the provided texts, here is a summary of the two ma |
