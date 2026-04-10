# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 18:37
**Branch**: `ek/turbo-opt-0-fix-default-path`
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

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | turbo3v2 | 620.5 | 98.3 | 400 | 193ms | 3.9991 | 4.5386 | 0.1126 | 0.0996 | 1010MB | 1.21GB | 16MB | 20MB | The text you provided is a two-part composition from F. Scot |
| summarization | 256 | 251 | turbo3v2 | 876.4 | 95.5 | 400 | 287ms | 3.3117 | 3.6858 | 0.1203 | 0.1906 | 1010MB | 1.43GB | 11MB | 24MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 512 | 506 | turbo3v2 | 988.1 | 96.9 | 360 | 513ms | 4.0302 | 1.3093 | 0.1639 | 0.0134 | 1010MB | 1.88GB | 22MB | 33MB | The provided text is a passage from F. Scott Fitzgerald's *T |
| summarization | 1024 | 1021 | turbo3v2 | 1053.9 | 95.9 | 400 | 969ms | 2.6886 | 4.0516 | 0.1454 | 0.1692 | 1010MB | 2.48GB | 13MB | 53MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 2048 | 2044 | turbo3v2 | 1078.5 | 96.7 | 400 | 1914ms | 2.4413 | 2.6593 | 0.1962 | 0.1724 | 1010MB | 3.18GB | 40MB | 92MB | This text presents the prologue to *The Great Gatsby* by F.  |
| summarization | 4096 | 4087 | turbo3v2 | 1248.9 | 95.4 | 400 | 3313ms | 2.8731 | 2.3028 | 0.1909 | 0.1107 | 1010MB | 3.14GB | 63MB | 169MB | This text is the opening chapter of the novel **The Great Ga |
| summarization | 8192 | 8192 | turbo3v2 | 1340.7 | 94.2 | 201 | 6153ms | 3.1391 | 1.6602 | 0.1434 | -0.0985 | 1010MB | 3.34GB | 71MB | 316MB | Here is a summary of the text from "The Great Gatsby" by F.  |
| summarization | 16384 | 16363 | turbo3v2 | 1353.9 | 87.9 | 400 | 12226ms | 3.6114 | 2.5795 | 0.2085 | 0.2497 | 1010MB | 3.62GB | 207MB | 630MB | Based on the text provided, here is a summary organized by t |
| summarization | 32768 | 32702 | turbo3v2 | 1274.1 | 72.4 | 400 | 26054ms | 3.6841 | 1.0776 | 0.1504 | 0.0275 | 1010MB | 4.18GB | 400MB | 1.22GB | Here is a summary of **The Great Gatsby** by F. Scott Fitzge |
| summarization | 65536 | 65470 | turbo3v2 | 1029.5 | 49.2 | 400 | 69104ms | 1.9033 | 3.5563 | 0.1078 | 0.1696 | 1010MB | 5.45GB | 783MB | 2.42GB | **The Great Gatsby - A Summary by F. Scott Fitzgerald**  Thi |
| summarization | 131072 | 130775 | turbo3v2 | 773.0 | 39.2 | 338 | 169656ms | 2.0603 | 1.0581 | 0.2247 | 0.0084 | 1011MB | 7.82GB | 1.51GB | 4.81GB | Based on the text provided, here is a summary of **The Age o |
