# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 18:15
**Branch**: `ek/turbo-opt-0-fix-default-path`
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

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 119 | turbo3v2 | 667.9 | 52.8 | 201 | 179ms | 2.5914 | 1.6939 | 0.0333 | 0.0038 | 3.51GB | 3.75GB | 15MB | 12MB | The text you provided contains two very different pieces. Th |
| summarization | 256 | 251 | turbo3v2 | 1152.7 | 55.0 | 379 | 218ms | 3.0559 | 2.6466 | 0.0208 | 0.0270 | 3.51GB | 3.94GB | 19MB | 24MB | Here is a summary of the content provided from Table of Cont |
| summarization | 512 | 506 | turbo3v2 | 1395.7 | 55.4 | 400 | 363ms | 2.6608 | 3.3904 | -0.0003 | 0.0256 | 3.51GB | 4.15GB | 18MB | 34MB | The text above is a reflection by F. Scott Fitzgerald on his |
| summarization | 1024 | 1021 | turbo3v2 | 1646.4 | 54.6 | 400 | 620ms | 2.9834 | 1.0359 | 0.0432 | -0.0045 | 3.51GB | 4.68GB | 23MB | 53MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 2048 | 2044 | turbo3v2 | 1648.5 | 54.2 | 201 | 1246ms | 3.2630 | 1.8047 | 0.0499 | 0.0003 | 3.51GB | 5.50GB | 36MB | 84MB | The provided text is **not** a summary of *The Great Gatsby* |
| summarization | 4096 | 4087 | turbo3v2 | 2023.1 | 54.5 | 400 | 2029ms | 2.3548 | 2.6970 | 0.0156 | 0.0379 | 3.51GB | 5.41GB | 53MB | 169MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 8192 | 8192 | turbo3v2 | 2160.2 | 52.1 | 400 | 3828ms | 3.3807 | 1.8028 | 0.0657 | 0.0280 | 3.51GB | 5.60GB | 75MB | 323MB | Here is a summary of the provided text, which is the first h |
| summarization | 16384 | 16363 | turbo3v2 | 2123.5 | 50.3 | 400 | 7816ms | 2.6093 | 4.0744 | 0.0123 | 0.0509 | 3.51GB | 5.89GB | 207MB | 630MB | This text presents an excerpt from F. Scott Fitzgerald's **" |
| summarization | 32768 | 32702 | turbo3v2 | 1963.5 | 46.8 | 400 | 16968ms | 2.3123 | 1.0083 | -0.0318 | 0.0008 | 3.51GB | 6.51GB | 399MB | 1.22GB | Here is a summary of the provided text, *The Great Gatsby* b |
| summarization | 65536 | 65470 | turbo3v2 | 1321.8 | 40.0 | 225 | 54541ms | 2.7344 | 1.3529 | 0.0574 | -0.0360 | 3.51GB | 7.79GB | 715MB | 2.41GB | This text is a **Table of Contents** and the first few chapt |
| summarization | 131072 | 130775 | turbo3v2 | 913.6 | 27.9 | 400 | 147316ms | 2.9755 | 3.7237 | 0.0553 | 0.0272 | 3.51GB | 10.22GB | 1.51GB | 4.82GB | This is a summary of the novel **"The Great Gatsby"** by F.  |
