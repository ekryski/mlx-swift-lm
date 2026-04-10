# Inference Benchmark - Qwen3.5 4B

**Date**: 2026-04-03 05:30
**Branch**: `ek/consolidated-benchmarks`
**Quantization**: bf16
**Model**: `mlx-community/Qwen3.5-4B-bf16`

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
| summarization | 128 | 117 | turbo3 | 394.9 | 29.6 | 400 | 297ms | 1.5991 | 1.6694 | 0.0195 | 0.0202 | 7.83GB | 8.07GB | 37MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo3 | 576.4 | 29.5 | 400 | 432ms | 1.3928 | 1.9203 | 0.0228 | 0.0126 | 7.83GB | 8.21GB | 43MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo3 | 696.4 | 29.3 | 400 | 724ms | 2.3800 | 2.2061 | 0.0259 | 0.0312 | 7.83GB | 8.55GB | 55MB | 40MB | The user wants a summary of the provided text from "The Grea |
| summarization | 1024 | 1019 | turbo3 | 761.8 | 29.4 | 400 | 1403ms | 1.8892 | 1.8767 | 0.0194 | 0.0387 | 7.83GB | 9.17GB | 70MB | 63MB | The user wants a summary of the provided text, which is Chap |
| summarization | 2048 | 2042 | turbo3 | 786.3 | 29.4 | 400 | 2704ms | 1.9367 | 1.9744 | 0.0372 | 0.0338 | 7.83GB | 10.07GB | 104MB | 109MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 4096 | 4085 | turbo3 | 834.4 | 29.1 | 400 | 5041ms | 2.2104 | 2.1274 | 0.0300 | 0.0256 | 7.83GB | 10.27GB | 167MB | 199MB | The user wants a summary of the provided text. The text is C |
| summarization | 8192 | 8190 | turbo3 | 850.9 | 28.5 | 400 | 9799ms | 1.8572 | 2.8052 | 0.0330 | 0.0574 | 7.83GB | 10.67GB | 294MB | 382MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | turbo3 | 820.0 | 27.4 | 400 | 20236ms | 1.9550 | 2.0131 | 0.0561 | 0.0500 | 7.83GB | 11.37GB | 415MB | 745MB | The user wants a summary of the provided text. The text prov |
| summarization | 32768 | 32700 | turbo3 | 741.2 | 25.4 | 400 | 44489ms | 1.8773 | 1.8291 | 0.0368 | 0.0017 | 7.83GB | 12.77GB | 1.04GB | 1.44GB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 65536 | 65468 | turbo3 | 567.0 | 22.4 | 400 | 115899ms | 1.6263 | 2.7753 | 0.0045 | 0.0278 | 7.83GB | 15.70GB | 2.04GB | 2.86GB | Here's a thinking process that leads to the summary:  1.  ** |
| summarization | 131072 | 130773 | turbo3 | 366.0 | 18.4 | 400 | 357791ms | 1.8038 | 2.0865 | 0.0473 | 0.0595 | 7.83GB | 21.44GB | 4.03GB | 5.69GB | The user wants a summary of the provided text. The text cons |
