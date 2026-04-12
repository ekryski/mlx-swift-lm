# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 17:21
- **Branch**: `ek/turbo-opt-0-fix-default-path`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-2B-4bit`

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
| summarization | 128 | 119 | turbo3 | 636.2 | 94.7 | 319 | 188ms | 3.1812 | 4.6658 | 0.1510 | 0.2437 | 1010MB | 1.21GB | 16MB | 19MB | Based on the text provided, here is a summary of the work an |
| summarization | 256 | 251 | turbo3 | 867.9 | 93.2 | 400 | 290ms | 3.7632 | 3.7003 | 0.1954 | 0.1963 | 1010MB | 1.43GB | 14MB | 29MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 512 | 506 | turbo3 | 959.2 | 91.3 | 400 | 528ms | 3.5324 | 4.5640 | 0.1581 | 0.0702 | 1010MB | 1.88GB | 17MB | 40MB | Here is a concise summary of the provided text, *The Great G |
| summarization | 1024 | 1021 | turbo3 | 1047.2 | 98.2 | 201 | 975ms | 2.4119 | 1.1604 | 0.1840 | 0.0753 | 1010MB | 2.48GB | 20MB | 54MB | This text is an excerpt from the novel *The Great Gatsby* by |
| summarization | 2048 | 2044 | turbo3 | 1071.3 | 95.2 | 400 | 1927ms | 3.4359 | 2.6516 | 0.2409 | 0.0508 | 1010MB | 3.18GB | 33MB | 109MB | Here is a summary of *The Great Gatsby* and its opening chap |
| summarization | 4096 | 4087 | turbo3 | 1240.0 | 94.2 | 400 | 3340ms | 2.5144 | 5.1965 | 0.1904 | 0.1688 | 1010MB | 3.14GB | 52MB | 199MB | ### **The Great Gatsby (Table of Contents & Preface) Summary |
| summarization | 8192 | 8192 | turbo3 | 1340.3 | 95.3 | 239 | 6146ms | 4.2661 | 1.1378 | 0.2218 | 0.0316 | 1010MB | 3.34GB | 107MB | 375MB | This selection presents a narrative that bridges **The Great |
| summarization | 16384 | 16363 | turbo3 | 1349.4 | 86.6 | 400 | 12258ms | 3.6180 | 1.2249 | 0.3310 | 0.0270 | 1010MB | 3.62GB | 207MB | 745MB | This is a short story by F. Scott Fitzgerald, written in the |
| summarization | 32768 | 32702 | turbo3 | 1282.0 | 71.3 | 400 | 25908ms | 3.8949 | 2.5770 | 0.1609 | 0.1356 | 1010MB | 4.18GB | 400MB | 1.44GB | This text is a modern retelling of the classic novel **The G |
| summarization | 65536 | 65470 | turbo3 | 1090.5 | 56.0 | 201 | 61078ms | 4.6503 | 1.2466 | 0.2690 | 0.3138 | 1010MB | 5.47GB | 780MB | 2.85GB | This novel, written in the first person by Jay Gatz (the nar |
| summarization | 131072 | 130775 | turbo3 | 751.2 | 39.4 | 201 | 178052ms | 2.7599 | 3.8810 | 0.0918 | 0.1731 | 1011MB | 7.82GB | 1.01GB | 5.68GB | Here is a summary of the provided text, which includes F. Sc |
