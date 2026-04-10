# Inference Benchmark - mlx-community/Qwen2.5-7B-Instruct-8bit

- **Date**: 2026-04-05 16:16
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: custom
- **Model**: `mlx-community/Qwen2.5-7B-Instruct-8bit`

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
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 125 | turbo4v2 | 384.8 | 40.1 | 200 | 326ms | — | 1.5918 | — | — | 7.54GB | 7.70GB | 6MB | 14MB | The provided content appears to be a mix of two different el |
| summarization | 256 | 254 | turbo4v2 | 468.5 | 39.8 | 154 | 543ms | — | 1.5896 | — | — | 7.54GB | 7.86GB | 14MB | 18MB | The excerpt begins with a dedication to Zelda Fitzgerald, th |
| summarization | 512 | 506 | turbo4v2 | 473.9 | 38.7 | 194 | 1068ms | — | 1.6954 | — | — | 7.54GB | 8.05GB | 28MB | 31MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald b |
| summarization | 1024 | 1020 | turbo4v2 | 516.9 | 39.2 | 200 | 2079ms | — | 1.6218 | — | — | 7.54GB | 8.33GB | 45MB | 54MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 2048 | 2035 | turbo4v2 | 513.6 | 37.8 | 200 | 4088ms | — | 1.6729 | — | — | 7.54GB | 8.73GB | 36MB | 99MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald b |
| summarization | 4096 | 4031 | turbo4v2 | 510.8 | 36.7 | 200 | 8071ms | — | 1.7968 | — | — | 7.54GB | 8.83GB | 200MB | 188MB | The excerpt from F. Scott Fitzgerald's "The Great Gatsby" in |
| summarization | 8192 | 8026 | turbo4v2 | 481.3 | 34.9 | 200 | 16875ms | — | 1.8302 | — | — | 7.54GB | 9.00GB | 392MB | 366MB | The excerpt from *The Great Gatsby* by F. Scott Fitzgerald i |
| summarization | 16384 | 16003 | turbo4v2 | 443.2 | 31.2 | 200 | 36392ms | — | 1.8274 | — | — | 7.54GB | 9.25GB | 858MB | 720MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald i |
| summarization | 32768 | 31929 | turbo4v2 | 385.7 | 27.2 | 200 | 83236ms | — | 1.7101 | — | — | 7.54GB | 10.01GB | 1.54GB | 1.39GB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald p |
| summarization | 65536 | 63738 | turbo4v2 | 293.2 | 17.6 | 200 | 220980ms | — | 1.6913 | — | — | 7.54GB | 11.59GB | 3.23GB | 2.77GB | The passage is a detailed account of the events surrounding  |
| summarization | 131072 | 128037 | turbo4v2 | 197.0 | 10.7 | 200 | 650267ms | — | 2.2038 | — | — | 7.54GB | 15.23GB | 6.85GB | 5.56GB | The passage describes a series of interactions between Newla |
