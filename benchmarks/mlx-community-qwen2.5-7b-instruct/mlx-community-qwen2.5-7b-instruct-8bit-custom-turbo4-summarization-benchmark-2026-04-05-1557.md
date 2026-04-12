# Inference Benchmark - mlx-community/Qwen2.5-7B-Instruct-8bit

- **Date**: 2026-04-05 15:57
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
| summarization | 128 | 125 | turbo4 | 381.0 | 40.3 | 200 | 329ms | — | 1.8382 | — | — | 7.54GB | 7.70GB | 6MB | 19MB | The content you've provided appears to be a table of content |
| summarization | 256 | 254 | turbo4 | 466.2 | 39.8 | 149 | 545ms | — | 1.6615 | — | — | 7.54GB | 7.86GB | 10MB | 23MB | The excerpt provided is from the introduction to F. Scott Fi |
| summarization | 512 | 506 | turbo4 | 489.7 | 39.5 | 167 | 1034ms | — | 1.7902 | — | — | 7.54GB | 8.05GB | 28MB | 39MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald b |
| summarization | 1024 | 1020 | turbo4 | 519.2 | 39.2 | 200 | 2036ms | — | 1.7682 | — | — | 7.54GB | 8.33GB | 49MB | 71MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 2048 | 2035 | turbo4 | 515.1 | 38.2 | 200 | 4096ms | — | 1.6839 | — | — | 7.54GB | 8.73GB | 102MB | 130MB | The excerpt from F. Scott Fitzgerald's "The Great Gatsby" be |
| summarization | 4096 | 4031 | turbo4 | 517.5 | 37.6 | 200 | 7944ms | — | 1.5855 | — | — | 7.54GB | 8.83GB | 224MB | 246MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald s |
| summarization | 8192 | 8026 | turbo4 | 487.1 | 33.5 | 200 | 16663ms | — | 1.7935 | — | — | 7.54GB | 9.00GB | 440MB | 478MB | The excerpt from F. Scott Fitzgerald's "The Great Gatsby" in |
| summarization | 16384 | 16003 | turbo4 | 449.7 | 32.2 | 200 | 35842ms | — | 1.9148 | — | — | 7.54GB | 9.25GB | 795MB | 941MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald i |
| summarization | 32768 | 31929 | turbo4 | 388.2 | 27.2 | 200 | 82678ms | — | 1.9638 | — | — | 7.54GB | 10.01GB | 1.29GB | 1.82GB | This excerpt from F. Scott Fitzgerald's novel "The Great Gat |
| summarization | 65536 | 63738 | turbo4 | 291.6 | 17.6 | 200 | 222867ms | — | 1.8461 | — | — | 7.54GB | 11.59GB | 3.36GB | 3.63GB | The passage is a detailed and lengthy narrative about the de |
| summarization | 131072 | 128037 | turbo4 | 196.9 | 10.8 | 200 | 650582ms | — | 1.9583 | — | — | 7.54GB | 15.23GB | 5.87GB | 7.28GB | This passage appears to be a fragmented narrative about Newl |
