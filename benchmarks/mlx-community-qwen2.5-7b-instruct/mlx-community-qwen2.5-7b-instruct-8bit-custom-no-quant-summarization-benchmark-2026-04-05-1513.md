# Inference Benchmark - mlx-community/Qwen2.5-7B-Instruct-8bit

**Date**: 2026-04-05 15:13
**Branch**: `ek/tom-eric-moe-tuning`
**Quantization**: custom
**Model**: `mlx-community/Qwen2.5-7B-Instruct-8bit`

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
| summarization | 128 | 125 | no-quant | 93.3 | 38.9 | 200 | 1356ms | — | 1.4251 | — | — | 7.54GB | 7.70GB | 4MB | 71MB | The content you've provided appears to be a mix of a table o |
| summarization | 256 | 254 | no-quant | 473.8 | 38.4 | 161 | 537ms | — | 1.6965 | — | — | 7.54GB | 7.86GB | 9MB | 91MB | The excerpt you've provided from F. Scott Fitzgerald's "The  |
| summarization | 512 | 506 | no-quant | 474.6 | 34.0 | 200 | 1067ms | — | 1.6971 | — | — | 7.54GB | 8.05GB | 20MB | 154MB | The excerpt from "The Great Gatsby" begins with a dedication |
| summarization | 1024 | 1020 | no-quant | 492.6 | 37.3 | 200 | 2153ms | — | 1.6558 | — | — | 7.54GB | 8.33GB | 48MB | 267MB | The excerpt from *The Great Gatsby* begins with a dedication |
| summarization | 2048 | 2035 | no-quant | 505.9 | 37.2 | 200 | 4151ms | — | 1.7629 | — | — | 7.54GB | 8.73GB | 108MB | 489MB | The excerpt from F. Scott Fitzgerald's "The Great Gatsby" in |
| summarization | 4096 | 4031 | no-quant | 507.1 | 37.4 | 200 | 8102ms | — | 1.8146 | — | — | 7.54GB | 8.83GB | 192MB | 926MB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald i |
| summarization | 8192 | 8026 | no-quant | 482.6 | 34.5 | 196 | 16822ms | — | 1.8634 | — | — | 7.54GB | 9.00GB | 448MB | 1.76GB | The passage from "The Great Gatsby" by F. Scott Fitzgerald i |
| summarization | 16384 | 16003 | no-quant | 440.5 | 31.3 | 200 | 36589ms | — | 1.9714 | — | — | 7.54GB | 9.25GB | 779MB | 3.46GB | The excerpt from "The Great Gatsby" by F. Scott Fitzgerald p |
| summarization | 32768 | 31929 | no-quant | 374.8 | 26.5 | 200 | 85666ms | — | 1.6962 | — | — | 7.54GB | 10.01GB | 1.50GB | 6.86GB | The excerpt from F. Scott Fitzgerald's "The Great Gatsby" pr |
| summarization | 65536 | 63738 | no-quant | 270.2 | 11.2 | 200 | 247503ms | — | 1.9188 | — | — | 7.54GB | 11.61GB | 3.11GB | 13.66GB | This passage from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 131072 | 128037 | no-quant | 195.0 | 10.7 | 153 | 656969ms | — | 2.3115 | — | — | 7.54GB | 15.23GB | 5.87GB | 27.38GB | The passage describes a visit Archer makes to the Marchiones |
