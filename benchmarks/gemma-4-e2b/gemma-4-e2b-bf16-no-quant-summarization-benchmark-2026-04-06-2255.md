# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-06 22:55
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: bf16
- **Model**: `mlx-community/gemma-4-e2b-it-bf16`

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
| Top K | 64 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 116 | no-quant | 219.8 | 13.0 | 400 | 530ms | 1.4253 | — | — | — | 8.66GB | 11.55GB | 10MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 570.5 | 13.0 | 400 | 447ms | 1.5806 | 1.8486 | — | — | 8.66GB | 11.69GB | 14MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 811.8 | 12.9 | 400 | 619ms | 1.2639 | — | — | — | 8.66GB | 11.94GB | 14MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 915.0 | 12.9 | 400 | 1110ms | 1.4111 | — | — | — | 8.66GB | 12.44GB | 21MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 830.0 | 12.8 | 400 | 2667ms | 1.5274 | — | — | — | 8.66GB | 13.46GB | 23MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1342.5 | 12.8 | 400 | 3244ms | 1.4745 | — | — | — | 8.66GB | 13.50GB | 57MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8229 | no-quant | 3511.0 | 12.7 | 255 | 2358ms | — | 1.7662 | — | — | 8.66GB | 11.46GB | 107MB | 1.81GB | This excerpt from *The Great Gatsby* focuses on Nick Carrawa |
| summarization | 16384 | 16395 | no-quant | 3592.6 | 12.4 | 400 | 4622ms | — | 1.9888 | — | — | 8.66GB | 11.63GB | 204MB | 3.59GB | The provided text is an excerpt from F. Scott Fitzgerald's * |
| summarization | 32768 | 32815 | no-quant | 3152.5 | 11.8 | 400 | 10552ms | — | 1.8817 | — | — | 8.66GB | 12.28GB | 392MB | 7.10GB | This excerpt is a collection of narrative passages drawn fro |
| summarization | 65536 | 65896 | no-quant | 1837.2 | 10.9 | 400 | 36108ms | — | 2.0132 | — | — | 8.66GB | 14.66GB | 778MB | 14.16GB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 131072 | 130563 | no-quant | 1013.5 | 10.6 | 400 | 129028ms | — | 1.8909 | — | — | 8.66GB | 19.92GB | 1.51GB | 27.98GB | This text is a collection of excerpts from F. Scott Fitzgera |
