# Inference Benchmark - Gemma 4 E2B

- **Date**: 2026-04-06 23:26
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 8bit
- **Model**: `mlx-community/gemma-4-e2b-it-8bit`

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
| summarization | 128 | 116 | no-quant | 425.3 | 55.6 | 400 | 275ms | 1.2298 | — | 1.8167 | — | 4.61GB | 5.04GB | 0MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 719.1 | 54.4 | 400 | 356ms | 1.3923 | — | 0.6417 | — | 4.61GB | 5.42GB | 15MB | 143MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 512 | 502 | no-quant | 900.6 | 53.3 | 400 | 559ms | 1.3357 | — | 0.2863 | — | 4.61GB | 6.13GB | 17MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 1049.6 | 52.7 | 400 | 967ms | 1.3947 | — | 0.2481 | — | 4.61GB | 6.69GB | 24MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 1000.8 | 52.0 | 400 | 2079ms | 1.4608 | — | 0.1735 | — | 4.61GB | 7.77GB | 34MB | 533MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 4096 | 4094 | no-quant | 1536.5 | 51.2 | 400 | 2712ms | 1.4380 | — | 0.2693 | — | 4.61GB | 7.69GB | 59MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8229 | no-quant | 3471.5 | 41.4 | 394 | 2386ms | — | 1.7657 | 0.3137 | — | 4.61GB | 6.66GB | 105MB | 1.84GB | This excerpt is a collection of several significant passages |
| summarization | 16384 | 16395 | no-quant | 2834.7 | 44.9 | 361 | 5844ms | — | 1.7874 | 0.2960 | — | 4.61GB | 7.30GB | 204MB | 3.58GB | This excerpt from *The Great Gatsby* introduces the narrator |
| summarization | 32768 | 32815 | no-quant | 3033.5 | 38.2 | 400 | 10875ms | — | 1.7488 | 0.3210 | — | 4.61GB | 8.50GB | 328MB | 7.10GB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 65536 | 65896 | no-quant | 1866.8 | 29.5 | 400 | 35446ms | — | 1.7198 | 0.3170 | — | 4.61GB | 10.90GB | 520MB | 14.16GB | This text is an excerpt from **The Great Gatsby** by F. Scot |
| summarization | 131072 | 130563 | no-quant | 1047.0 | 28.0 | 400 | 124843ms | — | 1.8115 | 0.0340 | — | 4.61GB | 15.86GB | 1.01GB | 27.98GB | This text is a compilation of excerpts from several major wo |
