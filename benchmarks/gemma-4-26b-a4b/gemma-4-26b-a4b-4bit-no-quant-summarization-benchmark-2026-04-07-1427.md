# Inference Benchmark - Gemma 4 26B A4B

- **Date**: 2026-04-07 14:27
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 4bit
- **Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`

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
| summarization | 128 | 116 | no-quant | 222.7 | 27.4 | 400 | 523ms | 1.3057 | — | 0.6278 | — | 13.48GB | 14.13GB | 206MB | 113MB | <\|channel>thought  *   Title: *The Great Gatsby* *   Author: |
| summarization | 256 | 255 | no-quant | 48.4 | 27.1 | 400 | 5555ms | 1.3823 | — | 0.7889 | — | 13.48GB | 14.47GB | 262MB | 143MB | <\|channel>thought  *   Text provided: An excerpt from *The G |
| summarization | 512 | 502 | no-quant | 109.0 | 26.4 | 400 | 4919ms | 1.3423 | — | 0.5664 | — | 13.48GB | 14.96GB | 333MB | 197MB | <\|channel>thought  *   Text provided: The opening pages of F |
| summarization | 1024 | 1014 | no-quant | 234.9 | 26.3 | 400 | 4620ms | 1.3971 | — | 0.7501 | — | 13.48GB | 15.95GB | 392MB | 309MB | <\|channel>thought  *   Text provided: The opening pages (Tit |
| summarization | 2048 | 2037 | no-quant | 427.4 | 25.9 | 400 | 5057ms | 1.4350 | — | 0.7407 | — | 13.48GB | 17.64GB | 408MB | 533MB | <\|channel>thought  *   Source Material: The opening pages of |
| summarization | 4096 | 4094 | no-quant | 519.3 | 25.0 | 400 | 8184ms | 1.4333 | — | 0.6837 | — | 13.48GB | 18.43GB | 432MB | 983MB | <\|channel>thought  *   Source text: An excerpt from the begi |
| summarization | 8192 | 8229 | no-quant | 609.8 | 24.2 | 400 | 13566ms | 1.4535 | — | 0.6054 | — | 13.48GB | 17.49GB | 664MB | 1.84GB | <\|channel>thought - **Introductory Context**: The text provi |
| summarization | 16384 | 16395 | no-quant | 577.3 | 22.1 | 400 | 28542ms | 1.0065 | 1.3969 | 9.4553 | 0.4848 | 13.48GB | 18.50GB | 976MB | 3.59GB | --- <\|channel>-thought <channel\|>The provided text consists  |
| summarization | 32768 | 32815 | no-quant | 514.7 | 17.1 | 400 | 64117ms | 1.7089 | 1.2087 | 0.4864 | 0.4156 | 13.48GB | 21.16GB | 1.59GB | 7.10GB | --- <\|channel>thought  there's a lot of content here. I'll s |
| summarization | 65536 | 65896 | no-quant | 371.9 | 12.5 | 400 | 177486ms | 1.2745 | 1.2487 | 0.5178 | 0.2592 | 13.48GB | 26.72GB | 2.85GB | 14.16GB | ---  <\|channel>synthesis The provided text contains the comp |
| summarization | 131072 | 130563 | no-quant | 221.3 | 8.1 | 400 | 590355ms | — | 1.1802 | 0.4300 | — | 13.48GB | 37.50GB | 5.37GB | 27.98GB | s uniquelyinglyte ownlyte of the provided text is a compilat |
