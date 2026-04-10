# Inference Benchmark - Gemma 4 E4B

- **Date**: 2026-04-07 01:55
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 8bit
- **Model**: `mlx-community/gemma-4-e4b-it-8bit`

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
| summarization | 128 | 116 | no-quant | 280.2 | 33.8 | 400 | 418ms | 1.4142 | — | 0.6065 | — | 7.47GB | 7.97GB | 34MB | 113MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 256 | 255 | no-quant | 409.8 | 33.4 | 149 | 627ms | — | 2.6570 | 0.4723 | — | 7.47GB | 8.32GB | 42MB | 88MB | The provided content is a collection of literary excerpts, a |
| summarization | 512 | 502 | no-quant | 482.7 | 32.6 | 400 | 1043ms | 2.0381 | — | 0.2799 | — | 7.47GB | 8.99GB | 46MB | 197MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 1024 | 1014 | no-quant | 512.5 | 32.4 | 400 | 2093ms | 2.0170 | — | 0.2516 | — | 7.47GB | 9.73GB | 67MB | 309MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 2048 | 2037 | no-quant | 485.2 | 31.7 | 400 | 4336ms | 2.0143 | 1.5306 | 0.4269 | 0.7350 | 7.47GB | 10.79GB | 100MB | 533MB | <\|channel>thought Here's a plan to structure the summary: 1. |
| summarization | 4096 | 4094 | no-quant | 733.6 | 30.3 | 400 | 5750ms | 2.1040 | — | 0.8346 | — | 7.47GB | 10.85GB | 162MB | 983MB | <\|channel>thought Here's a thinking process to arrive at the |
| summarization | 8192 | 8229 | no-quant | 1076.6 | 27.6 | 34 | 7678ms | — | 2.9578 | 1.1074 | — | 7.47GB | 9.63GB | 295MB | 1.77GB | The provided text is a lengthy, impressionistic excerpt that |
| summarization | 16384 | 16395 | no-quant | 1058.8 | 24.8 | 373 | 15560ms | — | 1.4008 | 0.3740 | — | 7.47GB | 10.31GB | 551MB | 3.58GB | The provided text is a compilation of excerpts from the nove |
| summarization | 32768 | 32815 | no-quant | 975.4 | 18.5 | 119 | 33805ms | — | 1.9020 | 0.7480 | — | 7.47GB | 11.88GB | 1.04GB | 7.04GB | The provided texts are different forms of the novel *The Gre |
| summarization | 65536 | 65896 | no-quant | 777.2 | 13.2 | 81 | 85048ms | — | 2.1956 | 0.6499 | — | 7.47GB | 15.20GB | 778MB | 14.09GB | The text provided is a collection of different passages from |
| summarization | 131072 | 130563 | no-quant | 548.2 | 9.8 | 249 | 238403ms | — | 3.0304 | 0.0374 | — | 7.47GB | 21.60GB | 4.03GB | 27.94GB | *The provided texts are several drafts of an extensive narra |
