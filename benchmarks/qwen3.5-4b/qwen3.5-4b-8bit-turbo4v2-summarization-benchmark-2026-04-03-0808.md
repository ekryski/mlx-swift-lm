# Inference Benchmark - Qwen3.5 4B

- **Date**: 2026-04-03 08:08
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-4B-8bit`

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
| summarization | 128 | 117 | turbo4v2 | 333.2 | 47.5 | 291 | 353ms | 1.2827 | 1.8540 | 0.0335 | 0.0596 | 4.16GB | 4.48GB | 38MB | 18MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4v2 | 420.1 | 47.2 | 400 | 593ms | 1.3764 | 1.9107 | 0.0129 | -0.0015 | 4.16GB | 4.76GB | 43MB | 29MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4v2 | 459.0 | 47.1 | 400 | 1100ms | 1.6403 | 2.2148 | 0.0315 | 0.0182 | 4.16GB | 5.11GB | 54MB | 40MB | The user wants a summary of the provided text, which is the  |
| summarization | 1024 | 1019 | turbo4v2 | 483.1 | 47.0 | 400 | 2204ms | 1.4312 | 1.9694 | 0.0371 | 0.0049 | 4.16GB | 5.76GB | 70MB | 63MB | The user wants a summary of the provided text from *The Grea |
| summarization | 2048 | 2042 | turbo4v2 | 490.6 | 46.6 | 400 | 4270ms | 2.1839 | 2.0114 | 0.0145 | 0.0226 | 4.16GB | 6.52GB | 102MB | 109MB | The user wants a summary of the provided text, which is an e |
| summarization | 4096 | 4085 | turbo4v2 | 532.7 | 45.8 | 400 | 7785ms | 1.8427 | 2.3178 | 0.0484 | 0.0347 | 4.16GB | 6.72GB | 157MB | 199MB | The user wants a summary of the provided text, which is Chap |
| summarization | 8192 | 8190 | turbo4v2 | 546.9 | 44.9 | 400 | 15128ms | 2.0139 | 2.3959 | 0.0087 | 0.0343 | 4.16GB | 7.05GB | 295MB | 382MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | turbo4v2 | 539.3 | 41.9 | 400 | 30576ms | 2.2232 | 2.5879 | 0.0755 | 0.0473 | 4.16GB | 7.78GB | 551MB | 745MB | The user wants a summary of the provided text, which is an e |
| summarization | 32768 | 32700 | turbo4v2 | 505.8 | 37.4 | 400 | 65013ms | 1.6244 | 2.4145 | 0.0144 | 0.0371 | 4.16GB | 9.19GB | 997MB | 1.44GB | The user wants a summary of the provided text, which is Chap |
| summarization | 65536 | 65468 | turbo4v2 | 404.7 | 30.9 | 400 | 162153ms | 2.7820 | 2.9916 | 0.0503 | 0.0758 | 4.16GB | 12.17GB | 2.04GB | 2.86GB | The user wants a summary of the provided text. The text is t |
| summarization | 131072 | 130773 | turbo4v2 | 302.1 | 23.2 | 400 | 433335ms | 1.7860 | 1.8041 | -0.0054 | 0.0421 | 4.16GB | 17.94GB | 4.03GB | 5.69GB | The user wants a summary of the text provided. The text prov |
