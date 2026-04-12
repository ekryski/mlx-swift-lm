# Inference Benchmark - Qwen3.5 4B

- **Date**: 2026-04-03 05:02
- **Branch**: `ek/consolidated-benchmarks`
- **Quantization**: bf16
- **Model**: `mlx-community/Qwen3.5-4B-bf16`

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
| summarization | 128 | 117 | turbo4 | 357.7 | 28.8 | 400 | 328ms | 1.8688 | 1.5986 | 0.0184 | 0.0278 | 7.83GB | 8.07GB | 48MB | 30MB | The user is asking me to summarize the content above. Let me |
| summarization | 256 | 249 | turbo4 | 552.1 | 28.9 | 400 | 451ms | 1.3577 | 1.8198 | 0.0227 | 0.0305 | 7.83GB | 8.21GB | 49MB | 38MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 512 | 504 | turbo4 | 666.6 | 28.6 | 306 | 756ms | 1.3568 | 1.8691 | 0.0117 | 0.0135 | 7.83GB | 8.55GB | 54MB | 47MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4 | 737.4 | 28.8 | 400 | 1421ms | 1.2599 | 2.0690 | -0.0160 | 0.0139 | 7.83GB | 9.17GB | 72MB | 82MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4 | 758.5 | 28.8 | 400 | 2851ms | 1.5280 | 1.8564 | 0.0225 | 0.0297 | 7.83GB | 10.07GB | 103MB | 142MB | Thinking Process:  1.  **Analyze the Request:**     *   **In |
| summarization | 4096 | 4085 | turbo4 | 820.9 | 28.5 | 400 | 5150ms | 2.2032 | 2.1616 | 0.0308 | 0.0404 | 7.83GB | 10.27GB | 169MB | 261MB | The user wants a summary of the provided text, which is Chap |
| summarization | 8192 | 8190 | turbo4 | 834.0 | 27.9 | 400 | 10035ms | 1.8410 | 2.1563 | 0.0313 | 0.0307 | 7.83GB | 10.67GB | 223MB | 499MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | turbo4 | 807.4 | 26.8 | 400 | 20600ms | 2.0473 | 2.4452 | 0.0258 | 0.0383 | 7.83GB | 11.37GB | 550MB | 974MB | The user wants a summary of the provided text, which is Chap |
| summarization | 32768 | 32700 | turbo4 | 721.3 | 24.7 | 400 | 45770ms | 2.1505 | 2.4085 | 0.0379 | 0.0454 | 7.83GB | 12.78GB | 1.04GB | 1.88GB | The user wants a summary of the provided text. The text is t |
| summarization | 65536 | 65468 | turbo4 | 429.8 | 20.8 | 400 | 156401ms | 1.9925 | 2.5035 | 0.0193 | 0.0692 | 7.83GB | 15.66GB | 2.04GB | 3.74GB | The user wants a summary of the provided text, which is a ve |
| summarization | 131072 | 130773 | turbo4 | 283.9 | 17.7 | 400 | 462802ms | 1.5758 | 1.9658 | 0.0352 | 0.0151 | 7.83GB | 21.44GB | 3.78GB | 7.44GB | Here's a thinking process that leads to the suggested summar |
