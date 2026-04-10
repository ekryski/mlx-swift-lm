# Inference Benchmark - Qwen3.5 4B

- **Date**: 2026-04-03 05:55
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
| summarization | 128 | 117 | turbo4v2 | 388.0 | 29.4 | 400 | 303ms | 1.6601 | 2.0326 | 0.0414 | 0.0094 | 7.83GB | 8.07GB | 46MB | 23MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 256 | 249 | turbo4v2 | 565.2 | 29.4 | 304 | 441ms | 1.5776 | 1.9339 | 0.0319 | 0.0179 | 7.83GB | 8.21GB | 45MB | 25MB | The user wants me to summarize the content provided. The tex |
| summarization | 512 | 504 | turbo4v2 | 695.4 | 29.3 | 201 | 725ms | 1.5922 | 10.6303 | 0.0380 | 0.0007 | 7.83GB | 8.55GB | 46MB | 31MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 1024 | 1019 | turbo4v2 | 761.9 | 29.4 | 400 | 1356ms | 1.4333 | 2.0863 | 0.0164 | 0.0214 | 7.83GB | 9.17GB | 67MB | 63MB | Thinking Process:  1.  **Analyze the Request:**     *   Inpu |
| summarization | 2048 | 2042 | turbo4v2 | 785.3 | 29.2 | 400 | 2816ms | 2.0470 | 2.0216 | 0.0208 | 0.0307 | 7.83GB | 10.07GB | 103MB | 109MB | The user wants a summary of the provided text from *The Grea |
| summarization | 4096 | 4085 | turbo4v2 | 846.1 | 29.1 | 400 | 4986ms | 2.0498 | 2.6947 | 0.0245 | 0.0542 | 7.83GB | 10.27GB | 170MB | 199MB | The user wants a summary of the provided text, which is Chap |
| summarization | 8192 | 8190 | turbo4v2 | 859.3 | 28.5 | 400 | 9722ms | 1.9229 | 2.7065 | 0.0306 | 0.0291 | 7.83GB | 10.67GB | 295MB | 382MB | The user wants a summary of the provided text, which is the  |
| summarization | 16384 | 16361 | turbo4v2 | 822.5 | 27.4 | 400 | 20186ms | 2.0662 | 2.2256 | 0.0208 | 0.0432 | 7.83GB | 11.37GB | 552MB | 745MB | The user wants a summary of the provided text, which is the  |
| summarization | 32768 | 32700 | turbo4v2 | 744.1 | 25.4 | 400 | 44305ms | 2.0363 | 2.1343 | 0.0360 | 0.0419 | 7.83GB | 12.77GB | 1.04GB | 1.44GB | The user wants a summary of the provided text, which is the  |
| summarization | 65536 | 65468 | turbo4v2 | 536.2 | 21.9 | 400 | 123234ms | 1.9756 | 2.1954 | 0.0223 | 0.0377 | 7.83GB | 15.70GB | 2.04GB | 2.86GB | The user wants a summary of the provided text, which is the  |
| summarization | 131072 | 130773 | turbo4v2 | 368.6 | 18.5 | 400 | 355171ms | 1.8002 | 1.7592 | 0.0290 | 0.0236 | 7.83GB | 21.44GB | 4.03GB | 5.69GB | The user wants a summary of the provided text. The text cons |
