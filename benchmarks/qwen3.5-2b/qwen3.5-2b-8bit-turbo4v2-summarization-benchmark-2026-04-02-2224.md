# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 22:24
- **Branch**: `ek/turbo-opt-0-fix-default-path`
- **Quantization**: 8bit
- **Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | turbo4v2 | 611.5 | 79.0 | 347 | 196ms | 3.3201 | 2.2741 | 0.0489 | 0.0258 | 1.86GB | 2.08GB | 16MB | 21MB | The text provided contains two distinct pieces of writing:   |
| summarization | 256 | 251 | turbo4v2 | 843.4 | 79.6 | 400 | 298ms | 2.4783 | 3.3676 | 0.0365 | 0.0315 | 1.86GB | 2.31GB | 18MB | 29MB | The text provided is the **prologue** to *The Great Gatsby*, |
| summarization | 512 | 506 | turbo4v2 | 954.9 | 80.0 | 201 | 530ms | 2.8964 | 1.1603 | 0.0042 | 0.1135 | 1.86GB | 2.71GB | 19MB | 31MB | This excerpt from *The Great Gatsby* is the "prologue," a fi |
| summarization | 1024 | 1021 | turbo4v2 | 1026.8 | 78.6 | 201 | 995ms | 2.5795 | 1.2105 | 0.0321 | 0.0425 | 1.86GB | 3.27GB | 14MB | 54MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 2048 | 2044 | turbo4v2 | 1044.6 | 78.3 | 201 | 1980ms | 2.7987 | 1.6568 | 0.0013 | 0.1051 | 1.86GB | 3.97GB | 25MB | 100MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 4096 | 4087 | turbo4v2 | 1225.4 | 76.8 | 201 | 3392ms | 2.7549 | 1.6217 | 0.0284 | 0.0383 | 1.86GB | 3.95GB | 61MB | 191MB | Here is a summary of the provided excerpt from *The Great Ga |
| summarization | 8192 | 8192 | turbo4v2 | 1319.7 | 75.0 | 231 | 6258ms | 2.6728 | 2.7053 | 0.0492 | -0.0004 | 1.86GB | 4.04GB | 108MB | 374MB | Here is a summary of the provided text, which serves as the  |
| summarization | 16384 | 16363 | turbo4v2 | 1324.0 | 70.4 | 201 | 12488ms | 2.7081 | 2.2789 | 0.0667 | -0.0442 | 1.86GB | 4.39GB | 204MB | 736MB | Here is a summary of the text provided, *The Great Gatsby* b |
| summarization | 32768 | 32702 | turbo4v2 | 1232.5 | 63.5 | 400 | 26890ms | 2.9827 | 4.0510 | 0.0669 | -0.0105 | 1.86GB | 5.00GB | 399MB | 1.44GB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* o |
| summarization | 65536 | 65470 | turbo4v2 | 951.2 | 41.3 | 400 | 72915ms | 3.9388 | 2.8475 | 0.0292 | 0.0602 | 1.86GB | 6.34GB | 783MB | 2.86GB | This text is the complete story of **The Great Gatsby** by F |
