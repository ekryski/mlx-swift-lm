# Inference Benchmark - Qwen3.5 2B

**Date**: 2026-04-02 22:10
**Branch**: `ek/turbo-opt-0-fix-default-path`
**Quantization**: 8bit
**Model**: `mlx-community/Qwen3.5-2B-8bit`

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
| summarization | 128 | 119 | turbo3v2 | 596.4 | 81.6 | 400 | 201ms | 3.3453 | 2.9457 | 0.0099 | 0.0207 | 1.86GB | 2.08GB | 18MB | 20MB | The text you provided is **not** a summary of *The Great Gat |
| summarization | 256 | 251 | turbo3v2 | 826.1 | 79.4 | 201 | 305ms | 2.3117 | 1.4081 | 0.0379 | 0.0267 | 1.86GB | 2.31GB | 16MB | 17MB | This text presents the opening segment of **F. Scott Fitzger |
| summarization | 512 | 506 | turbo3v2 | 956.5 | 78.2 | 370 | 530ms | 3.3622 | 3.1386 | 0.0705 | 0.0686 | 1.86GB | 2.71GB | 22MB | 33MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby* o |
| summarization | 1024 | 1021 | turbo3v2 | 1048.5 | 82.7 | 400 | 974ms | 2.5871 | 3.2217 | 0.0231 | 0.0477 | 1.86GB | 3.27GB | 26MB | 53MB | This excerpt from *The Great Gatsby* by F. Scott Fitzgerald  |
| summarization | 2048 | 2044 | turbo3v2 | 1082.2 | 80.1 | 400 | 1909ms | 2.4470 | 3.3935 | 0.0243 | 0.0053 | 1.86GB | 3.97GB | 39MB | 92MB | This excerpt from **F. Scott Fitzgerald's** *The Great Gatsb |
| summarization | 4096 | 4087 | turbo3v2 | 1236.7 | 78.5 | 400 | 3364ms | 4.0238 | 3.7498 | 0.0586 | 0.0813 | 1.86GB | 3.95GB | 64MB | 169MB | Based on the text provided, here is a summary of *The Great  |
| summarization | 8192 | 8192 | turbo3v2 | 1352.0 | 78.0 | 400 | 6113ms | 2.8010 | 1.2531 | 0.0269 | 0.0006 | 1.86GB | 4.04GB | 93MB | 323MB | This text provides the opening chapters of F. Scott Fitzgera |
| summarization | 16384 | 16363 | turbo3v2 | 1362.5 | 73.3 | 201 | 12126ms | 3.3315 | 1.2125 | 0.0183 | 0.3942 | 1.86GB | 4.39GB | 204MB | 623MB | Here is a summary of the provided excerpts from F. Scott Fit |
| summarization | 32768 | 32702 | turbo3v2 | 1305.3 | 65.2 | 201 | 25391ms | 2.6047 | 1.2717 | 0.0414 | 0.0476 | 1.86GB | 5.00GB | 397MB | 1.21GB | This is a summary of the provided text, *The Great Gatsby* b |
| summarization | 65536 | 65470 | turbo3v2 | 1038.9 | 47.6 | 201 | 67839ms | 3.7107 | 1.2659 | 0.0222 | 0.3879 | 1.86GB | 6.34GB | 715MB | 2.41GB | Here is a summary of F. Scott Fitzgerald's *The Great Gatsby |
| summarization | 131072 | 130775 | turbo3v2 | 763.7 | 32.2 | 400 | 171791ms | 2.5182 | 1.0115 | 0.0189 | -0.0041 | 1.86GB | 8.63GB | 1.51GB | 4.82GB | The provided text contains excerpts from two major works of  |
