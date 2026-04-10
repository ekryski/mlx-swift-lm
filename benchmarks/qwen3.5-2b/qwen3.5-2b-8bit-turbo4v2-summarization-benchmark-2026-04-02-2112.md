# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 21:12
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
| summarization | 128 | 119 | turbo4v2 | 574.9 | 79.0 | 201 | 208ms | 3.3778 | 1.5656 | 0.0447 | 0.0014 | 1.86GB | 2.08GB | 8MB | 14MB | The text provided is the **Table of Contents** for F. Scott  |
| summarization | 256 | 251 | turbo4v2 | 776.8 | 81.7 | 400 | 324ms | 3.3423 | 2.8273 | 0.0032 | 0.0153 | 1.86GB | 2.31GB | 6MB | 29MB | The text you provided opens the classic essay **"Once Again  |
| summarization | 512 | 506 | turbo4v2 | 981.4 | 79.8 | 337 | 516ms | 3.2581 | 2.6038 | 0.0271 | 0.0152 | 1.86GB | 2.71GB | 22MB | 37MB | In this excerpt from *The Great Gatsby*, F. Scott Fitzgerald |
| summarization | 1024 | 1021 | turbo4v2 | 1052.1 | 80.7 | 201 | 971ms | 2.6774 | 1.3812 | 0.0293 | 0.0284 | 1.86GB | 3.27GB | 25MB | 54MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 2048 | 2044 | turbo4v2 | 1082.3 | 80.3 | 400 | 1927ms | 2.9748 | 1.0257 | 0.0231 | -0.0035 | 1.86GB | 3.97GB | 39MB | 109MB | Here is a summary of the provided text, which consists of Ch |
| summarization | 4096 | 4087 | turbo4v2 | 1254.6 | 79.6 | 201 | 3305ms | 3.4252 | 1.3840 | 0.0748 | 0.0624 | 1.86GB | 3.95GB | 59MB | 191MB | Here is a summary of the provided text, which is part of F.  |
| summarization | 8192 | 8192 | turbo4v2 | 1335.9 | 76.0 | 201 | 6183ms | 3.6013 | 1.2861 | 0.0134 | 0.0513 | 1.86GB | 4.04GB | 109MB | 373MB | Here is a summary of the provided text, which contains the f |
| summarization | 16384 | 16363 | turbo4v2 | 1364.5 | 73.0 | 400 | 12106ms | 3.8767 | 3.2387 | 0.0603 | 0.0716 | 1.86GB | 4.39GB | 207MB | 745MB | Here is a summary of the provided text from F. Scott Fitzger |
| summarization | 32768 | 32702 | turbo4v2 | 1303.7 | 64.9 | 400 | 25405ms | 4.4135 | 3.3217 | 0.0326 | 0.0271 | 1.86GB | 5.00GB | 399MB | 1.44GB | Here is a summary of the contents provided (parts I through  |
| summarization | 65536 | 65470 | turbo4v2 | 981.6 | 48.6 | 201 | 72852ms | 3.3024 | 2.0392 | 0.0159 | 0.1983 | 1.86GB | 6.20GB | 715MB | 2.85GB | Here is a summary of the provided text, **F. Scott Fitzgeral |
| summarization | 131072 | 130775 | turbo4v2 | 781.9 | 34.7 | 400 | 169297ms | 2.4144 | 1.1826 | 0.0296 | 0.0118 | 1.86GB | 8.63GB | 1.51GB | 5.69GB | This is a detailed summary of the provided texts, *The Great |
