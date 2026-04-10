# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 20:10
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
| summarization | 128 | 119 | turbo3v2-p2 | 604.0 | 77.8 | 322 | 198ms | 4.3112 | 4.0847 | 0.0580 | 0.0393 | 1.86GB | 2.08GB | 16MB | 17MB | You've shared a famous excerpt from F. Scott Fitzgerald's no |
| summarization | 256 | 251 | turbo3v2-p2 | 842.5 | 76.3 | 201 | 298ms | 2.6566 | 1.1223 | 0.0219 | 0.2298 | 1.86GB | 2.31GB | 9MB | 17MB | The provided text contains the opening chapters of *The Grea |
| summarization | 512 | 506 | turbo3v2-p2 | 945.3 | 76.8 | 337 | 536ms | 2.6804 | 3.9112 | 0.0285 | 0.0238 | 1.86GB | 2.71GB | 22MB | 32MB | This excerpt from F. Scott Fitzgerald's *The Great Gatsby*,  |
| summarization | 1024 | 1021 | turbo3v2-p2 | 1012.7 | 77.9 | 201 | 1009ms | 2.4564 | 1.4565 | 0.0222 | 0.0489 | 1.86GB | 3.27GB | 25MB | 46MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 2048 | 2044 | turbo3v2-p2 | 1054.2 | 77.6 | 400 | 1971ms | 2.5026 | 2.9683 | 0.0427 | 0.0684 | 1.86GB | 3.97GB | 40MB | 92MB | Based on the provided text from *The Great Gatsby*, here is  |
| summarization | 4096 | 4087 | turbo3v2-p2 | 1220.3 | 74.2 | 400 | 3411ms | 3.1807 | 3.6739 | 0.0248 | 0.0481 | 1.86GB | 3.95GB | 64MB | 169MB | This excerpt is from the beginning of F. Scott Fitzgerald's  |
| summarization | 8192 | 8192 | turbo3v2-p2 | 1299.6 | 72.7 | 383 | 6354ms | 2.1934 | 1.0205 | 0.0215 | 0.0008 | 1.86GB | 4.04GB | 111MB | 322MB | Here is a summary of *The Great Gatsby* as presented in the  |
| summarization | 16384 | 16363 | turbo3v2-p2 | 1322.1 | 69.3 | 400 | 12515ms | 2.9042 | 1.0095 | -0.0023 | -0.0004 | 1.86GB | 4.39GB | 208MB | 630MB | Here is a summary of the provided text, which consists of th |
| summarization | 32768 | 32702 | turbo3v2-p2 | 1250.7 | 62.0 | 400 | 26513ms | 2.8702 | 1.0888 | 0.0418 | 0.0022 | 1.86GB | 5.00GB | 400MB | 1.22GB | Here is a summary of the provided text, *The Great Gatsby*,  |
| summarization | 65536 | 65470 | turbo3v2-p2 | 954.9 | 42.1 | 400 | 75721ms | 3.2097 | 2.2813 | 0.0167 | 0.0260 | 1.86GB | 6.34GB | 653MB | 2.42GB | Here is a summary of the novel *The Great Gatsby* by F. Scot |
| summarization | 131072 | 130775 | turbo3v2-p2 | 755.1 | 31.3 | 400 | 175971ms | 2.9283 | 2.5790 | 0.0365 | -0.0003 | 1.86GB | 8.63GB | 1.39GB | 4.82GB | This collection of short stories is a dramatic, psychologica |
