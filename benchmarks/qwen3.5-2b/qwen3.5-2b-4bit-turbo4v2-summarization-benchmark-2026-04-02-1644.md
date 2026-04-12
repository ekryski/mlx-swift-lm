# Inference Benchmark - Qwen3.5 2B

- **Date**: 2026-04-02 16:44
- **Branch**: `ek/turbo-opt-0-fix-default-path`
- **Quantization**: 4bit
- **Model**: `mlx-community/Qwen3.5-2B-4bit`

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
| summarization | 128 | 119 | turbo4v2 | 624.2 | 96.9 | 400 | 192ms | 2.4911 | 1.9397 | 0.1694 | 0.0694 | 1010MB | 1.21GB | 15MB | 23MB | The text you provided is a stylized, fragmented excerpt comb |
| summarization | 256 | 251 | turbo4v2 | 852.7 | 89.1 | 388 | 295ms | 4.4166 | 3.1010 | 0.1857 | 0.1384 | 1010MB | 1.43GB | 18MB | 28MB | The excerpt from F. Scott Fitzgerald's *The Great Gatsby* co |
| summarization | 512 | 506 | turbo4v2 | 980.7 | 95.2 | 302 | 516ms | 2.4502 | 3.9776 | 0.1456 | 0.2821 | 1010MB | 1.88GB | 11MB | 36MB | Here is a summary of the text provided by F. Scott Fitzgeral |
| summarization | 1024 | 1021 | turbo4v2 | 1032.5 | 94.9 | 400 | 989ms | 2.8530 | 3.1882 | 0.1419 | 0.2024 | 1010MB | 2.48GB | 28MB | 63MB | Here is a summary of the provided text, which consists of ex |
| summarization | 2048 | 2044 | turbo4v2 | 1074.1 | 94.7 | 400 | 1917ms | 3.2477 | 2.8981 | 0.1647 | 0.0873 | 1010MB | 3.18GB | 38MB | 109MB | This text is the opening of **"The Great Gatsby"** by F. Sco |
| summarization | 4096 | 4087 | turbo4v2 | 1240.1 | 95.1 | 201 | 3339ms | 3.4226 | 1.8575 | 0.2011 | 0.1980 | 1010MB | 3.14GB | 50MB | 191MB | Here is a summary of the provided text:  **Introduction and  |
| summarization | 8192 | 8192 | turbo4v2 | 1341.2 | 94.8 | 400 | 6150ms | 3.3542 | 3.5993 | 0.2084 | 0.1075 | 1010MB | 3.34GB | 92MB | 382MB | This text presents a condensed, narrative summary of F. Scot |
| summarization | 16384 | 16363 | turbo4v2 | 1347.6 | 85.6 | 400 | 12280ms | 3.0186 | 4.6929 | 0.1423 | 0.1979 | 1010MB | 3.62GB | 208MB | 745MB | Here is a summary of the provided text from *The Great Gatsb |
| summarization | 32768 | 32702 | turbo4v2 | 1262.3 | 75.8 | 400 | 26275ms | 3.3399 | 3.6956 | 0.1809 | 0.2203 | 1010MB | 4.18GB | 399MB | 1.44GB | This story, **"The Great Gatsby" by F. Scott Fitzgerald**, c |
| summarization | 65536 | 65470 | turbo4v2 | 1010.6 | 47.1 | 400 | 69970ms | 3.8500 | 1.0363 | 0.1986 | 0.0027 | 1010MB | 5.31GB | 782MB | 2.86GB | Here is a summary of the novel *The Great Gatsby* by F. Scot |
| summarization | 131072 | 130775 | turbo4v2 | 787.9 | 35.0 | 400 | 168718ms | 3.4711 | 1.3302 | 0.1714 | 0.0659 | 1011MB | 7.82GB | 1.51GB | 5.69GB | The provided text is a collection of short stories or short  |
