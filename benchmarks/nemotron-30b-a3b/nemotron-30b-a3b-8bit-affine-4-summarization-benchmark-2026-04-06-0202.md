# Inference Benchmark - Nemotron 30B A3B

- **Date**: 2026-04-06 02:02
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 8bit
- **Model**: `mlx-community/Nemotron-Cascade-2-30B-A3B-8bit`

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
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 400 |
| Thinking Budget | 200 |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 128 | 135 | affine-4 | 63.9 | 26.0 | 230 | 2377ms | 1.5816 | 1.3079 | 0.0000 | 0.0000 | 31.26GB | 32.44GB | 47MB | 25MB | We need to summarize the content above. The content is a sni |
| summarization | 1024 | 1049 | affine-4 | 19.8 | 25.6 | 400 | 53657ms | 1.7450 | 1.7906 | 0.0760 | 0.0058 | 31.26GB | 39.82GB | 47MB | 99MB | We need to summarize given content. The content includes typ |
| summarization | 4096 | 4110 | affine-4 | 46.7 | 25.4 | 400 | 88811ms | 1.7042 | 2.3383 | 0.0153 | 0.0264 | 31.26GB | 58.84GB | 53MB | 308MB | We need to provide a summary of the content above. The conte |
| summarization | 32768 | 32553 | affine-4 | 112.5 | 23.3 | 400 | 290201ms | 1.7006 | 1.5158 | -0.0034 | 0.0131 | 31.26GB | 59.14GB | 108MB | 2.20GB | We need to provide a summary of the given text. The text is  |
