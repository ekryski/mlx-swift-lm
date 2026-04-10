# Inference Benchmark - Gemma 4 31B

- **Date**: 2026-04-07 08:27
- **Branch**: `ek/tom-eric-moe-tuning`
- **Quantization**: 8bit
- **Model**: `mlx-community/gemma-4-31b-it-8bit`

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
| summarization | 128 | 116 | no-quant | 32.1 | 8.7 | 293 | 4521ms | 1.2374 | 1.0829 | — | — | 30.38GB | 31.29GB | 491MB | 89MB | <\|channel>thought  *   Input text: A snippet containing the  |
| summarization | 256 | 255 | no-quant | 68.3 | 8.6 | 313 | 4419ms | 1.2027 | 1.0662 | — | — | 30.38GB | 31.56GB | 552MB | 124MB | <\|channel>thought  *   Text provided: The beginning of *The  |
| summarization | 512 | 502 | no-quant | 69.1 | 8.6 | 400 | 7934ms | — | 1.0209 | — | — | 30.38GB | 32.09GB | 785MB | 197MB | // a single sentence summary // a single own-// // a own own |
| summarization | 1024 | 1014 | no-quant | 69.9 | 8.4 | 400 | 15263ms | — | 1.0086 | — | — | 30.38GB | 33.14GB | 816MB | 309MB | // a single single-quote single-quote own own own own own ow |
| summarization | 2048 | 2037 | no-quant | 69.0 | 8.8 | 400 | 30623ms | — | 1.0373 | — | — | 30.38GB | 35.69GB | 928MB | 533MB | // a la single single single single single single single sin |
| summarization | 4096 | 4094 | no-quant | 70.6 | 8.3 | 400 | 58848ms | — | 1.0179 | — | — | 30.38GB | 36.91GB | 1.03GB | 983MB | // a same same single single same same single same single ow |
| summarization | 8192 | 8229 | no-quant | 71.6 | 8.4 | 400 | 115141ms | 1.0378 | 1.0620 | — | — | 30.38GB | 35.29GB | 1.25GB | 1.84GB | // a same same single single same same single same single sa |
| summarization | 16384 | 16395 | no-quant | 69.8 | 7.8 | 400 | 235002ms | — | 1.0146 | — | — | 30.38GB | 36.92GB | 2.03GB | 3.59GB | // a same single same same same same same same same same sam |
| summarization | 32768 | 32815 | no-quant | 66.5 | 7.0 | 400 | 493741ms | — | 1.0040 | — | — | 30.38GB | 40.17GB | 2.89GB | 7.10GB | // a same single same same same same same same same same sam |
| summarization | 65536 | 65896 | no-quant | 60.0 | 5.2 | 400 | 1098305ms | — | 1.0199 | — | — | 30.38GB | 46.81GB | 5.16GB | 14.16GB | // a same single same same single same single single same si |
| summarization | 131072 | 130563 | no-quant | 30.4 | 3.7 | 400 | 4316263ms | — | 1.0017 | — | — | 30.38GB | 60.01GB | 10.73GB | 27.98GB | // same same same same same same same same same same same sa |
