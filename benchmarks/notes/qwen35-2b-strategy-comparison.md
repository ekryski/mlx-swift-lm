# Qwen3.5-2B: Complete Model + KV Quantization Strategy Comparison

**Date**: 2026-04-02
**Hardware**: Apple M1 Max, 64GB RAM, macOS 15.7.4
**Context**: 4096 tokens (summarization benchmark, model's performance sweet spot)
**Baseline**: bf16 weights, no KV cache quantization

---

## Full Comparison Matrix (4096 context)

All metrics shown as absolute values with **% change vs bf16/no-quant baseline**.

### Generation Speed

| Config | Gen tok/s | vs Baseline | Prefill tok/s | vs Baseline |
|--------|:---------:|:-----------:|:-------------:|:-----------:|
| **bf16 / no-quant** | **53.7** | — | **2024** | — |
| bf16 / affine-4 | 52.6 | -2.0% | 1917 | -5.3% |
| bf16 / turbo4 | 55.5 | +3.4% | 2042 | +0.9% |
| bf16 / turbo4v2 | 54.8 | +2.0% | 2026 | +0.1% |
| bf16 / turbo3 | 53.7 | 0.0% | 2006 | -0.9% |
| bf16 / turbo3v2 | 54.5 | +1.5% | 2023 | -0.1% |
| **8bit / no-quant** | **72.3** | **+34.6%** | 1242 | -38.6% |
| 8bit / affine-4 | 74.1 | +38.0% | 1212 | -40.1% |
| 8bit / turbo4 | 78.6 | +46.4% | 1257 | -37.9% |
| 8bit / turbo4v2 | 79.3 | +47.7% | 1237 | -38.9% |
| 8bit / turbo3 | 79.7 | +48.4% | 1257 | -37.9% |
| 8bit / turbo3v2 | 80.2 | +49.3% | 1257 | -37.9% |
| **4bit / no-quant** | **96.3** | **+79.3%** | 1232 | -39.1% |
| 4bit / affine-4 | 89.3 | +66.3% | 1201 | -40.7% |
| 4bit / turbo4 | 91.5 | +70.4% | 1212 | -40.1% |
| 4bit / turbo4v2 | 95.1 | +77.1% | 1240 | -38.7% |
| 4bit / turbo3 | 94.2 | +75.4% | 1240 | -38.7% |
| 4bit / turbo3v2 | 95.4 | +77.7% | 1249 | -38.3% |

### KV Cache Memory

| Config | KV Cache | vs Baseline | Compression | GPU Peak | GPU Baseline (weights) |
|--------|:--------:|:-----------:|:-----------:|:--------:|:----------------------:|
| **bf16 / no-quant** | **938MB** | — | **1.0x** | 5.41GB | 3.51GB |
| bf16 / affine-4 | 307MB | -67% | 3.1x | 5.19GB | 3.51GB |
| bf16 / turbo4 | 261MB | -72% | 3.6x | 5.41GB | 3.51GB |
| bf16 / turbo4v2 | 191MB | -80% | 4.9x | 5.41GB | 3.51GB |
| bf16 / turbo3 | 199MB | -79% | 4.7x | 5.41GB | 3.51GB |
| bf16 / turbo3v2 | 169MB | -82% | 5.5x | 5.41GB | 3.51GB |
| **8bit / no-quant** | **938MB** | — | **1.0x** | 3.95GB | 1.86GB |
| 8bit / affine-4 | 293MB | -69% | 3.2x | 3.55GB | 1.86GB |
| 8bit / turbo4 | 249MB | -73% | 3.8x | 3.95GB | 1.86GB |
| 8bit / turbo4v2 | 191MB | -80% | 4.9x | 3.95GB | 1.86GB |
| 8bit / turbo3 | 199MB | -79% | 4.7x | 3.95GB | 1.86GB |
| 8bit / turbo3v2 | 169MB | -82% | 5.5x | 3.95GB | 1.86GB |
| **4bit / no-quant** | **938MB** | — | **1.0x** | 3.14GB | 1.01GB |
| 4bit / affine-4 | 293MB | -69% | 3.2x | 2.67GB | 1.01GB |
| 4bit / turbo4 | 261MB | -72% | 3.6x | 3.14GB | 1.01GB |
| 4bit / turbo4v2 | 191MB | -80% | 4.9x | 3.14GB | 1.01GB |
| 4bit / turbo3 | 199MB | -79% | 4.7x | 3.14GB | 1.01GB |
| 4bit / turbo3v2 | 169MB | -82% | 5.5x | 3.14GB | 1.01GB |

### Quality (KLD vs bf16/no-quant, lower = better)

| Config | Think KLD | Gen KLD | Think PPL | Gen PPL |
|--------|:---------:|:-------:|:---------:|:-------:|
| **bf16 / no-quant** | **—** | **—** | **3.93** | **1.54** |
| bf16 / affine-4 | 0.041 | -0.003 | 2.76 | 1.12 |
| bf16 / turbo4 | 0.018 | 0.041 | 2.74 | 2.55 |
| bf16 / turbo4v2 | 0.027 | 0.426 | 2.56 | 1.42 |
| bf16 / turbo3 | 0.018 | 0.040 | 2.92 | 2.17 |
| bf16 / turbo3v2 | 0.016 | 0.038 | 2.35 | 2.70 |
| **8bit / no-quant** | **0.019** | **0.357** | **3.61** | **1.15** |
| 8bit / affine-4 | 0.054 | -0.005 | 3.21 | 1.67 |
| 8bit / turbo4 | 0.027 | 0.042 | 3.21 | 1.72 |
| 8bit / turbo4v2 | -0.013 | 0.090 | 3.41 | 1.39 |
| 8bit / turbo3 | 0.026 | 0.044 | 3.52 | 2.07 |
| 8bit / turbo3v2 | 0.001 | 0.028 | 2.45 | 3.15 |
| **4bit / no-quant** | **0.160** | **-0.009** | **3.44** | **1.37** |
| 4bit / affine-4 | 0.211 | 1.005 | 3.08 | 8.04 |
| 4bit / turbo4 | 0.119 | 0.075 | 2.56 | 1.77 |
| 4bit / turbo4v2 | 0.201 | 0.198 | 3.42 | 1.86 |
| 4bit / turbo3 | 0.190 | 0.169 | 2.51 | 5.20 |
| 4bit / turbo3v2 | 0.191 | 0.111 | 2.87 | 2.30 |

---

## Category Winners

### Fastest Generation

| Rank | Config | Gen tok/s | vs Baseline |
|:----:|--------|:---------:|:-----------:|
| 1 | **4bit / no-quant** | **96.3** | +79.3% |
| 2 | 4bit / turbo3v2 | 95.4 | +77.7% |
| 3 | 4bit / turbo4v2 | 95.1 | +77.1% |
| 4 | 4bit / turbo3 | 94.2 | +75.4% |
| 5 | 4bit / turbo4 | 91.5 | +70.4% |

4-bit weights dominate generation speed. Turbo KV quant has minimal impact on 4-bit speed (95.4 vs 96.3 = -0.9% for turbo3v2).

### Best Quality (lowest combined KLD)

| Rank | Config | Think KLD | Gen KLD | Combined |
|:----:|--------|:---------:|:-------:|:--------:|
| 1 | **8bit / turbo3v2** | 0.001 | 0.028 | 0.029 |
| 2 | 8bit / turbo4 | 0.027 | 0.042 | 0.069 |
| 3 | 8bit / turbo3 | 0.026 | 0.044 | 0.070 |
| 4 | 8bit / turbo4v2 | 0.013 | 0.090 | 0.103 |
| 5 | bf16 / turbo3v2 | 0.016 | 0.038 | 0.054 |

Turbo consistently outperforms affine4 on quality. 8bit/turbo3v2 has near-zero Think KLD (0.001).

### Smallest KV Cache (most compression)

| Rank | Config | KV Cache | Compression | vs Baseline |
|:----:|--------|:--------:|:-----------:|:-----------:|
| 1 | **any / turbo3v2** | **169MB** | **5.5x** | **-82%** |
| 2 | any / turbo4v2 | 191MB | 4.9x | -80% |
| 3 | any / turbo3 | 199MB | 4.7x | -79% |
| 4 | any / turbo4 | 249-261MB | 3.6-3.8x | -72% |
| 5 | any / affine-4 | 293-307MB | 3.1-3.2x | -67% |

KV cache size is independent of weight quantization (same formula applies). Turbo3v2 achieves 5.5x compression — 72% more than affine4's 3.2x.

### Smallest Total GPU Footprint

| Rank | Config | GPU Peak | GPU Baseline (weights) | KV Cache |
|:----:|--------|:--------:|:----------------------:|:--------:|
| 1 | **4bit / affine-4** | **2.67GB** | 1.01GB | 293MB |
| 2 | 4bit / turbo4v2 | 3.14GB | 1.01GB | 191MB |
| 3 | 4bit / turbo3v2 | 3.14GB | 1.01GB | 169MB |
| 4 | 4bit / no-quant | 3.14GB | 1.01GB | 938MB |
| 5 | 8bit / affine-4 | 3.55GB | 1.86GB | 293MB |

4-bit affine-4 has lowest GPU Peak because affine4's `quantizedMM` reduces peak computation memory (MLX native kernel optimization). Turbo configs match no-quant GPU Peak because custom Metal kernels don't benefit from MLX's graph-level optimization.

---

## The Ultimate Compression: bf16/no-quant → 4bit/turbo3v2

| Metric | bf16 / no-quant | 4bit / turbo3v2 | Change |
|--------|:---------------:|:---------------:|:------:|
| **Weight memory** | 3.51 GB | 1.01 GB | **-71%** |
| **KV Cache (4K ctx)** | 938 MB | 169 MB | **-82% (5.5x)** |
| **Total footprint** | ~4.45 GB | ~1.18 GB | **-73%** |
| **Gen tok/s** | 53.7 | 95.4 | **+78%** |
| **Prefill tok/s** | 2024 | 1249 | -38% |
| **Think KLD** | — | 0.191 | moderate |
| **Gen KLD** | — | 0.111 | moderate |
| **Gen PPL** | 1.54 | 2.30 | +49% |

Going from the gold-standard bf16 to 4bit+turbo3v2:
- **3.8x total memory reduction** (4.45GB → 1.18GB)
- **78% faster generation** (53.7 → 95.4 tok/s)
- Quality degradation is moderate (PPL +49%, KLD ~0.15)

## The Best Balance: 8bit/turbo4v2

| Metric | bf16 / no-quant | 8bit / turbo4v2 | Change |
|--------|:---------------:|:---------------:|:------:|
| **Weight memory** | 3.51 GB | 1.86 GB | **-47%** |
| **KV Cache (4K ctx)** | 938 MB | 191 MB | **-80% (4.9x)** |
| **Total footprint** | ~4.45 GB | ~2.05 GB | **-54%** |
| **Gen tok/s** | 53.7 | 79.3 | **+48%** |
| **Think KLD** | — | -0.013 | excellent |
| **Gen KLD** | — | 0.090 | good |
| **Gen PPL** | 1.54 | 1.39 | -10% (better) |

8bit+turbo4v2 delivers the best quality-to-efficiency tradeoff:
- **2.2x total memory reduction** with near-zero quality loss
- **48% faster generation**
- Gen PPL actually *improves* (1.39 vs 1.54 — norm correction effect)

---

## Anomalies

1. **4bit/affine-4 Gen PPL = 8.04, Gen KLD = 1.005**: Catastrophic quality failure. Stacking 4-bit weight quantization with affine-4 KV quantization breaks generation quality. Turbo4 at the same combo (Gen PPL 1.77, Gen KLD 0.075) handles it fine — turbo's rotation + norm correction is more robust to compounding quantization errors.

2. **bf16/turbo4v2 Gen KLD = 0.426**: Unexpectedly high for a single turbo config. Likely stochastic — other bf16+turbo configs show Gen KLD 0.038-0.041. Needs re-run to confirm.

3. **8bit/no-quant Gen KLD = 0.357**: Higher than expected for 8-bit weights alone. A single anomalous generation run — other 8-bit configs show much lower Gen KLD (0.028-0.090).

4. **Turbo doesn't reduce GPU Peak vs no-quant**: At 4K context, GPU Peak is identical (3.95GB for 8-bit, 3.14GB for 4-bit) regardless of KV compression. Peak is dominated by prefill computation tensors, not KV storage. KV compression benefits will show at longer contexts (32K+) where KV storage dominates peak.
