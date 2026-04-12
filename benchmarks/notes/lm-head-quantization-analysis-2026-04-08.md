# LM Head Quantization Analysis

Date: 2026-04-08

## Background

Phase 1d microbenchmarks showed massive speedups from quantizing LM head projections — up to 20.8x faster (Int4 vs FP16) for Gemma4's 262K vocab at decode. This prompted investigation into whether our benchmark models would benefit.

## Current State of LM Head Quantization

Checked actual safetensor files for our primary benchmark models:

| Model | Config | LM Head Status | Mechanism |
|-------|--------|---------------|-----------|
| Gemma4 E2B | 4bit | Already 4-bit quantized | Tied to `embed_tokens` (has scales/biases) |
| Gemma4 26B | 4bit | Already 4-bit quantized | Tied to `embed_tokens` (has scales/biases) |
| Qwen3.5 35B | 4bit | Already 4-bit quantized | Separate `lm_head` (has scales/biases) |
| Qwen3.5 35B | 8bit | 8-bit quantized | Separate `lm_head` (has scales/biases) |
| Qwen3.5 35B | bf16 | **FP16 (not quantized)** | Separate `lm_head` (weight only) |

**Key detail:** Gemma4 uses tied word embeddings (`tie_word_embeddings: true`), so `embed_tokens.weight` serves as both the input embedding lookup and the output projection. Quantizing it affects both paths.

## Where 2e Would Help

- **BF16 models:** LM head is FP16 → quantizing to Int4 gives ~20x decode speedup on that op (Gemma4 262K vocab: 30ms → 1.5ms per token)
- **8-bit models:** LM head is Int8 → downquantizing to Int4 gives ~1.5x speedup (Int4 0.25x vs Int8 0.38x of FP16 from microbenchmarks)
- **4-bit models:** Already quantized. No benefit.

## Decision

Deprioritized for now — our primary benchmark models (4-bit) already have quantized LM heads. The optimization would matter for bf16/8-bit deployment scenarios.

## Broader Hypothesis: Per-Layer Adaptive Quantization

The Gemma4 26B model config already demonstrates this approach — mlx-community's 4-bit quantization uses **per-layer mixed precision**:

```json
{
    "bits": 4,                    // default: 4-bit
    "language_model.model.layers.0.mlp.gate_proj": { "bits": 8 },  // MLP layers: 8-bit
    "language_model.model.layers.0.mlp.down_proj": { "bits": 8 },
    "language_model.model.layers.0.mlp.up_proj":   { "bits": 8 },
    "language_model.model.layers.0.router.proj":   { "bits": 8 },
    // ... repeated for all 30 layers
}
```

All MLP expert projections and routers are kept at 8-bit while attention projections are at 4-bit. This is exactly the kind of targeted quantization that could be optimized further.

### Opportunities for Targeted Quantization

The 1d microbenchmark data suggests different optimal bit widths depending on **layer type** and **inference regime** (prefill vs decode):

| Component | Decode (1 token) | Prefill (1K+ tokens) | Recommendation |
|-----------|------------------|---------------------|----------------|
| Expert projections | Int8 ≈ Int4 (both faster than FP16) | FP16 wins (compute-bound) | Int8 for quality, Int4 for size |
| Attention Q/K/V | Int4 fast enough | FP16 better at large T | Keep at model's default |
| LM head (large vocab) | Int4 massively faster (20x) | Int4 still 1.6x faster at 128T | Always quantize aggressively |
| Embeddings | Lookup only, no matmul | Lookup only | Precision matters for tied weights |
| Router projections | Tiny (hidden→numExperts) | Negligible cost | Keep at 8-bit for routing accuracy |

### Key Insight: Regime-Dependent Quantization

The crossover point (~256-512 tokens) means optimal quantization strategy differs between:
- **Decode-optimized:** Quantize aggressively (Int4) — memory bandwidth is the bottleneck, dequant overhead is hidden
- **Prefill-optimized:** Keep FP16 — GPU compute is saturated, dequant overhead is visible

A future "adaptive quantization" system could select precision per-layer based on the current operation mode. This aligns with the TurboQuant KV cache approach (compress for decode, decompress for attention) applied to model weights.

### Next Steps

- **Phase 2g (TurboQuant Int8):** Test Int8 keys + Int4 values for KV cache — same per-component precision principle
- **Future:** Investigate runtime bit-width selection for expert projections (Int4 during decode, FP16 during prefill via dual weight storage)
- **Future:** Custom quantization recipes per model architecture (e.g., Gemma4's large head_dim=512 global attention may need higher precision than head_dim=256 sliding attention)
