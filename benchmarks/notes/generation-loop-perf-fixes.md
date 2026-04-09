# Gemma 4 Forward Pass Performance Fixes

**Date**: 2026-04-08
**Model**: Gemma 4 E2B 4bit (`mlx-community/gemma-4-e2b-it-4bit`)
**Hardware**: M1 Max 64GB

## Context

Controlled A/B testing against mlx-vlm (Python) showed our decode throughput was ~20% slower
on the same hardware with the same model. CPU profiling (`MLX_CPU_PROFILE=1`) revealed that
`asyncEval` consumed 85% of per-token time (11ms/token), confirming the GPU forward pass is
the bottleneck — not CPU overhead, sampling, or stream scheduling.

## Root Causes Found

### 1. Norm-before-transpose ordering (highest impact)
Python applies Q/K/V RMSNorm on `[B, L, heads, dim]` THEN transposes to `[B, heads, L, dim]`.
Our Swift code was transposing FIRST, then norming on the transposed layout. This changes memory
access patterns for the fused RMSNorm Metal kernel across all 35 attention layers.

### 2. Missing kernel fusion — logit softcapping
Python compiles `tanh(x/softcap)*softcap` into one fused kernel via `@mx.compile(shapeless=True)`.
We were dispatching 3 separate kernels (divide, tanh, multiply) on the 262K vocab.

### 3. Missing kernel fusion — GEGLU in MLP
Python compiles `gelu_approx(gate) * x` into one fused kernel. We ran them as separate
ops across 35 layers.

### 4. Embedding scale dtype overhead
We created `MLXArray(sqrt(1536), dtype: .bfloat16)` then `.asType(h.dtype)` — an unnecessary
dtype conversion. Changed to plain `Float` scalar multiplication.

### 5. Per-layer input re-slicing
Python pre-slices all 35 per-layer inputs upfront as a list. We were re-slicing the 4D
tensor inside the loop on each iteration.

## Fixes Applied

| Fix | File | Lines |
|-----|------|-------|
| Norm before transpose | `Gemma4.swift` | Attention forward pass |
| Compiled logit softcap | `Gemma4.swift` | `compiledLogitSoftcap` module-level |
| Compiled GEGLU | `Gemma4.swift` | `compiledGeglu` module-level |
| Scalar embed scale | `Gemma4.swift` | `Gemma4ModelInner.callAsFunction` |
| Pre-sliced PLE | `Gemma4.swift` | `perLayerSlices` array before loop |
| Profiling cleanup | `Evaluate.swift` | Single `profiling` guard in `next()` |

## Results (no KV quantization)

| Context | Before Prefill | After Prefill | Before Decode | After Decode |
|---------|---------------|--------------|--------------|-------------|
| 1024 | 1,013 tok/s | 1,067 tok/s (+5.3%) | 80.1 tok/s | 81.8 tok/s (+2.1%) |
| 4096 | 1,619 tok/s | 1,659 tok/s (+2.5%) | 78.9 tok/s | 80.7 tok/s (+2.3%) |
| 16384 | 2,576 tok/s | 2,633 tok/s (+2.2%) | 68.6 tok/s | 72.1 tok/s (+5.1%) |

## What Didn't Work

### Dedicated generation stream (6x regression)
Wrapping the generation loop in `Stream.withNewDefaultStream(device: .gpu)` caused decode to
drop from 80 tok/s to 13.5 tok/s. Root cause: in Metal, creating a separate `MTLCommandQueue`
(MLX Stream) causes cross-queue synchronization every time the forward pass reads model weights
from the default queue. The `asyncEval` pipelining breaks because work can't overlap across
queues without explicit barrier management.

Python mlx-lm gets away with this because it creates `generation_stream` once at module import
and uses it consistently. The model loading and generation both dispatch on the same persistent
stream. In our Swift code, model weights are loaded on the default stream, and switching to a
new stream for generation creates constant cross-stream dependencies.

A `generationStream` constant is declared in `Evaluate.swift` for future use once we figure
out how to pre-warm it with model weights.

### Compiled sampling (CumSum crash)
Compiling `TopPSampler.sample()` with `compile(shapeless: true)` crashed because `CumSum`
cannot infer output shapes without concrete input dimensions. Without `shapeless`, the compiled
function needs to recompile whenever input shapes change. Individual function compilation
(matching Python's per-function `@mx.compile`) is feasible but needs random state management
via `compile(inputs: [randomState], outputs: [randomState])` — deferred for follow-up.

## Remaining Gap

After fixes: 80.7 tok/s at 4k. mlx-vlm Python: 99 tok/s at 4k. Remaining gap: ~18%.

Likely causes:
- Per-token dispatch overhead differences between Swift and Python MLX bindings
- Model weight access patterns (Python's lazy eval graph structure may differ)
- The persistent stream approach (Python uses one, we don't yet)
