# Gemma 4 (26b / 31b) — Full Decode Code Trace

Investigation of decode-side incoherence for Gemma 4 26b-A4B (MoE) and 31b (dense).
Traces all code paths through `mlx-swift-lm`, `mlx-swift`, `mlx` (C++), and `mlx-c`.

**Date**: 2026-04-14

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Configuration Parsing](#2-configuration-parsing)
3. [Cache Creation](#3-cache-creation)
4. [Prefill Path (prepare)](#4-prefill-path)
5. [Decode Path — Full Trace](#5-decode-path)
6. [Attention — Fused NormRoPE + SDPA](#6-attention)
7. [MoE Routing (26b)](#7-moe-routing)
8. [Per-Layer Embeddings (PLE)](#8-per-layer-embeddings)
9. [Metal Kernel Dispatch — SDPA](#9-metal-sdpa)
10. [Metal Kernel Dispatch — RoPE](#10-metal-rope)
11. [mlx-c Bridge Layer](#11-mlx-c-bridge)
12. [KV Sharing Mechanism](#12-kv-sharing)
13. [Verified Correct Behaviors](#13-verified)
14. [Bugs & Potential Issues Found](#14-bugs)
15. [MLXArray Safety Audit](#15-safety-audit)

---

## 1. Architecture Overview

Both 26b and 31b share these features:

| Feature | Value |
|---------|-------|
| Attention types | Alternating `sliding_attention` + `full_attention` |
| Sliding head_dim | 256 |
| Full/global head_dim | 512 |
| Sliding RoPE | Standard, base=10,000, all dims rotated |
| Full RoPE | ProportionalRoPE, base=1,000,000, 25% dims rotated |
| Attention scale | **1.0** (NOT 1/sqrt(d)) — relies on QK-norm |
| K=V optimization | Full attention only: V = kProj(x), no vProj |
| Sliding window | 1024 tokens (RotatingKVCache) |
| Full attention cache | Unbounded (StandardKVCache / KVCacheSimple) |
| KV sharing | Last N layers reuse K/V from last non-shared of same type |
| Q/K norm | Learnable RMSNorm per head |
| V norm | RMSNorm without learnable weight (unit scale) |
| Final logit softcap | tanh(logits/30) * 30 |
| Layer scalar | Learnable per-layer output scale |
| Fused norm+RoPE | MLXFast.rmsNormRoPE Metal kernel (default on) |

**26b-specific**: MoE with FusedGateUpSwitchGLU, Gemma4Router, per-expert scale
**31b-specific**: Dense MLP only, no MoE

---

## 2. Configuration Parsing

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:171-337`

```
Gemma4TextConfiguration.init(from decoder:)
├── Decodes from config.json
├── modelType, hiddenSize, hiddenLayers, intermediateSize, ...
├── headDim (default: 256)         — sliding attention head dimension
├── globalHeadDim (default: 512)   — full attention head dimension
├── kvHeads (default: 8)           — sliding KV heads
├── globalKvHeads (fallback: kvHeads) — full KV heads
├── slidingWindow (default: 1024)
├── layerTypes: [String]           — ["sliding_attention", "full_attention", ...]
├── enableMoeBlock: Bool           — true for 26b, false for 31b
├── attentionKEqV: Bool            — K=V optimization flag
├── numKvSharedLayers: Int         — shared layer count (9 for 26b)
├── finalLogitSoftcapping: Float?  — 30.0
├── RoPE parameters (lines 283-298):
│   ├── Try nested rope_parameters.sliding_attention.rope_theta
│   ├── Try nested rope_parameters.full_attention.rope_theta
│   ├── Try nested rope_parameters.full_attention.partial_rotary_factor
│   └── Fallback to top-level defaults
├── partialRotaryFactor (default: 0.25) — 25% of full headDim rotated
├── hiddenSizePerLayerInput — PLE dimension (0 = disabled)
└── vocabSizePerLayerInput — PLE vocab size
```

---

## 3. Cache Creation

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:1265-1283`

```
Gemma4TextModel.newCache(parameters:)
├── For each layerType in config.layerTypes:
│   ├── "full_attention":
│   │   ├── If maxKVSize set → RotatingKVCache(maxSize: maxKVSize, keep: 0)
│   │   └── Else → StandardKVCache()  (unbounded, KVCacheSimple typedef)
│   └── "sliding_attention":
│       └── RotatingKVCache(maxSize: config.slidingWindow=1024, keep: 0)
└── Returns [KVCache] — one per layer (including shared layers, which don't use theirs)
```

**StandardKVCache (KVCacheSimple)** — `KVCache.swift:355-522`
- Grows by `step=256` token increments
- `offset` tracks total tokens cached
- `update()`: writes at `keys[..., previous..<offset, ...]`, returns `keys[..., ..<offset, ...]`

**RotatingKVCache** — `KVCache.swift:525-846`
- Circular buffer, `maxCacheSize` fixed at init
- `idx`: current write position (wraps at maxCacheSize)
- `offset`: total tokens seen (never wraps)
- Single-token decode: `updateSingleToken()` → write at `idx`, increment, wrap
- Multi-token prefill: `updateMultiToken()` → concatenate + trim to tail

---

## 4. Prefill Path

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:1129-1192`

```
Gemma4TextModel.prepare(input, cache, windowSize)
├── prefillStepSize = max(windowSize ?? 512, 4096)
├── Optional: GemmaPrefillBridge (NATIVE_PREFILL=1)
├── while y.tokens.size > 1:
│   ├── chunkSize = min(prefillStepSize, y.tokens.size - 1)
│   ├── input = y[.newAxis, ..<chunkSize]
│   ├── model(input.tokens, cache: cache, prefillMode: true)  ← SHARED LAYERS SKIPPED
│   ├── asyncEval(nonSharedCacheArrays)   ← GPU/CPU overlap
│   └── y = y[chunkSize...]
├── MLX.Memory.clearCache()
└── return .tokens(y)   ← last token for decode loop
```

**Key**: During prefill (`prefillMode: true`), shared layers are completely skipped (line 1074-1077).
This is correct: shared layers don't update any cache, so their effect on `h` is dead code.

---

## 5. Decode Path — Full Trace

### Entry: Token Generation Loop

**File**: `Libraries/MLXLMCommon/Evaluate.swift`

```
generateLoopTask()                                          (line 2111)
├── SendableBox wraps TokenIterator + Handler                (line 2123-2124)
├── Task {                                                   (line 2127)
│   ├── iterator = iterator.consume()                        (line 2133)
│   └── while let token = iterator.next() {                  (line 2146)
│       ├── handler.onToken(token, emit: continuation.yield) (line 2173)
│       └── Check EOS, cancellation                          (line 2148, 2163)
```

### TokenIterator.next()

**File**: `Libraries/MLXLMCommon/Evaluate.swift:1369-1445`

```
TokenIterator.next()
├── Check pendingTokens / ngramStep()                       (line 1372-1378)
├── previousY = y                                            (line 1400)
├── y = LMInput.Text(tokens: token)                          (line 1401)
├── step(previous: previousY)                                (line 1404)
│   ├── model(previous[text: .newAxis], cache, state)        (line 1189)
│   │   → Gemma4TextModel.callAsFunction()
│   │     → Gemma4ModelInner.callAsFunction()                  (FULL TRACE BELOW)
│   ├── maybeQuantizeKVCache()                               (line 1199)
│   └── convertToToken(logits: result.logits)                (line 1207)
│       ├── logits = logits[0, -1, 0...]                     — last position
│       ├── processor.process(logits:)                       — repetition/freq penalty
│       └── sampler.sample(logits:)                          — TopPSampler
├── y = LMInput.Text(tokens: sampledToken)                   (line 1406)
├── asyncEval(sampledToken)                                  (line 1407) ← GPU overlap
├── processor.didSample(token:)                              (line 1409)
├── tokenCount += 1                                          (line 1411)
├── if tokenCount % 256 == 0: MLX.Memory.clearCache()        (line 1412)
├── tokenId = previousY.tokens.item(Int.self)                (line 1414) ← GPU→CPU sync
├── Phase classification (thinking/generation logprobs)       (line 1416-1435)
└── return Int(tokenId)                                      (line 1437)
```

### Model Forward Pass (Decode, L=1)

**File**: `Libraries/MLXLLM/Models/Gemma4.swift`

```
Gemma4TextModel.callAsFunction(inputs, cache)                (line 1194)
├── out = model(inputs, cache)                                (line 1195)
│   → Gemma4ModelInner.callAsFunction(inputs, cache, prefillMode=false)  (line 967)
│
│   ┌─── EMBEDDING ────────────────────────────────────────
│   ├── h = embedTokens(inputs)                               (line 973)
│   ├── h = h * sqrt(Float(config.hiddenSize))                 (line 977)
│   │
│   ┌─── PER-LAYER EMBEDDINGS (PLE) ──────────────────────
│   ├── if hiddenSizePerLayerInput > 0:                        (line 983)
│   │   ├── pli = embedTokensPerLayer(inputs) * scale          (line 985-993)
│   │   ├── pli.reshaped(B, T, numLayers, plDim)               (line 995-996)
│   │   ├── plProj = perLayerModelProjection(h) * scale        (line 999-1002)
│   │   ├── plProj = perLayerProjectionNorm(plProj)            (line 1005-1006)
│   │   └── pli = (plProj + pli) * 2^(-0.5)                   (line 1009)
│   │
│   ┌─── MASK CREATION ───────────────────────────────────
│   ├── seqLen = h.dim(1)  // = 1 during decode                (line 1024)
│   ├── fullMask = nil, slidingMask = nil                      (line 1025-1026)
│   │   // Lazy: created on first use per type
│   │   // For n=1 decode: both resolve to .none               (line 1044-1061)
│   │
│   ┌─── DONOR OFFSET TRACKING ───────────────────────────
│   ├── donorPreUpdateOffsets = [0, 0, ..., 0]  (N elements)   (line 1033)
│   ├── intermediateKVs = [nil, nil, ..., nil]  (N elements)   (line 1037)
│   │
│   ┌─── LAYER LOOP ──────────────────────────────────────
│   ├── for (i, layer) in layers.enumerated():                 (line 1039)
│   │
│   │   ┌─ MASK SELECTION ─────────────────────────────────
│   │   ├── if layerTypes[i] == "full_attention":
│   │   │   └── maskMode = fullMask ?? makeAttentionMask(n:1, cache:cache[fullIdx], windowSize:nil)
│   │   │       → cache.makeMask(n:1, windowSize:nil)
│   │   │       → BaseKVCache default: n==1 → return .none
│   │   └── else: // sliding_attention
│   │       └── maskMode = slidingMask ?? makeAttentionMask(n:1, cache:cache[slidingIdx], windowSize:1024)
│   │           → RotatingKVCache.makeMask(n:1, windowSize:1024)
│   │           → maxCacheSize(1024) == windowSize(1024) → .none
│   │
│   │   ┌─ DONOR vs SHARED ───────────────────────────────
│   │   ├── donorIdx = previousKVs[i]
│   │   ├── isShared = (donorIdx != i)
│   │   │
│   │   ├── if isShared (layers M..N-1):                       (line 1068)
│   │   │   └── h = layer(h, mask: maskMode, cache: nil,
│   │   │         useSharedKV: true,
│   │   │         sharedKVArrays: intermediateKVs[donorIdx],
│   │   │         donorOffset: donorPreUpdateOffsets[donorIdx])  (line 1078-1080)
│   │   │
│   │   └── else (donor layers 0..M-1):                        (line 1081)
│   │       ├── donorPreUpdateOffsets[i] = cache[i]?.offset     (line 1082)
│   │       ├── h = layer(h, mask: maskMode, cache: cache[i])  (line 1083)
│   │       └── intermediateKVs[i] = (cache[i].lastReturnedKeys,
│   │                                  cache[i].lastReturnedValues)  (line 1086-1091)
│   │
│   └── return norm(h)                                         (line 1094)
│
├── Logit projection:                                          (line 1196-1200)
│   ├── if tieWordEmbeddings → model.embedTokens.asLinear(out)
│   └── else → lmHead(out)
│
└── Logit softcapping:                                         (line 1203-1205)
    └── if softcap > 0: out = compiledLogitSoftcap(30.0, out)
        → tanh(out / 30.0) * 30.0
```

---

## 6. Attention — Detailed Trace

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:559-648`

### Normal Path (Donor Layer)

```
Gemma4Attention.callAsFunction(x, mask, cache, ...)
├── (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))              (line 568)
│   // B=1, L=1 during decode
│
├── queries = qProj(x).reshaped(B, L, nHeads, -1)            (line 570)
│   // [1, 1, nHeads, headDim]
│
├── keys = kProj(x).reshaped(B, L, nKVHeads, -1)             (line 600)
│
├── values:                                                    (line 602-606)
│   ├── if attentionKEqV: values = keys                       (K=V: full attn only)
│   └── else: values = vProj(x).reshaped(...)
│
├── values = MLXFast.rmsNorm(values, weight: mlxNone, eps)    (line 609)
│   // V norm: normalize without learnable weight
│   → mlx_fast_rms_norm(values, null_weight, eps)
│   → C++ fast::rms_norm() with weight=nullopt
│   → Metal rms_norm kernel: x * rsqrt(mean(x^2) + eps)
│
├── offset = cache?.offset ?? 0                                (line 611)
│
├── if _fusedInvFreqs (DEFAULT ON):                            (line 612)
│   ├── queries = MLXFast.rmsNormRoPE(                        (line 615-617)
│   │     queries, weight: qNorm.weight, invFreqs: invFreqs,
│   │     eps: rmsNormEps, offset: offset, nHeads: nHeads, seqLen: L)
│   │   → mlx_fast_rms_norm_rope()
│   │   → C++ fast::rms_norm_rope()
│   │   → Metal rms_norm_rope kernel (see §10)
│   ├── queries = queries.transposed(0, 2, 1, 3)             (line 618)
│   │   // [B, L, H, D] → [B, H, L, D]
│   ├── keys = MLXFast.rmsNormRoPE(                           (line 619-621)
│   │     keys, weight: kNorm.weight, invFreqs: invFreqs, ...)
│   └── keys = keys.transposed(0, 2, 1, 3)                   (line 622)
│
├── else (FUSED DISABLED via GEMMA4_FUSED_NORM_ROPE=0):       (line 623)
│   ├── queries = qNorm(queries)
│   ├── keys = kNorm(keys)
│   ├── queries = queries.transposed(0, 2, 1, 3)
│   ├── keys = keys.transposed(0, 2, 1, 3)
│   ├── queries = rope(queries, offset: offset)
│   │   → ProportionalRoPE or RoPE depending on layer type
│   └── keys = rope(keys, offset: offset)
│
├── values = values.transposed(0, 2, 1, 3)                    (line 633)
│
├── output = attentionWithCacheUpdate(                         (line 635-642)
│     queries, keys, values, cache, scale=1.0, mask=.none)
│   → AttentionUtils.swift:37-87
│   ├── cache.update(keys, values) → (cachedK, cachedV)
│   │   → StandardKVCache.update() or RotatingKVCache.updateSingleToken()
│   └── MLXFast.scaledDotProductAttention(
│         queries, cachedK, cachedV, scale=1.0, mask=.none)
│       → mlx_fast_scaled_dot_product_attention(
│             queries, keys, values, 1.0, "", null_mask, null_sinks)
│       → C++ fast::scaled_dot_product_attention()   (fast.cpp:902)
│       → ScaledDotProductAttention primitive
│       → Metal sdpa_vector kernel (decode, L_q=1)   (see §9)
│
├── output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1) (line 643-644)
└── return oProj(output)                                       (line 646)
```

### Shared Layer Path

```
Gemma4Attention.callAsFunction(x, mask, cache=nil,
    useSharedKV=true, sharedKVArrays=(donorK, donorV), donorOffset=X)
├── queries = qProj(x).reshaped(B, L, nHeads, -1)            (line 570)
├── offset = donorOffset  (= donor's pre-update offset)       (line 574)
├── queries = rmsNormRoPE(queries, qNorm.weight, invFreqs, offset=X) (line 576-578)
├── queries = queries.transposed(0, 2, 1, 3)                  (line 579)
├── output = MLXFast.scaledDotProductAttention(                (line 586-592)
│     queries, cachedKeys=donorK, cachedValues=donorV,
│     scale=1.0, mask=.none)
│   // Uses donor's full cached K/V directly (no cache.update)
│   // No new K/V computed — only Q projected and RoPE'd
└── return oProj(output.transposed(...).reshaped(...))         (line 593-596)
```

---

## 7. MoE Routing (26b Only)

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:830-864`

```
Gemma4TransformerBlock.callAsFunction() — MoE path
├── preFFNNorm = preFeedforwardLayerNorm(h)                    (line 836)
├── h1 = sharedMLP(preFFNNorm)                                 (line 837)
│   → Gemma4SharedMLP: downProj(gelu(gateProj(x)) * upProj(x)) (line 696-698)
├── h1 = postFeedforwardLayerNorm1(h1)                         (line 838)
│
├── routerLogits = router(h)                                   (line 841)
│   → Gemma4Router.callAsFunction():                           (line 715-724)
│     ├── normWeight = scale * sqrt(dimensions)
│     ├── normed = MLXFast.rmsNorm(x, weight: normWeight, eps)
│     └── return proj(normed)   // [B, T, numExperts]
│
├── (topKLogits, topKIndices) = gemma4TopK(routerLogits, k=topKExperts) (line 842)
│   → gemma4TopK():                                            (line 341-348)
│     ├── partitionedIndices = argPartition(a, kth: -k, axis: -1)
│     ├── topKIndices = partitionedIndices[.ellipsis, (-k)...]
│     └── topKValues = takeAlong(a, topKIndices, axis: -1)
│
├── stopIndices = MLX.stopGradient(topKIndices)                (line 843)
├── expertWeights = softmax(topKLogits, axis: -1, precise: true) (line 844)
├── expertWeights = expertWeights * router.perExpertScale[topKIndices] (line 845)
│   ⚠️ No bounds check on topKIndices into perExpertScale
│
├── preFFNNorm2 = preFeedforwardLayerNorm2(h)                  (line 846)
├── h2 = experts(preFFNNorm2, stopIndices)                     (line 847)
│   → FusedGateUpSwitchGLU: routes input to selected experts
├── h2 = h2 * expandedDimensions(expertWeights, axis: -1)      (line 848)
├── h2 = h2.sum(axis: -2)                                     (line 849)
├── h2 = postFeedforwardLayerNorm2(h2)                         (line 850)
│
├── ffnOut = h1 + h2                                           (line 852)
└── h = rmsNormResidual(ffnOut, residual: h, ...)              (line 853-856)
```

---

## 8. Per-Layer Embeddings (PLE)

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:866-877`

```
Gemma4TransformerBlock.callAsFunction() — PLE path (per layer)
├── if perLayerInputGate exists:                               (line 867)
│   ├── residual = h                                           (line 872)
│   ├── g = compiledGeluMul(gate(h), pli)                      (line 873)
│   │   → gelu_approx(gate(h)) * pli
│   ├── g = perLayerProjection(g)                              (line 874)
│   ├── g = postPerLayerInputNorm(g)                           (line 875)
│   └── h = residual + g                                      (line 876)
│
└── h = h * layerScalar                                        (line 880)
```

---

## 9. Metal SDPA Kernel — Decode Path

### Dispatch Decision

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:599-664`

```
ScaledDotProductAttention::use_fallback()
├── if is_training → true (fallback)
├── if output_logsumexp → true
├── if device == CPU → true
│
├── sdpa_vector_supported_head_dim:                             (line 634)
│   query_head_dim == value_head_dim &&
│   (64 || 96 || 128 || 256 || (512 && is_half))
│   ✅ Gemma4 256: supported (all dtypes)
│   ✅ Gemma4 512: supported (float16/bfloat16)
│
├── supports_sdpa_vector:                                       (line 658)
│   query_seq_len <= 8 && query_seq_len <= key_seq_len &&
│   sdpa_vector_supported && (query_seq_len * gqa_factor) <= 32
│   ✅ For decode (L_q=1): always true for Gemma4
│
└── return !(supports_sdpa_full || supports_sdpa_vector)
```

### sdpa_vector Kernel

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h:16-177`

```
sdpa_vector<T, D=256|512, V=D>
├── Thread organization:
│   ├── threadgroup: 1024 threads (BN=32 simd groups × 32 lanes)
│   ├── grid: (B*H, query_seq_len=1, 1)
│   ├── qk_per_thread = D/32   → 8 (D=256) or 16 (D=512)
│   └── v_per_thread = V/32    → same
│
├── Load Q and apply scale:                                    (line 84-86)
│   for i in 0..<qk_per_thread:
│       q[i] = float(scale) * queries[i]
│   // scale = 1.0 for Gemma4 → q[i] = float(queries[i])
│
├── For each key position (i = simd_gid; i < N; i += BN):     (line 99)
│   ├── Mask check:                                             (line 100-107)
│   │   ├── do_causal: use_key = i <= (N - tpg.y + q_seq_idx)
│   │   ├── bool_mask: use_key = bmask[0]
│   │   ├── float_mask: use_key = (fmask[0] >= finite_min)
│   │   └── For Gemma4 decode: mask=.none → no mask applied
│   │
│   ├── Load key, compute score = sum(q[j] * k[j])            (line 110-119)
│   │   → simd_sum() across 32 lanes
│   │
│   ├── Add float mask if present:                              (line 120-122)
│   │   score += float(fmask[0])
│   │
│   ├── Online softmax update:                                  (line 124-131)
│   │   new_max = max(max_score, score)
│   │   factor = fast::exp(max_score - new_max)    ← NATURAL exp
│   │   exp_score = fast::exp(score - new_max)     ← NATURAL exp
│   │
│   └── Accumulate output:                                      (line 133-135)
│       o[j] = o[j] * factor + exp_score * values[j]
│
├── Cross-simd-group reduction:                                (line 149-169)
│   ├── Share max_scores and sum_exp_scores via threadgroup memory
│   ├── Compute global max and normalize
│   └── Transpose + reduce output accumulator
│
└── Write output (simd_lid == 0):                              (line 172-176)
    out[i] = T(o[i])
```

### Steel Attention Kernel (Prefill, for reference)

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`

```
attention<T, BQ, BK, BD=256|512, ...>
├── Scale transformation:                                       (line 166)
│   scale = params->scale * M_LOG2E_F    (= 1.0 * 1.4427 = 1.4427)
│
├── Score computation:
│   Stile = Q @ K.T                                            (line 259-276)
│   Stile *= scale                                              (line 280-282)
│
├── Softmax (log2 domain):                                      (line 394-395)
│   Stile = exp2(Stile - rowmax(Stile))    ← BASE-2 exp
│   factor = exp2(max_old - max_new)       ← BASE-2 exp
│
└── Mathematically equivalent to sdpa_vector:
    exp2(log2(e) * x) = e^x = exp(x)    ✅ VERIFIED IDENTICAL
```

---

## 10. Metal RoPE Kernels

### Fused RMSNorm+RoPE Kernel

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/rms_norm_rope.metal:28-96`

```
rms_norm_rope<T>(x, w, inv_freqs, out, eps, axis_size, offset, n_heads, seq_len)
├── Thread organization:
│   ├── half_dim = axis_size / 2
│   ├── threadgroup_size = half_dim (rounded up to SIMD multiple)
│   │   head_dim=256 → 128 threads, head_dim=512 → 256 threads
│   ├── One threadgroup per row (batch × seq_pos × head)
│   └── Each thread handles one rotation pair (lid, lid + half_dim)
│
├── Phase 1: RMS norm                                          (line 48-76)
│   ├── v1 = float(x[lid]),  v2 = float(x[lid + half_dim])
│   ├── sum_sq = v1² + v2²
│   ├── simd_sum(sum_sq)  → per-SIMD-group partial sums
│   ├── Cross-SIMD reduction via threadgroup memory
│   └── inv_rms = rsqrt(total / axis_size + eps)
│
├── Phase 2: Weight + RoPE                                     (line 82-95)
│   ├── Position: l = (gid / n_heads) % seq_len
│   ├── pos = float(offset + int(l))
│   ├── normed_a = v1 * w[lid] * inv_rms
│   ├── normed_b = v2 * w[lid + half_dim] * inv_rms
│   ├── theta = pos * inv_freqs[lid]
│   │   For ProportionalRoPE: inv_freqs has 0s for non-rotated dims
│   │   theta = 0 → cos=1, sin=0 → pass-through ✅
│   ├── out[lid] = normed_a * cos(theta) - normed_b * sin(theta)
│   └── out[lid+half_dim] = normed_a * sin(theta) + normed_b * cos(theta)
```

### ProportionalRoPE Frequency Construction

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:489-501`

```
Fused path (for full attention, headDim=512, rotatedDims=128):
├── propExponents = [0, 2, 4, ..., 126] / 512
├── realFreqs = globalRopeTheta ^ propExponents    (64 values, positive exponents)
├── infPadding = [inf, inf, ..., inf]              (192 values)
├── allFreqs = [realFreqs..., infPadding...]       (256 values)
└── _fusedInvFreqs = 1.0 / allFreqs
    = [theta_0, theta_1, ..., theta_63, 0, 0, ..., 0]   (256 values)
    where theta_i = globalRopeTheta^(-2i/512)
```

### Standard RoPE (C++ Fallback)

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp:724-808`

```
rope() fallback:
├── if custom freqs provided:
│   inv_freqs = reciprocal(custom_freqs)           (line 758-759)
│   ← IMPORTANT: C++ inverts the provided freqs!
├── else:
│   inv_freqs = exp(arange(0, -dims/2, -1) * log(base) / (dims/2))
├── theta = positions * inv_freqs
├── coss = cos(theta), sins = sin(theta)
├── Non-traditional pairing: (x[0..half], x[half..dims])
│   out_1 = x1 * cos - x2 * sin
│   out_2 = x1 * sin + x2 * cos
└── Concatenate rotated + unrotated dims
```

---

## 11. mlx-c Bridge Layer

**File**: `mlx-swift/Source/Cmlx/mlx-c/mlx/c/fast.cpp`

All bridges follow the same pattern — no transformations, direct pass-through:

```
Swift                  → C (mlx-c)                  → C++ (mlx)
─────────────────────────────────────────────────────────────────
MLXFast.scaledDotProductAttention()
  queries.ctx            mlx_array_get_(queries)      fast::scaled_dot_product_attention()
  keys.ctx               mlx_array_get_(keys)
  values.ctx             mlx_array_get_(values)
  scale (Float)          scale (float)                 scale (float)
  mask.mode (String)     std::string(mask_mode)        mask_mode (std::string)
  mask.mask?.ctx         ctx ? optional(get) : nullopt  optional<array>
  │                                                     │
  └── .mlxNone.ctx = nullptr → nullopt                  └── No mask for .none

MLXFast.rmsNormRoPE()
  x.ctx                  mlx_array_get_(x)            fast::rms_norm_rope()
  weight.ctx             mlx_array_get_(weight)
  invFreqs.ctx           astype(invFreqs, float32)     inv_freqs forced to float32
  eps (Float)            eps (float)
  Int32(offset)          offset (int)                  ← Int→Int32 truncation (safe: <100K)
  Int32(nHeads)          n_heads (int)
  Int32(seqLen)          seq_len (int)

MLXFast.RoPE()
  freqs?.ctx             ctx ? optional(get) : nullopt  optional<array> freqs
                         if freqs: reciprocal(freqs)   ← C++ INVERTS custom freqs
```

---

## 12. KV Sharing Mechanism

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:930-945`

```
KV Donor Mapping (VERIFIED: matches Python mlx-lm exactly):
├── N = config.hiddenLayers (e.g., 35)
├── M = N - config.numKvSharedLayers (e.g., 26)
├── mapping = [0, 1, 2, ..., N-1]  (identity)
├── kvsByType: [String: Int] = [:]
├── for i in 0..<M:
│   kvsByType[layerTypes[i]] = i   ← overwrites → keeps LAST of each type
├── for j in M..<N:
│   mapping[j] = kvsByType[layerTypes[j]]  ← all shared→last donor of type
│
│ Example (35 layers, 9 shared):
│   Layers 0-25: non-shared (donors)
│   Layer 24 = last sliding non-shared
│   Layer 25 = last full non-shared
│   Layers 26-34: shared
│     Shared sliding → all map to layer 24
│     Shared full → all map to layer 25
```

### Decode Flow with KV Sharing

```
Layer 24 (donor, sliding):
  ├── donorPreUpdateOffsets[24] = cache[24].offset = X         CAPTURE
  ├── h = layer(h, cache: cache[24])                           RUN
  │   └── cache[24].offset becomes X+1
  └── intermediateKVs[24] = cache[24].lastReturned(K,V)        STORE

Layer 25 (donor, full):
  ├── donorPreUpdateOffsets[25] = cache[25].offset = X         CAPTURE
  ├── h = layer(h, cache: cache[25])
  └── intermediateKVs[25] = cache[25].lastReturned(K,V)

Layer 26 (shared, sliding):
  └── h = layer(h, cache: nil,
        useSharedKV: true,
        sharedKVArrays: intermediateKVs[24],                   ← donor 24's K/V
        donorOffset: donorPreUpdateOffsets[24] = X)            ← donor 24's pre-update offset
        → Q RoPE at position X (same as donor's Q/K position) ✅
```

---

## 13. Verified Correct Behaviors

| Behavior | Verification | Location |
|----------|-------------|----------|
| Scale domain (decode vs prefill) | `exp(s*x)` ≡ `exp2(s*log2e*x)` | sdpa_vector.h:126 vs steel_attention.h:166,395 |
| KV sharing mapping | Matches Python mlx-lm exactly | Gemma4.swift:930-945 vs gemma4_text.py:424-433 |
| ProportionalRoPE freqs | Fused: `1/base^(2i/d)` direct. Non-fused: C++ `reciprocal()` of `base^(2i/d)` | Gemma4.swift:489-501, fast.cpp:758-759 |
| Non-rotated dims pass-through | `inv_freqs=0` → `theta=0` → `cos=1,sin=0` → identity | rms_norm_rope.metal:89-95 |
| V norm (weight=none) | `rmsNorm(x, none)` = `rmsNorm(x, ones)` = `x*rsqrt(mean(x²)+eps)` | Gemma4.swift:609 |
| Decode mask = .none | Correct: single Q attends to all cached K/V | KVCache.swift:793-813 |
| RotatingKVCache circular | RoPE encoded in key values, not buffer position | Gemma4.swift:611, KVCache.swift:620-624 |
| Prefill shared skip | Dead code: shared layers don't update cache | Gemma4.swift:1074-1077 |
| mlx-c bridge | No parameter transformations, correct null handling | mlx-c/fast.cpp |

---

## 14. Bugs & Potential Issues Found

### BUG: rms_norm_rope C++ fallback double-inverts frequencies

**File**: `mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp:185-203`
**Severity**: Low (GPU-only in production)

```cpp
auto fallback = [...](const std::vector<array>& inputs) {
    auto normed = rms_norm(inputs[0], inputs[1], eps, s);
    auto transposed = transpose(normed, {0, 2, 1, 3}, s);
    auto rotated = rope(transposed, ..., offset,
        inputs[2],  // ← _fusedInvFreqs = base^(-2i/d), already inverted
        s);
    // But rope() does: inv_freqs = reciprocal(inputs[2])
    // = reciprocal(base^(-2i/d)) = base^(+2i/d) ← WRONG direction!
```

The fallback passes already-inverted frequencies to `rope()`, which inverts them again.
Result: `theta = position * base^(+2i/d)` instead of `theta = position * base^(-2i/d)`.

**Impact**: Only triggers on CPU (`use_fallback()` returns true only for `device == CPU`).
The Metal kernel path is correct.

### CONCERN: bfloat16 precision with head_dim=512

With 512-dimensional heads and QK-norm (scores in [-1, 1]):
- Small score differences are critical for attention weights
- bfloat16 has ~7 bits of mantissa (0.4% relative error)
- 512-wide dot product accumulates in float32 (safe), but Q/K stored as bfloat16
- No immediate bug, but could amplify subtle errors over 35+ layers

### CONCERN: Fused rmsNormRoPE with head_dim=512

- 256 threads per threadgroup (half_dim), 8 SIMD groups
- Within Metal limits, but more cross-SIMD communication than smaller heads
- Verified correct via manual trace of reduction logic

---

## 15. MLXArray Safety Audit

### Thread Safety Violations

#### T1. TokenIterator crosses thread boundary via SendableBox (CRITICAL)

**File**: `Libraries/MLXLMCommon/Evaluate.swift:2123-2133`

```swift
let iterator = SendableBox(iterator)  // wraps in @unchecked Sendable
let task = Task {
    var iterator = iterator.consume()  // MLXArray-containing struct crosses thread
```

TokenIterator contains `cache: [KVCache]` (MLXArray), `y: LMInput.Text` (MLXArray),
`logProbSum: MLXArray`, etc. All created in one context, consumed in Task's context.
SendableBox bypasses compiler checks but doesn't make MLXArray thread-safe.

#### T2. asyncEval() without thread affinity guarantee

**File**: `Libraries/MLXLMCommon/Evaluate.swift:1351, 1362, 1407`

```swift
asyncEval(y.tokens)  // Schedule GPU eval
// ... later ...
let tokenId = previousY.tokens.item(Int.self)  // Consume result
```

`asyncEval` schedules GPU work. The subsequent `.item()` call synchronizes.
If the Task executor switches threads between these calls, the MLXArray
graph is built on one thread and evaluated on another.

**Mitigation**: In practice, Swift cooperative thread pool usually keeps tasks
on the same thread for short sequences, but this is NOT guaranteed.

#### T3. Compiled functions marked @Sendable capture MLXArray state

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:653-680`

```swift
private let compiledLogitSoftcap: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { softcap, x in tanh(x / softcap) * softcap }
```

These file-level compiled closures are `@Sendable` and callable from any thread.
If two model instances share these (they do — they're module-level `let`s),
concurrent calls could conflict in MLX's compilation cache.

### Memory Safety Violations

#### M1. RotatingKVCache.trim() can make idx negative (CRITICAL)

**File**: `Libraries/MLXLMCommon/KVCache.swift:765-771`

```swift
public override func trim(_ n: Int) -> Int {
    let trimmed = min(offset, n)
    offset -= trimmed
    idx -= trimmed        // ← If idx < trimmed, idx becomes NEGATIVE
    return trimmed
}
```

If `idx = 3` and `trimmed = 5`, then `idx = -2`. Subsequent writes at
`keys[..., idx ..< idx+1, ...]` will index out-of-bounds (no bounds check).

**Scenario**: Speculative decoding with n-gram verification (Evaluate.swift:1344-1348)
trims cache by rejected draft tokens. If the trim amount exceeds `idx`, corruption occurs.

#### M2. circularWrite() doesn't validate idx (CRITICAL)

**File**: `Libraries/MLXLMCommon/KVCache.swift:565-585`

```swift
private func circularWrite(...) {
    let S = newKeys.dim(2)
    let end = idx + S
    if end <= maxCacheSize {
        cache.0[.ellipsis, idx ..< end, 0...] = newKeys  // idx assumed valid
```

No validation that `0 <= idx < maxCacheSize`. If idx was corrupted by trim(),
this silently writes to out-of-bounds memory.

#### M3. MoE topKIndices used without bounds validation

**File**: `Libraries/MLXLLM/Models/Gemma4.swift:845`

```swift
expertWeights = expertWeights * router.perExpertScale[topKIndices]
```

`topKIndices` comes from `argPartition` — values are indices into the expert dimension.
MLXArray indexing does NOT bounds-check. If `argPartition` produces an index ≥ numExperts
(which shouldn't happen in practice but is not validated), it reads garbage memory.

#### M4. .item() calls without size validation

**File**: `Libraries/MLXLMCommon/Evaluate.swift:1304, 1332, 1440`

```swift
let tokenId = previousY.tokens.item(Int.self)   // Assumes exactly 1 element
```

`.item()` on MLXArray does not validate the array has exactly 1 element.
If `previousY.tokens` is empty or multi-element, behavior is undefined.

#### M5. circularWrite wrap-around can produce negative secondPart

**File**: `Libraries/MLXLMCommon/KVCache.swift:577-578`

```swift
let firstPart = maxCacheSize - idx
let secondPart = S - firstPart   // If firstPart > S, secondPart < 0
```

If `idx` is close to 0 and `S` is small but `maxCacheSize - idx > S`,
we enter the wrap-around branch incorrectly (since `end > maxCacheSize` was true
only if `idx + S > maxCacheSize`). Actually, this specific case can't happen because
we only enter this branch when `end > maxCacheSize`, which means `S > maxCacheSize - idx`,
so `firstPart < S` and `secondPart > 0`. The real risk is if idx was corrupted by M1.

---

## File Index

| File | Lines | Role |
|------|-------|------|
| `Libraries/MLXLLM/Models/Gemma4.swift` | 1292 | Model: config, attention, MoE, PLE, cache, forward |
| `Libraries/MLXLMCommon/KVCache.swift` | ~1900 | StandardKVCache, RotatingKVCache, QuantizedKVCache |
| `Libraries/MLXLMCommon/Evaluate.swift` | ~2500 | TokenIterator, generate loops, sampling |
| `Libraries/MLXLMCommon/AttentionUtils.swift` | 87 | attentionWithCacheUpdate routing |
| `mlx-swift/Source/MLX/MLXFast.swift` | ~300 | Swift wrappers for fast ops |
| `mlx-swift/Source/Cmlx/mlx-c/mlx/c/fast.cpp` | 1011 | C bridge: Swift → C++ |
| `mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp` | ~1150 | C++ primitives: SDPA, RoPE, rmsNorm |
| `mlx-swift/.../metal/scaled_dot_product_attention.cpp` | 826 | Metal dispatch + use_fallback() |
| `mlx-swift/.../metal/kernels/sdpa_vector.h` | 395 | Decode SDPA Metal kernel |
| `mlx-swift/.../metal/kernels/steel/attn/.../steel_attention.h` | 476 | Prefill SDPA Metal kernel |
| `mlx-swift/.../metal/kernels/rms_norm_rope.metal` | 109 | Fused RMSNorm+RoPE Metal kernel |
| `mlx-swift/.../metal/normalization.cpp` | ~490 | Metal dispatch for norms |

---

## 16. E2B/E4B vs 26b/31b — Config Differential

E2B and E4B produce coherent output. 26b and 31b do not (decode-side incoherence).
All four models use the same `Gemma4.swift` implementation — differences are config-driven.

### Config Comparison (from actual config.json files)

| Config Field | E2B (works) | E4B (works) | 26b-A4B (broken) | 31b (broken) |
|---|---|---|---|---|
| **`attention_k_eq_v`** | **`false`** | **`false`** | **`true`** | **`true`** |
| `enable_moe_block` | `false` | `false` | `true` | `false` |
| `num_kv_shared_layers` | 20 | 18 | 0 | 0 |
| `hidden_size_per_layer_input` | 256 (PLE on) | 256 (PLE on) | 0 (PLE off) | 0 (PLE off) |
| `use_double_wide_mlp` | `true` | `false` | `false` | `false` |
| `hidden_size` | 1536 | 2560 | 2816 | 5376 |
| `num_hidden_layers` | 35 | 42 | 30 | 60 |
| `num_attention_heads` | 8 | 8 | 16 | 32 |
| `num_key_value_heads` | 1 | 2 | 8 | 16 |
| **`num_global_key_value_heads`** | **`null`→1** | **`null`→2** | **2** | **4** |
| `sliding_window` | 512 | 512 | 1024 | 1024 |
| `num_experts` | null(0) | null(0) | 128 | null(0) |
| `top_k_experts` | null(0) | null(0) | 8 | null(0) |
| `intermediate_size` | 6144 | 10240 | 2112 | 21504 |
| Quantization | 8-bit | 8-bit | 8-bit | 4-bit |
| Layer pattern | 4s+1f | 5s+1f | 5s+1f | 5s+1f |
| GQA sliding | 8:1 | 4:1 | 2:1 | 2:1 |
| GQA full | 8:1 | 4:1 | **8:1** | **8:1** |

Config.json paths:
- E2B: `~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-8bit/.../config.json`
- E4B: `~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/.../config.json`
- 26b: `~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-8bit/.../config.json`
- 31b: `~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/.../config.json`

### The Single Differentiating Feature

**`attention_k_eq_v = true`** is the only feature that BOTH broken models (26b, 31b) have
and BOTH working models (E2B, E4B) lack.

- MoE can't be the root cause: 31b is broken but has `enable_moe_block = false`
- KV sharing can't help: 26b/31b have `num_kv_shared_layers = 0` (simpler path)
- PLE can't help: 26b/31b have `hidden_size_per_layer_input = 0` (disabled, fewer ops)

### K=V Code Path — Python vs Swift Comparison

**Python** (gemma4_text.py:242-254):
```python
keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
values = keys                                    # same tensor ref
if not self.use_k_eq_v:
    values = self.v_proj(x).reshape(...)

offset = mx.array(cache.offset) ...

keys = self.k_norm(keys)                         # learnable RMSNorm -> new tensor
keys = keys.transpose(0, 2, 1, 3)
keys = self.rope(keys, offset=offset)

values = self.v_norm(values)                     # RMSNormNoScale -> new tensor
values = values.transpose(0, 2, 1, 3)
```

**Swift** (Gemma4.swift:600-633):
```swift
var keys = kProj(x).reshaped(B, L, nKVHeads, -1)
if attentionKEqV { values = keys }               // same ref
else { values = vProj!(x).reshaped(...) }

values = MLXFast.rmsNorm(values, weight: .mlxNone, eps: rmsNormEps)

keys = MLXFast.rmsNormRoPE(keys, weight: kNorm.weight, invFreqs: invFreqs, ...)
keys = keys.transposed(0, 2, 1, 3)
values = values.transposed(0, 2, 1, 3)
```

Both produce identical computational graphs (lazy evaluation makes line order irrelevant):
- `keys`   = rope(transpose(k_norm_learnable(kProj(x))))
- `values` = transpose(rms_norm_no_weight(kProj(x)))

Both derive from the original `kProj(x)` tensor. **Structurally correct.**

### KV Head Selection — Also Matches

Python:
```python
if self.use_k_eq_v and config.num_global_key_value_heads is not None:
    self.n_kv_heads = config.num_global_key_value_heads   # 2 for 26b
else:
    self.n_kv_heads = config.num_key_value_heads          # 8 for 26b
```

Swift:
```swift
if isSliding { self.nKVHeads = config.kvHeads }           // 8 for 26b
else { self.nKVHeads = config.globalKvHeads }              // 2 for 26b
```

Same results for all four models. Full-attention K=V layers get `globalKvHeads`.

### Features Only Exercised by Working Models (can't cause 26b/31b bug)

- **KV sharing** (`num_kv_shared_layers > 0`) — only E2B/E4B
- **PLE** (`hidden_size_per_layer_input = 256`) — only E2B/E4B
- **Double-wide MLP** — only E2B

### Remaining Hypotheses

1. **Weight loading for K=V**: When `attentionKEqV=true`, no `v_proj` weights exist.
   Does the weight loader correctly skip, or does it zero-init phantom weights?

2. **Quantization + K=V interaction**: `k_proj` output serves as both K and V.
   Quantization error affects both attention scores AND value accumulation,
   potentially doubling its impact vs separate K/V projections.

3. **head_dim=512 precision with scale=1.0**: Accumulating 512 multiply-adds
   in float32 from bfloat16 Q/K with QK-norm scores in [-1,1] may amplify
   subtle numerical errors over 30-60 layers.

4. **Upstream issue**: mlx-community weight conversion may be incorrect for K=V models.

### Recommended Diagnostic Tests

```bash
# 1. Disable fused norm+rope → isolate Metal kernel
GEMMA4_FUSED_NORM_ROPE=0

# 2. Force SDPA fallback for head_dim=512 → isolate Metal SDPA
MLX_SDPA_NO_BD512=1

# 3. DECISIVE: Run same model through Python mlx-lm
#    Python coherent + Swift incoherent → Swift-specific bug
#    Both incoherent → upstream weight/quant issue
python3 -m mlx_lm.generate --model mlx-community/gemma-4-26b-a4b-it-8bit \
    --prompt "Explain quantum computing in simple terms"
```

---

## 17. Git Timeline — Regression Window

**Working state**: `6e47f6b` (Apr 7, 2026) — "fix issues with Gemma model - proportional RoPE and reusable KV cache bugs"

**Regression window**: April 8, 2026 — multiple fused kernel changes:

| Commit | Date | Change | Affects |
|--------|------|--------|---------|
| `6796226` | Apr 8 | Replace manual `rmsNormNoScale()` with `MLXFast.rmsNorm(weight: .mlxNone)` | V norm in all Gemma4 |
| `b4087a4` | Apr 8 | Cache RotatingKVCache `peek()` results | KV shared layers |
| `c72aca3` | Apr 8 | Add fused RMSNorm+RoPE Metal kernel | New kernel |
| `524feba` | Apr 8 | Use `MLXFast.rmsNormRoPE` in Gemma4 attention | Q/K norm+RoPE in all Gemma4 |
| `d786576` | Apr 8-9 | `MLXFast.rmsNormResidual` fused norm+add | Post-attn/FFN residual in all Gemma4 |

**Key structural change**: Before April 8, Q/K/V were transposed BEFORE norm/RoPE:
```
WORKING:  Q = transpose(qProj(x)) → qNorm → RoPE       (norm on [B,H,L,D])
CURRENT:  Q = qProj(x)            → rmsNormRoPE → transpose  (norm on [B,L,H,D])
```
Mathematically equivalent (RMSNorm normalizes along last dim regardless), but different
Metal kernel dispatch and thread organization.

**Update (Apr 14)**: E2B/E4B also show repetition — issue affects ALL Gemma4 models,
not just K=V models. Fused kernels applied to ALL variants are prime suspects.

### A/B Test Results (E2B 8-bit)

| Config | Output | tok/s |
|--------|--------|-------|
| `GEMMA4_FUSED_NORM_ROPE=0` (disabled) | "Hello! I am Gemma" (coherent) | 80.9 |
| Fused kernel enabled (default) | "Hello! I am Gemma" (coherent) | 83.4 |

E2B produces coherent start in both cases. Repetition may appear later in generation.
26b-A4B tests pending.
