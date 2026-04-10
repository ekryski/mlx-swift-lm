# Performance Optimization Plan: Inference Pipeline, Batching, Speculation, ANE

## Context

Previous sessions achieved major wins on MoE inference (6.3x prefill, +19% decode GPT-OSS) and implemented full Gemma 4 support (all 4 variants validated). This plan covers the comprehensive performance optimization roadmap, ordered by priority: core inference pipeline first, then batching/speculation, then ANE offloading.

### Quick Test Protocol

After each optimization attempt, run this quick regression/improvement check:

```bash
benchmark.sh --model <model> --quant 4bit --kv none,turbo4v2 --ppl --kld --think --method summarization --context 1024,4096
```

Primary models for testing:
- **Gemma 4 E2B** (pure attention, dense, PLE, KV sharing)
- **Gemma 4 26B-A4B** (pure attention, MoE)
- **Qwen3.5-35B-A3B** (GatedDeltaNet hybrid, MoE)
- **GPT-OSS-20B** (pure attention, MoE)

After all optimizations complete, run comprehensive benchmarks across all contexts, model quants, and KV cache strategies with and without PPL/KLD/thinking flags.

---

## Research Findings

### vLLM Paged Attention vs Flat Super-Sequence

**vLLM Paged Attention**: Block tables mapping logical→physical memory pages, 16 tokens/block. Custom CUDA kernels with online softmax across non-contiguous blocks. 7-13% instruction overhead, up to 28% slower prefill, but 2-4x throughput at 32+ concurrent sequences. vllm-metal exists but benchmarks slower than MLX for single-user inference.

**llama.cpp Continuous Batching**: Flat unified cache with per-cell bitset sequence tracking. No indirection — simpler access, better cache locality.

**For our use case (1-4 concurrent on Apple Silicon)**: Flat super-sequence wins. Paging overhead not justified below 8 sequences.

### TurboQuant + Batching Synergy

TurboQuant Metal kernels already support batch>1 via flattening: `totalQ = B * nQHeads * L`. Memory math: TurboQuant 4-bit compresses KV 13-14x. At 4K context: FP16 = 4.2GB/seq, TurboQuant = 0.3GB/seq. **Batch=4 feasible where batch=2 was the max with FP16.**

### ANE (Apple Neural Engine)

[Orion](https://github.com/mechramc/Orion) ([paper](https://arxiv.org/abs/2603.06728)) bypasses CoreML for direct ANE access. 170+ tok/s on GPT-2 124M, ~19 TFLOPS fp16, 32MB SRAM. Softmax 33.8x faster than CPU. Dispatch overhead ~0.095ms but round-trip ~2.3ms. ANE runs parallel to GPU — useful for offloading operations while GPU handles attention.

### Warp Decode for MoE (from [Cursor blog](https://cursor.com/blog/warp-decode))

Flips MoE parallelism: each SIMD group handles one OUTPUT NEURON across all routed experts (vs standard expert-centric approach). Eliminates 5 of 8 pipeline stages that exist purely for data management. Metal's `simd_shuffle_xor` maps to CUDA's `__shfl_xor_sync`. **Est: 1.3-1.6x decode for MoE models.**

---

## Phase 1: MLX Kernel Options — Empirical Testing

Kernel options B/D/E were analyzed theoretically but never benchmarked. Get empirical data.

### 1a. Option A — Already Done ✅

gather_qmm_rhs threshold B>=4. Benchmarked: +6-14% prefill, +5-11% decode. Keeping.

### 1b. Option B — Fused Activation via compile()

Test if MLX `compile()` fuses split + silu + multiply between gatherQuantizedMM calls.

**File**: `Libraries/MLXLMCommon/SwitchLayers.swift`

```swift
let fusedActivation: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { gateUp in
    let parts = MLX.split(gateUp, parts: 2, axis: -1)
    return silu(parts[0]) * parts[1]
}
```

### 1c. Option D — Sort Threshold A/B

Empirically measure L2 locality benefit of sorting:

```bash
MOE_SORT_THRESHOLD=128 benchmark.sh --model qwen35-35b-a3b --quant 4bit --kv none,turbo4v2 --ppl --method summarization --context 1024,4096
MOE_SORT_THRESHOLD=0   benchmark.sh --model qwen35-35b-a3b --quant 4bit --kv none,turbo4v2 --ppl --method summarization --context 1024,4096
MOE_SORT_THRESHOLD=32  benchmark.sh --model qwen35-35b-a3b --quant 4bit --kv none,turbo4v2 --ppl --method summarization --context 1024,4096
```

### 1d. Option E — Int8 Matmul Microbenchmark

```python
import mlx.core as mx
a = mx.random.normal([1, 2048]).astype(mx.float16)
b = mx.random.normal([2048, 512]).astype(mx.float16)
# Compare throughput
```

### 1e. Document Results

`benchmarks/notes/mlx-kernel-options-empirical-2026-04-08.md`

---

## Phase 2: TurboQuant-Plus Optimizations (from Tom's fork)

Tom's [`feature/turboquant-plus`](https://github.com/TheTom/mlx/tree/feature/turboquant-plus) branch has optimizations we haven't adopted:

### 2a. Delegated KVCache + Pre-allocated Buffers

Current TurboQuant: 72+ allocations per decode step. Tom's approach: delegate FP16 storage to internal KVCache with pre-allocated buffers, batch recompress every 64 tokens. **Result: 0.99x baseline decode (nearly zero compression overhead).**

**Action**: Review delegated KVCache pattern. Adopt pre-allocation strategy.

### 2b. Async CPU Encoding

`turbo_encode_cpu()` dispatches encoding to CPU stream parallel with GPU SDPA. **Hides 75-88% of encode latency.** Bit-exact output.

**Action**: Investigate MLX concurrent CPU/GPU streams. Move encode off GPU path.

### 2c. Two-Pass TurboFlash Verification

At 32K context: "single 3.504ms → two-pass 1.758ms (1.99x)". Verify our TurboFlash has this.

### 2d. `split_logsumexp` Check

Single commit from Awni Hannun. Check if merged to upstream MLX main. Low priority — only benefits standard (non-TurboQuant) SDPA for `--kv none` long-context paths. Our TurboFlash already has its own two-pass.

### 2e. Quantized LM Head for FP16/BF16 Models

**Empirical finding (Phase 1d):** Gemma4's 262K vocab LM head projection is 20.8x faster with Int4 quantization at decode (1 token), and still 3x faster at 32 tokens. Even at 128 tokens, quantized is 1.6x faster.

**Action:** In `sanitize()`, quantize `lm_head.weight` to Int4 even for FP16/BF16 models. Use `quantizedMatmul` in `callAsFunction` for the LM head path. Benefits any model with large vocabulary (Gemma4 262K, Qwen3.5 152K).

**Expected:** Significant per-token decode latency reduction — the LM head is one of the largest single ops per token.

### 2f. Benchmark affine8 KV Cache

Int8 `quantizedMatmul` is 17-27% faster than FP16 at decode batch sizes (1-32 tokens), with nearly zero dequant overhead. Int8 KV cache should offer much better PPL than 4-bit with no decode speed penalty.

**Action:** Run full benchmark suite with `--kv affine8` across all 4 primary models. Compare PPL and speed vs affine4 and no-quant baselines.

### 2g. TurboQuant Int8 Variant Investigation

Investigate asymmetric `turbo8v4` (8-bit keys, 4-bit values). Rationale: keys need higher precision for attention score accuracy, values can tolerate more compression. Int8 dequant is nearly free at decode sizes per 1d findings.

**Action:** Prototype Int8 TurboQuant encoding. Benchmark quality (PPL/KLD) vs turbo4v2.

---

## Phase 3: Gemma 4 & Cross-Model Inference Optimizations

Implements items from `gemma4-performance-optimization-plan.md` and `gemma4-26b-a4b-optimization-plan.md`, plus cross-model improvements.

### Tier 1 — Quick Wins

#### 3a. Increase Prefill Chunk Size (Pure Attention Models)

Gemma 4 is pure attention — no GatedDeltaNet sequential bottleneck. Increase `prefillStepSize` from 512/1024 to 4096.

**Files**: `Libraries/MLXLLM/Models/Gemma4.swift` (override `prepare()`), benchmark GenerateParameters
**Models**: Gemma 4 (all), GPT-OSS
**Expected**: 15-25% TTFT reduction

#### 3b. FusedGateUpSwitchGLU for Gemma 4 26B MoE

Same proven pattern as GPT-OSS: fuse gate+up weights in sanitize, use FusedGateUpSwitchGLU with `geluApproximate` activation.

**Files**: `Libraries/MLXLLM/Models/Gemma4.swift` (sanitize + switch to FusedGateUpSwitchGLU)
**Expected**: 10-15% decode for 26B (saves 30 Metal dispatches)

#### 3c. v_norm → MLXFast.rmsNorm Fusion

Replace manual `rmsNormNoScale()` (3 dispatches: square, mean, rsqrt*mul) with `MLXFast.rmsNorm(x, weight: ones, eps:)`.

**Files**: `Libraries/MLXLLM/Models/Gemma4.swift`, `Libraries/MLXVLM/Models/Gemma4.swift`
**Expected**: 2-4% decode

#### 3d. Split Fused/Unfused GDN by T (Qwen3.5)

Fused GDN kernel gains +4-6% decode but loses -6-12% prefill due to register pressure. Use fused for T=1 (decode), original for T>1 (prefill).

**Files**: `Libraries/MLXLLM/Models/GatedDelta.swift`
**Expected**: Recover 6-12% prefill regression while keeping decode gain

### Tier 2 — Medium Effort

#### 3e. RotatingKVCache peek() Caching

Cache `temporalOrder()` result across multiple `peek()` calls per token. Invalidate on `update()`.

**Files**: `Libraries/MLXLMCommon/KVCache.swift`
**Models**: E2B (20 shared layers), E4B (18 shared layers)
**Expected**: 3-5% decode for KV-shared models

#### 3f. Symbolic Sliding Window Mask

Add `.slidingWindow(size:)` case to `ScaledDotProductAttentionMaskMode`. Currently materializes full N×N boolean array (16M elements at 4K context × 28 sliding layers).

**Files**: MLX framework (`ScaledDotProductAttention`), `Libraries/MLXLMCommon/KVCache.swift`
**Expected**: ~15% overall (reduced memory bandwidth)

#### 3g. Profile-Guided Expert Reordering

Collect co-selection matrix over calibration corpus. Spectral ordering (Fiedler vector) produces cache-optimal expert permutation. Eliminates sort/no-sort tradeoff.

**Files**: New `ExpertCalibration.swift`, `Libraries/MLXLMCommon/SwitchLayers.swift`
**Models**: Qwen3.5-35B MoE, Gemma 4 26B, GPT-OSS
**Expected**: +25% at T=128 (recover sort overhead), +0-5% at larger T

#### 3h. Cache PLE Embeddings (E2B/E4B)

`embedTokensPerLayer(inputs)` is called every decode step. The per-layer embedding lookup is cheap but `perLayerModelProjection(h)` is a full matmul. Consider splitting per-layer or deferring.

**Files**: `Libraries/MLXLLM/Models/Gemma4.swift`
**Expected**: 5-10% decode for E2B/E4B

### Tier 3 — Custom Metal Kernel Work

#### 3i. Steel Attention for head_dim=256/512

MLX Steel kernels only support head_dim=64,80,128. Add instantiations for 256/512, tune tile sizes.

**Files**: MLX framework (`steel_attention.h`, `scaled_dot_product_attention.cpp`)
**Expected**: 15-25% attention speedup on long sequences

#### 3j. Fused Norm+RoPE Kernel

Combine RMSNorm + RoPE into single dispatch. Saves 84+ dispatches across 42 layers.

**Files**: New Metal kernel in MLX framework
**Expected**: 8-12% decode

#### 3k. Circular KV Cache

Replace physical array reordering with logical circular indexing. Eliminates O(maxSize) concat per token.

**Files**: `Libraries/MLXLMCommon/KVCache.swift`, attention kernel masking
**Expected**: 30% KV cache update latency reduction

#### 3l. Warp-Parallel MoE Decode Kernel

Flip MoE parallelism: each SIMD group handles one output neuron across all routed experts. Uses Metal `simd_shuffle_xor` for butterfly reduction. Two fused kernels replace entire SwitchGLU path.

**Files**: New Metal kernels, `Libraries/MLXLMCommon/SwitchLayers.swift`
**Models**: All MoE (Qwen3.5-35B, Gemma 4 26B, GPT-OSS)
**Expected**: 1.3-1.6x decode improvement

---

## Phase 4: True Batched GPU Inference

### Approach: Flat Super-Sequence with BatchKVCache

Stack B sequences along batch dim 0, run ONE forward pass. Pure attention models only (Gemma 4, GPT-OSS). Skip GatedDeltaNet models.

### 4a. BatchKVCache Wrapper

**File**: `Libraries/MLXLMCommon/KVCache.swift`

Wraps B independent KVCache instances. Stacks K/V for batched attention, unstacks results. Works transparently because all models carry B dimension through forward pass. TurboQuant flattens batch into `totalQ = B * nQHeads * L`.

### 4b. BatchTokenIterator

**File**: `Libraries/MLXLMCommon/Evaluate.swift`

Manages B sequences simultaneously. Single forward pass returns `[B, vocab]` logits. Sample B tokens independently.

### 4c. Batch Sampling

Extend sampler for `[B, vocab]` logits with per-sequence penalties.

### 4d. Benchmark

```bash
benchmark.sh --model gemma4-e2b --quant 4bit --kv none,turbo4v2 --ppl --method summarization --context 1024,4096 --batch 2
benchmark.sh --model gemma4-e2b --quant 4bit --kv none,turbo4v2 --ppl --method summarization --context 1024,4096 --batch 4
```

**Expected**: 1.5-2.5x aggregate throughput at batch=2, 2-3.5x at batch=4.

---

## Phase 5: Speculative Decoding Improvements

### 5a. Hash-Based N-gram Lookup

**File**: `Libraries/MLXLMCommon/Evaluate.swift`

Replace O(n*m) linear scan with O(1) hash table. FNV-1a hash on last N token IDs. Incrementally built during generation.

### 5b. Multi-Size N-gram (2-5)

Hash tables for sizes 2, 3, 4, 5. Check longest match first.

### 5c. Dynamic Draft Length

Adapt based on rolling acceptance rate (8 tokens at >70%, 3 at 20-40%, disable at <20%).

### 5d. Fix Acceptance Rate Metric

Divide by actual proposals, not slot capacity.

### 5e. Expose Speculation Metrics

Add `ngramProposed`, `ngramAccepted`, `ngramAcceptanceRate` to `GenerateCompletionInfo`.

---

## Phase 6: ANE Offloading Investigation

### 6a. Investigate Orion Integration

[Orion](https://github.com/mechramc/Orion) bypasses CoreML via `_ANEClient`/`_ANECompiler`. [ensue-network.ai](https://ensue-network.ai/lab/ane) achieved 6.31x faster than CoreML on DistilBERT (M5 Max) by fusing all 6 layers + classifier into a single ANE dispatch with weight folding and sigmoid GELU. See `benchmarks/notes/ane-kernel-example-distilbert.md` for full reference kernel.

Understand data format requirements (IOSurface, fp16/bf16/int8, [1,C,1,S] layout) and whether ANE can pipeline with GPU.

### 6b. Prototype: LM Head on ANE

Vocabulary projection is one of the largest ops. ANE softmax is 33.8x faster than CPU. Pipeline: GPU hidden state → ANE logits + softmax → CPU sample.

### 6c. Prototype: PLE on ANE (E2B/E4B)

`perLayerModelProjection(h)` could run on ANE parallel with GPU attention on previous layer.

### 6d. Risk Assessment

- Private APIs: `_ANEClient` may break between macOS versions
- 32K channel limit: vocab projections >32K fall to CPU
- Supported dtypes: **fp32, bf16, fp16, int8** (confirmed by Apple engineer; Orion paper only documented fp16)
- SRAM cliff: 30% perf drop when working set >32MB

### 6e. Decision Criteria

Worth it if parallel offloading achieves >5% per-token latency reduction with <1 week effort.

---

## Phase 7: Nemotron Cascade 2 Benchmarks

Data collection only — no implementation.

```bash
benchmark.sh --model nemotron-30b-a3b --quant 4bit --kv none,turbo4v2 --ppl --method summarization --quick
```

---

---

## Implementation Status (as of 2026-04-09)

### Phase 1 — DONE

| Item | Status | Result |
|------|--------|--------|
| 1a | ✅ Done (prior) | +6-14% prefill, +5-11% decode |
| 1b | ✅ Done | compile() crashes/neutral. Custom Metal kernel also neutral — MLX native ops already efficient for simple fusions. Code retained, disabled by default. |
| 1c | Deferred | Env var ready (`MOE_SORT_THRESHOLD`), awaiting dedicated sweep |
| 1d | ✅ Done | Crossover at ~256-512 tokens. Int4 LM head 20.8x faster. See `mlx-kernel-options-empirical-2026-04-08.md` |
| 1e | ✅ Done | Documented in `benchmarks/notes/` |

### Phase 2 — DONE

| Item | Status | Result |
|------|--------|--------|
| 2a | ✅ Done | Pre-alloc step 256→1024 (neutral). Adaptive batch recompress (64K TTFT: -22.6%). MLX lazy eval already handles allocation amortization — Tom's delegated KVCache pattern unnecessary in Swift. |
| 2b | ✅ Assessed | Not applicable — solves Python-specific synchronous dispatch problem. In MLX Swift, lazy eval batches GPU ops naturally. |
| 2c | ✅ Verified | Two-pass TurboFlash already implemented (ARCHITECTURE.md) |
| 2d | Low priority | Not in mlx-swift. Only benefits `--kv none` standard SDPA. |
| 2e | ✅ Assessed | 4-bit models already have quantized LM heads. Only helps bf16 models. |
| 2f | ✅ Done | affine8 benchmarked. Speed-neutral for pure attention, -15% for Qwen GDN. Better PPL than turbo4v2. See `affine8-kv-cache-evaluation-2026-04-08.md` |
| 2g | ✅ Done | turbo8v4 and turbo8v2 available. turbo8v4 closes 82% of quality gap vs FP16 at 16K with 2.6x compression. |

### Phase 3 Tier 1 — DONE

| Item | Status | Result |
|------|--------|--------|
| 3a | ✅ Done | Prefill 2048 for pure attention. 4096 = -39% regression. |
| 3b | ✅ Done | FusedGateUpSwitchGLU for Gemma4 26B (LLM+VLM). +5-6% decode. |
| 3c | ✅ Done | v_norm → MLXFast.rmsNorm with mlxNone weight. 3→1 dispatch. |
| 3d | ✅ Already done | Qwen35.swift:268-308. Fused at T=1, standard at T>1. |

### Phase 3 Tier 2 — Partial

| Item | Status | Result |
|------|--------|--------|
| 3e | ✅ Done | peek() caching in RotatingKVCache. |
| 3f | Pending | Requires mlx-swift framework changes. |
| 3g | Pending | Self-contained in mlx-swift-lm. |
| 3h | Deprioritized | Projection depends on h, can't cache. <2% expected. |

### Phase 3 Tier 3 — Partial

| Item | Status | Result |
|------|--------|--------|
| 3i | ✅ Investigated | Steel BD=256: registers don't spill (208KB/core) but matmul fallback is faster on M1 Max at all tested contexts. Kernels compiled but dispatch disabled. |
| 3j | ✅ Done | **Framework-level** fused RMSNorm+RoPE in mlx-swift. +1-3% decode, -7% TTFT. JIT version was -17% — framework compilation was critical. |
| 3k | Pending | Pure Swift, no framework changes needed. |
| 3l | Pending | Most complex Metal kernel work. |

### Framework Kernel Migration (NEW — not in original plan)

Moved critical kernels from JIT (MLXFast.metalKernel) to framework-level (compiled into metallib):

| Kernel | Status | Impact |
|--------|--------|--------|
| `rms_norm_rope.metal` | ✅ Shipped | +1-3% decode. Full C++ primitive + C bridge + Swift binding. |
| `rms_norm_qgemv.metal` | ✅ Built & tested | Fused RMSNorm+4-bit GEMV. Correct but -3-5% regression — weight reads dominate, norm intermediate is negligible. Disabled. |
| `turbo_quant.metal` | ✅ Compiled into metallib | Score, encode, encode_wht, pass2, pass2_fused_rot, value kernels. |
| `turbo_flash.metal` | ✅ Compiled into metallib | Pass 1 standard + causal. |
| `gated_delta.metal` | ✅ Compiled into metallib | Standard + fused GDN with function constant mask. |
| TurboQuant buffer-arg dispatch | ✅ Shipped | Moved token_count/num_blocks/repeat_count from template to buffer args. Eliminates per-token Metal pipeline recompilation. +2-5% decode. |

### Critical Bug Fix

| Fix | Impact |
|-----|--------|
| `trackPerplexity` default `true` → `false` | Was computing full-vocab softmax (262K float32) on EVERY decode token. Python mlx-lm doesn't do this by default. Likely a major contributor to the speed gap vs Python. |

### Session 2026-04-08/09: Incept5 Comparison & Forward Pass Deep Dive

Ran controlled A/B benchmark against Python mlx-vlm/mlx-lm on same hardware (M1 Max 64GB).

**Incept5 benchmark comparison** — see `incept5-benchmark-comparison.md`:
- Prompt content (repetitive vs Gatsby) has zero effect on throughput (~3% from max_tokens only)
- Python mlx-lm: 99 tok/s decode, 3050 prefill at 4k. Our code: 80→88 tok/s decode after fixes.
- Gap is NOT methodology — it's implementation differences

**Gemma4 forward pass fixes** (committed):
- Norm-before-transpose ordering (RMSNorm on [B,L,heads,dim] before transpose — matches Python)
- Compiled GEGLU and logit softcapping via `compile(shapeless: true)`
- Compiled norm+residual chains (saves 1 dispatch per site × 35 layers)
- Scalar embedding scale (removed bfloat16→float32 dtype conversion)
- Pre-sliced per-layer inputs (35 views upfront instead of re-slicing in loop)

**Metal System Trace analysis** — see `metal-trace-gemma4-e2b-2026-04-08.csv`:
- 18.8 encoders/token, 4.47ms GPU, 13.6ms wall → 33% GPU utilization
- 65% of wall time is CPU dispatch overhead between encoders
- GPU compute is NOT the bottleneck — scheduling overhead is

**MLX internals investigation**:
- `asyncEval` blocks in `eval_impl` when GPU pipeline is full (backpressure at `n_active_tasks > 10`)
- Memory backoff (`get_active_memory() > get_memory_limit()`) does NOT fire — `block_limit_` is ~60GB
- `WiredMemoryTicket` only sets `wired_limit_` (residency set), NOT `block_limit_` (memory limit)
- `max_ops_per_buffer` tuning: 50→100 for Max/Ultra gives +5-8% decode (fewer command buffer commits)

**Buffer cache bloat investigation** — see `mlx-memory-over-allocation-root-cause-analysis.md`:
- Post-generation cache: 5.5GB (ours) vs 66MB (Python) — 80x difference
- Root cause: `clearCache()` after prefill fired BEFORE GPU completion handlers → buffers still "active"
- Fix: `eval()` + `synchronize()` + `clearCache()` after prefill → cache dropped to ~400MB
- Remaining gap: Python's `generation_stream` releases buffers via separate command queue lifecycle
- All prefill loops (LLMModel, Gemma4, Qwen35, GPTOSS) now have `clearCache()` matching Python pattern
- Prefill loops now process ALL tokens except the last 1 through chunked eval (matching Python exactly)

**Steel SDPA for head_dim=256/512** (committed to mlx-swift fork):
- BD=256 Steel kernel exists but was disabled in dispatch. Enabled for bfloat16/float16.
- BD=512 Steel kernel instantiated (BQ=8, BK=8, WM=1, WN=1) — 6% occupancy on M1 Max, viable on M5 Max
- BD=512 vector (decode) kernel also instantiated
- Float32 variants exceed 32KB threadgroup memory limit — only half-precision works
- **Critical finding**: 4-bit quantized models dequantize to bfloat16 (scales dtype), NOT float32
  - BUT: dtype leak found — layer 0 attention gets bfloat16, layer 1+ gets float32
  - Something in the decoder layer promotes bfloat16→float32 between layers
  - **This is the root cause of Steel SDPA not activating AND the 2GB memory gap**
  - Investigation ongoing — likely a scalar multiply, compiled function eps, or PLE computation

**MLX Swift framework fixes** (committed to mlx-swift fork):
- `Stream.setAsDefault()` — wraps `mlx_set_default_stream` C API
- `Stream.withStream()` — use existing stream as default (vs `withNewDefaultStream` which creates new)
- `StreamOrDevice.stream(_:)` bug fix — was ignoring the stream parameter
- Lazy `CommandEncoder` init in `get_command_encoder()` — Swift Tasks migrate between threads,
  causing "no Stream(gpu, 0)" crash when thread-local encoder map is empty. Auto-creates encoder.
- JIT source regeneration via `tools/update-mlx.sh` — fixes `affine_qmm_t_splitk` undeclared.
  **WARNING**: full regeneration causes 40% regression — only update `quantized.cpp`, keep other files.
- `build-metallib.sh` added to benchmark.sh pipeline (rebuilds after `swift build --build-tests`)

**Generation stream — FAILED** (all 3 approaches):
- `Stream.withNewDefaultStream`: 6x regression from cross-stream sync with model weights
- `Stream.setAsDefault()`: "no Stream(gpu, 0)" crash — replaces default, breaks existing references
- Separate stream without wrapping: no benefit without wrapping all ops
- Python succeeds because `mx.stream()` sets default at C level; Swift TaskLocal causes deps

### Remaining Items from This Session

| Item | Status | Notes |
|------|--------|-------|
| Fix bfloat16→float32 dtype leak in decoder layer | 🔴 Blocking | Root cause of Steel SDPA not activating AND 2GB peak memory gap. Something promotes dtype between layer 0 and layer 1. |
| Test Steel SDPA with bfloat16 pipeline (after dtype fix) | Pending | BD=256 prefill should eliminate L×L score materialization. BD=512 decode should use vector kernel. |
| M-series architecture notes (M5 Max Neural Accelerator) | Pending | M5 Max: 8.2 MiB register file, 614 GB/s bandwidth, dedicated matrix multiply blocks. BD=512 viable. |
| Reduce `max_ops_per_buffer` regression at large values | Done | 100 is sweet spot for M1 Max. 300+ causes memory bloat. Changed in device.cpp. |
| `update-mlx.sh` safe procedure | Documented | Only regenerate `quantized.cpp`. Other files have our optimizations baked in. |

### Combined Benchmark Results (Gemma4 E2B, Phase 2 baseline → current)

| Config | Metric | Phase 2 Baseline | Current | Delta |
|--------|--------|-----------------|---------|-------|
| 1K none | Decode | 80.1 | 82.7 | **+3.2%** |
| 4K none | Decode | 78.9 | 80.6 | **+2.2%** |
| 16K none | Decode | 68.6 | 72.0 | **+5.0%** |
| 1K none | TTFT | 997ms | 925ms | **-7.2%** |
| 1K turbo | Decode | 80.7 | 81.9 | +1.5% |
| 4K turbo | Decode | 78.6 | 81.4 | **+3.6%** |
| 32K turbo | Decode | 59.2 | 61.2 | **+3.4%** |
| 64K turbo | TTFT | 49702ms | 38464ms | **-22.6%** |
| 128K turbo | Decode | 30.9 | 37.3 | **+20.7%** |

---

## Phase 8: CPU Pipeline Optimization (NEW)

### Problem Statement

Profiling reveals that the GPU sits idle between tokens while the CPU builds the lazy computation graph. Each decode token constructs ~500 operation nodes (30 layers × ~17 ops) through the Swift → C bridge, taking an estimated 8-12ms. With the GPU only needing ~2.5-4ms for actual computation, **the CPU graph building is the primary bottleneck, not GPU compute or memory bandwidth.**

See `benchmarks/notes/inference-architecture-m1-max.md` for full hardware analysis.

### 8a. Compiled Decode Step via MLX compile()

MLX's `compile()` traces a function once, optimizes the graph (node deduplication, kernel fusion), and caches the compiled tape. Subsequent calls replay the tape without rebuilding the graph.

**Challenge**: compile() requires fixed tensor shapes between calls. The KV cache grows each token (seq_len increases by 1), which triggers recompilation.

**Solution**: Pre-allocate KV cache to maximum length at init. Use an offset parameter to track the "active" portion. Since the cache tensor shape never changes, compile() can cache the graph permanently.

**Requirements**:
- KVCacheSimple: pre-allocate to `maxSize` at init instead of growing incrementally
- RotatingKVCache: already has fixed `maxSize`, should work as-is
- Pass `cache.offset` as an MLXArray input (not a Swift Int) so it's part of the traced graph
- Wrap the model forward + sampling into a single compiled function

**Expected**: Eliminate 8-12ms graph building → ~2ms tape replay = 6-10ms saved per token = **60-80% decode speedup**

### 8b. Double-Buffer Async Pipeline

Currently: CPU builds graph → asyncEval → .item() blocks → return → build next graph
Ideal: CPU builds graph N+1 while GPU evaluates graph N

**The `.item()` sync on line 1435 of Evaluate.swift blocks the pipeline.** The token ID is needed to return to the caller, but the graph for the NEXT token could start building in parallel.

**Solution**: Restructure the decode loop to:
1. Start building token N+1's graph immediately after submitting token N
2. Only sync (.item()) when the caller actually needs the token value
3. Use a producer-consumer pattern where graph building runs ahead of GPU evaluation

### 8c. Reduce Swift→C Bridge Overhead

Each MLX operation (matmul, add, reshape, etc.) crosses the Swift → C → C++ boundary. With ~500 ops per token, even 10μs per crossing = 5ms overhead.

**Approaches**:
- Batch multiple MLX operations into single C calls where possible
- Use compile() to replace per-op bridge calls with a single compiled function call
- Profile the actual per-operation bridge overhead to quantify the gap

### 8d. Operation Count Reduction

Further fusion opportunities to reduce the ~500 operations per token:

| Fusion | Saves | Status |
|--------|-------|--------|
| RMSNorm + RoPE → single dispatch | 60 ops (2 per layer × 30) | ✅ Done |
| Norm + residual → compiledNormResidual | 60 ops (2 per layer × 30) | ✅ Already done |
| GEGLU (gelu + mul) → single op | 30 ops | ✅ Already done |
| Q+K+V projections → single batched matmul | 60 ops (2 saved per layer × 30) | Proposed |
| Full attention block → single compiled subgraph | 150+ ops | Proposed (via compile()) |

---

## Summary by File

| File | Phases | Changes |
|------|--------|---------|
| `SwitchLayers.swift` | 1b, 3b, 3g, 3l | compile() test, fused Metal kernel, FusedGateUp, expert reorder, warp kernel |
| `KVCache.swift` | 2a, 3e, 3k, 4a | Pre-allocation, peek() caching, circular cache, BatchKVCache |
| `Gemma4.swift` (LLM+VLM) | 3a-c, 3h, 3j | Prefill chunk, fused MoE, v_norm, PLE cache, fused NormRoPE |
| `GatedDelta.swift` | 3d | Split fused/unfused by T |
| `TurboQuantKernels.swift` | 2a | Adaptive batch recompress, buffer-arg dispatch |
| `Evaluate.swift` | 4b, 5a-e, 8a-b | BatchTokenIterator, hashed n-gram, metrics, compiled decode, async pipeline |
| `InferenceBenchmark.swift` | 4 | True batched benchmark |
| MLX framework (mlx-swift) | 3i, 3j, 8a | Steel BD=256, fused NormRoPE primitive, fused QGEMV, TurboQuant/GDN Metal kernels, compile() decode |
| `benchmarks/notes/` | 1e, 2f, 8 | Empirical results, architecture doc |

## Verification

After each optimization:
```bash
benchmark.sh --model <model> --quant 4bit --kv none,turbo4v2 --ppl --kld --think --method summarization --context 1024,4096,16384
```

Primary test models: Gemma 4 E2B, Gemma 4 26B-A4B, Qwen3.5-35B-A3B, GPT-OSS-20B.

Final comprehensive pass: all contexts, all quants, all KV strategies, with and without PPL/KLD/thinking.
