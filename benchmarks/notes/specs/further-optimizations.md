# Further Optimization Opportunities

**Date**: 2026-04-02

## Question 1: Why is GPU Peak so high even with compression?

### The Answer

At 4K context with 8-bit Qwen3.5-2B, GPU Peak is 3.95GB. Breakdown:

| Component | Size | % of Peak |
|-----------|:----:|:---------:|
| Model weights (8-bit) | 1.86 GB | 47% |
| **Prefill attention scores** | **~0.81 GB** | **21%** |
| Layer intermediates (28 layers) | ~1.0 GB | 25% |
| KV cache (turbo compressed) | ~0.17 GB | 4% |
| Misc (state, logits, etc) | ~0.11 GB | 3% |

**The KV cache is only 4% of peak memory.** Further KV compression has negligible peak impact.

The dominant transient cost is the **prefill attention score matrix**: `[1, 24 heads, 2048 prefill_chunk, 4087 seq_len]` = 201M float32 elements = 0.81 GB. MLX does NOT use flash attention — it materializes the full Q×K score matrix.

### How to Reduce Peak

**Reduce prefillStepSize** (currently 2048 in benchmarks):
- 2048 → 512: attention matrix drops from 0.81GB → 0.20GB, saving **~0.6 GB**
- Estimated peak: 3.95GB → ~3.3GB
- Trade-off: 4x more prefill passes, slightly slower TTFT

**Implement chunked/flash attention for prefill**: MLX's Steel Attention kernel materializes the full score matrix. A flash attention implementation would use O(1) memory for scores by computing softmax in streaming blocks. This would require contributing upstream to mlx-swift or writing a custom prefill kernel.

**Smaller eval granularity**: MLX's lazy evaluation keeps the entire computation graph alive through all 28 layers until `eval()` is called. Inserting `eval()` between layer groups (e.g., every 7 layers) would free intermediates sooner, reducing the ~1.0 GB layer intermediate overhead.

## Question 2: Can we create more fused kernels?

### Current Kernel Architecture

| Operation | Current Impl | Dispatches | Opportunity |
|-----------|-------------|:----------:|-------------|
| Encode (norm+rotate+quantize+pack+normcorrect) | **Fused Metal** | 1 | Already optimal |
| Score (Q×K from packed) | **Fused Metal** | 1 | Could add Int8 |
| Softmax | MLX op | 1 | Hard to fuse across kernel boundary |
| Value (Attn×V from packed) | **Fused Metal** | 1 | Could add sparse skip + Int8 |
| Query pre-rotation | MLX matmul | 1 | Could fuse into score kernel |
| Output inverse rotation | MLX matmul | 1 | Could fuse into value kernel |
| **Total** | | **6** | **Could reduce to 3** |

### Fused Kernel Opportunities

#### A. Fuse Query Rotation into Score Kernel (save 1 dispatch)

Currently: `qRot = matmul(queries, Π_key^T)` → then `score = Metal_score(qRot, packed_K)`.

Fused: Single kernel that loads Π_key^T, rotates Q in-register, then computes score against packed K. For dim=128, this is a 128×128 matmul per query — fits in shared memory.

**Expected impact**: -10-15% decode latency from eliminating one kernel dispatch + one intermediate FP32 array.

#### B. Fuse Output Rotation into Value Kernel (save 1 dispatch)

Currently: `rotOutput = Metal_value(attn, packed_V)` → then `output = matmul(rotOutput, Π_val)`.

Fused: Value kernel outputs into shared memory, applies Π_val rotation in-register, writes final output directly.

**Expected impact**: -10-15% decode latency.

#### C. Fuse Score + Softmax + Value (the holy grail — save 3 dispatches → 1)

A single "TurboFlashAttention" kernel that:
1. Computes Q×K scores from packed indices (existing score logic)
2. Applies online softmax (no full materialization — streaming)
3. Accumulates Attn×V from packed indices (existing value logic)
4. Applies output rotation

This matches SwiftLM's approach with `sdpa_vector` which does everything in one kernel. Would reduce 6 dispatches → 1.

**Challenge**: Online softmax requires two-pass or log-sum-exp trick within the kernel. More complex but well-understood (FlashAttention paper).

**Expected impact**: -30-40% decode latency at long contexts where kernel dispatch overhead dominates.

#### D. Fuse Score + Softmax (intermediate step)

Easier than full fusion: score kernel writes to shared memory, applies causal mask + softmax in the same dispatch. Eliminates the intermediate score array.

**Expected impact**: -15-20% decode latency.

### Priority Ranking

1. **A + B (rotation fusion)**: Low complexity, -20-30% combined. Can do with existing MLXFast.metalKernel.
2. **D (score + softmax fusion)**: Medium complexity, additional -15-20%.
3. **C (full SDPA fusion)**: High complexity, but the ultimate goal. Matches SwiftLM architecture.

## Question 3: What would Int8 attention require?

See full spec in `specs/int8-quantized-attention.md`. Summary:

### Three Approaches (in order of feasibility)

#### Approach A: Custom Metal Kernel via MLXFast.metalKernel (Now)

- Write Int8 score/value kernels using manual SIMD lane accumulation
- No AMX/Neural Engine access — just bandwidth savings from reading Int8 instead of FP32
- **Speedup: 1.1-1.3x**, **Complexity: Medium**, **Timeline: 1-2 weeks**

#### Approach B: Custom .metallib with AMX Intrinsics (Best Performance)

- Compile separate Metal library with `simdgroup_matrix<int8_t>` for hardware-accelerated Int8 GEMM
- Requires Metal 3+ (M3/M4/M5 for best performance, M1/M2 partial support)
- Load metallib at runtime, dispatch via Metal compute pipeline (bypass MLXFast)
- **Speedup: 1.4-1.8x**, **Complexity: High**, **Timeline: 3-4 weeks**

#### Approach C: Int8 KV Storage Only (Simplest)

- Store K/V as Int8 with per-row scales (no rotation, no turbo)
- Dequantize to FP16 before standard SDPA
- Memory savings only, no compute speedup
- **Speedup: 1.0x (memory only)**, **Complexity: Low**, **Timeline: 1 week**

### What We'd Need

1. **Verify M1 Max AMX Int8 support**: The Apple GPU ISA docs are sparse. Need a compilation test with `simdgroup_matrix<int8_t>` to confirm it compiles and runs on M1.

2. **Build infrastructure for custom .metallib**: SwiftLM has `build-metallib.sh` — we'd need similar. Currently all our Metal kernels go through MLXFast.metalKernel (JIT compiled from source strings).

3. **Quality validation**: Int8 attention on thinking models may degrade the thinking phase. Need to benchmark Think PPL and Think KLD specifically.

4. **Integration with turbo**: The combined approach (turbo compressed storage + Int8 attention compute) would modify our existing score/value kernels to dequant packed → Int8 instead of → FP32.

---

## Optimization Roadmap (Proposed)

| Priority | Optimization | Speed | Memory | Complexity |
|:--------:|-------------|:-----:|:------:|:----------:|
| 1 | Fuse Q rotation into score kernel | +10-15% | 0 | Low |
| 2 | Fuse output rotation into value kernel | +10-15% | 0 | Low |
| 3 | Reduce prefillStepSize to 512 | 0 (decode) | -0.6GB peak | Trivial |
| 4 | Int8 score kernel (Approach A) | +10-30% | 0 | Medium |
| 5 | Fuse score + softmax | +15-20% | 0 | Medium |
| 6 | Full TurboFlashAttention kernel | +30-40% | 0 | High |
| 7 | Int8 via custom metallib (Approach B) | +40-80% | 0 | High |
| 8 | SSD expert streaming | 0 | -80% for MoE | High |
