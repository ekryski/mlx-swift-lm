# Inference Architecture: MLX Swift on M1 Max

## Hardware: Apple M1 Max

### Memory Architecture (Unified Memory)

There is **no separate GPU VRAM**. CPU and GPU share the same physical LPDDR5 memory. The GPU reads model weights directly from the same memory addresses the CPU loaded them to — zero-copy access.

```
┌─────────────────────────────────────────────────┐
│                 M1 Max SoC                      │
│                                                 │
│  ┌─────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ CPU     │  │ GPU      │  │ Neural Engine │   │
│  │ 10 core │  │ 32 core  │  │ 16 core       │   │
│  └────┬────┘  └────┬─────┘  └───────┬───────┘   │
│       │            │                │           │
│  ┌────┴────────────┴────────────────┴───────┐   │
│  │       System Level Cache (SLC)           │   │
│  │              48 MB                       │   │
│  └──────────────────┬───────────────────────┘   │
│                     │                           │
│  ┌──────────────────┴───────────────────────┐   │
│  │    Memory Controller (512-bit LPDDR5)    │   │
│  │         400 GB/s bandwidth               │   │
│  └──────────────────┬───────────────────────┘   │
└─────────────────────┼───────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │    Unified Memory       │
         │    32 or 64 GB          │
         │  (model weights,        │
         │   KV cache,             │
         │   activations)          │
         └─────────────────────────┘
```

### GPU Cache Hierarchy

| Level | Size | Can cache model weights across tokens? |
|-------|------|---------------------------------------|
| Register file | 208 KB × 32 cores = 6.6 MB | No — cleared between kernel dispatches |
| Threadgroup memory | ~60 KB per core | No — per-threadgroup, not persistent |
| L1 cache | 8 KB × 32 = 256 KB | No — far too small, thrashed instantly |
| L2 cache | 512 KB (shared by all 32 GPU cores) | No — 0.05% of a 1 GB model |
| **SLC (L3)** | **48 MB** (shared by CPU + GPU) | Partially — holds ~4.7% of a 1 GB model |

### Weight Reading: Yes, ~1 GB Per Token

For Gemma4 E2B (2B params, 4-bit quantized ≈ 1 GB):

**Every decode token must read essentially all weights from DRAM.** Here's why:
- Decode is a sequential pass through ALL 30 layers
- Each layer's weights are used once per token
- By the time layer 30 finishes and the next token starts at layer 1, layer 1's weights have been evicted from the 48 MB SLC (the model cycles through 20× the SLC capacity)
- GPU L1 (8 KB) and L2 (512 KB) are irrelevant at this scale

**Theoretical decode floor**: 1 GB ÷ 400 GB/s = **2.5 ms per token** = 400 tok/s maximum. We achieve ~80 tok/s, meaning we're using ~20% of theoretical bandwidth. The remaining 80% is overhead from:
- KV cache reads/writes (~3 MB per token at 16K context)
- Activation intermediate memory traffic
- **CPU-side pipeline latency** (graph building, sync points)

---

## Software Pipeline: Token Generation Loop

### What Happens Per Decode Token

```
              CPU                                     GPU
              ───                                     ───
         ┌──────────────────────┐
    ①    │ Build computation     │              (idle, waiting
         │ graph for token N     │               for work)
         │ • 30 layer forwards   │
         │ • attention + MLP     │
         │ • logit processing    │
         │ • sampling            │
         │ (ALL LAZY — no GPU    │
         │  work yet)            │
         └──────────┬───────────┘
                    │
    ②    ┌──────────▼───────────┐
         │ asyncEval(token)      │──────────┐
         │ Submit graph to GPU   │          │
         │ (returns immediately) │          ▼
         └──────────┬───────────┘    ┌──────────────────┐
                    │                │ GPU evaluates     │
    ③    ┌──────────▼───────────┐    │ entire graph:     │
         │ didSample(token)      │    │ • 30 layers of    │
         │ Update penalty ring   │    │   matmul + norm   │
         │ (lazy, adds to next   │    │   + attention     │
         │  token's graph)       │    │ • logit softmax   │
         └──────────┬───────────┘    │ • sampling        │
                    │                │                   │
    ④    ┌──────────▼───────────┐    │    ...working...  │
         │ .item(Int.self)       │    │                   │
         │ *** CPU BLOCKS ***    │◄───┤ GPU finishes,     │
         │ waiting for token     │    │ returns result    │
         │ N-1's result          │    └──────────────────┘
         └──────────┬───────────┘
                    │
    ⑤    ┌──────────▼───────────┐
         │ Return tokenId to     │
         │ caller / detokenize   │
         └──────────────────────┘
```

### The Pipeline Problem

The "wait" the user observes is **step ①**: the CPU spends significant time building the lazy computation graph (30 layers of operations) before the GPU even starts work.

In an ideal pipeline, steps ① and GPU execution would overlap:

```
   Token N-1            Token N              Token N+1
   ────────             ──────               ─────────
   
CPU: [①build N-1][②submit][③④wait]  [①build N][②submit][③④wait]
GPU:              [████ evaluate N-1 ████]    [████ evaluate N ████]
                   ↑                          ↑
                   GPU starts when            Overlap: CPU builds N+1
                   CPU submits                while GPU evaluates N
```

**But in our implementation**, the `.item()` sync at step ④ forces the CPU to wait for the GPU to finish before it can start building the next token's graph:

```
CPU: [①build][②sub][③][④ WAIT ████████████][①build][②sub][③][④ WAIT ████████████]
GPU:                  [████ evaluate ████]                   [████ evaluate ████]
                                          ↑                 ↑
                                          GPU idle!         GPU idle again!
                                          CPU building      
                                          next graph        
```

The GPU goes idle between tokens while the CPU builds the graph. This is the "wait" — it's actually the GPU waiting for the CPU, not the CPU waiting for the GPU.

### Where CPU Time Goes (Graph Building)

For one Gemma4 E2B decode token, the CPU constructs:

| Operation | Count | CPU Cost |
|-----------|-------|----------|
| Layer forward calls | 30 | ~200μs each = 6ms |
| RMSNorm graph nodes | 90+ (3 per layer) | negligible per node |
| Linear projection graph nodes | 150+ (5 per layer) | negligible per node |
| Attention (SDPA) graph nodes | 30 | negligible per node |
| RoPE graph nodes | 60 | negligible per node |
| Cache update operations | 30 | ~50μs each = 1.5ms |
| Mask creation | 30 | ~20μs each = 0.6ms |
| Total graph nodes | ~500+ | |
| **Total CPU graph building** | | **~8-12ms** |

At 80 tok/s, each token takes ~12.5ms total. If 8-12ms of that is CPU graph building, **the GPU only has 0.5-4.5ms to actually compute** — far less than the theoretical 2.5ms weight-read floor.

This means: **the CPU graph building is the bottleneck, not the GPU compute or memory bandwidth.**

### Comparison with Python mlx-lm

Python `mlx` has the SAME lazy evaluation model, but:

1. **Python's `mx.async_eval()` and `mx.eval()` use the same C++ backend** as Swift's
2. **Python model forward pass** builds the same computation graph
3. **BUT**: Python's `.item()` sync happens in a different place in the loop, AND Python's GIL-based threading model may allow different pipelining

The key difference may be in **graph construction overhead**:
- Python: native C++ operators create graph nodes efficiently
- Swift: each MLX operation goes through Swift → C bridge → C++ operation
- The Swift bridge overhead per operation may be higher than Python's native bindings

With ~500 operations per token, even 10μs overhead per Swift→C bridge call = 5ms per token.

---

## Optimization Opportunities

### 1. Reduce Graph Building Time
- **Graph compilation / caching**: MLX's `compile()` can pre-compile subgraphs. If the same graph structure repeats each token (it does for decode), the compiled graph could be reused.
- **Reduce bridge calls**: Batch multiple operations into fewer Swift→C transitions.

### 2. Better Pipeline Overlap
- Move `.item()` earlier or use callbacks instead of blocking sync
- Double-buffer: start building token N+1's graph while GPU evaluates token N

### 3. Reduce Operation Count
- Fused operations (NormRoPE: already done, saves 60 graph nodes)
- Fused norm+residual (already done via compiledNormResidual)
- Further fusion opportunities in the attention block

### 4. Reduce Weight Reads (Structural)
- **Batch decode** (Phase 4): Process B tokens per weight read instead of 1
- **Model parallelism**: Split model across multiple GPU cores for pipelined execution

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Model weights (4-bit) | ~1 GB |
| Memory bandwidth | 400 GB/s |
| Theoretical decode floor | 2.5 ms/token (400 tok/s) |
| Actual decode | ~12.5 ms/token (80 tok/s) |
| CPU graph building | ~8-12 ms/token (estimated) |
| GPU compute | ~2.5-4 ms/token (estimated) |
| SLC cache | 48 MB (4.7% of model) |
| GPU L2 | 512 KB |
| Graph nodes per token | ~500+ |
| Swift→C bridge calls per token | ~500+ |
