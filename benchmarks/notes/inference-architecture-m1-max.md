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

> The M5 Max GPU architecture features a Total GPU Register File capacity estimated at ~8.2 MiB for the 40-core configuration. It also has a dedicated Neural Accelerator register pool.  Each GPU core now includes a dedicated hardware block for matrix multiplication. These units have their own internal high-speed buffers (register-adjacent memory) to handle the 1,024 fused multiply-accumulate operations per core, per cycle, which drastically reduces standard register pressure during AI tasks. The Unified memory bandwidth is 614 GB/s.

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

## What is an "Op"?

An **op** is one Metal compute dispatch — a single `dispatchThreads()` call encoded into a command buffer. Each op launches one Metal kernel on the GPU. Reshape, transpose, and slice operations are **NOT ops** — they just create array view metadata with zero GPU cost.

### Ops per decode token (all models)

#### Architecture summary

| Model | Layers | Hidden | Attn Heads | KV Heads | Head Dim | MoE | Experts×TopK | GDN | Sliding Window |
|-------|--------|--------|-----------|----------|----------|-----|-------------|-----|---------------|
| Gemma4 E2B | 30 | 2816 | 16 | 8 | 256 | No | — | No | 1024 |
| Gemma4 26B-A4B | 30 | 2816 | 16 | 8 | 256 | Yes | 128×8 | No | 1024 |
| Gemma4 31B | 60 | 5376 | 32 | 16 | 256 | No | — | No | 1024 |
| Qwen3.5-27B | 64 | 5120 | 24 | 4 | 256 | No | — | Yes (48/64) | — |
| Qwen3.5-35B-A3B | 40 | 2048 | 16 | 2 | 256 | Yes | 256×8 | Yes (30/40) | — |
| GPT-OSS-20B | 24 | 2880 | 64 | 8 | 64 | Yes | 32×4 | No | 128 |

#### Gemma4 E2B (30 layers, dense, pure attention)

| Op type | What it dispatches | Per layer | Total |
|---------|-------------------|-----------|-------|
| Quantized GEMV | Q, K, V, O projections | 4 | 120 |
| Quantized GEMV | MLP: gate, up, down | 3 | 90 |
| RMSNorm | inputNorm, postAttnNorm, preFFNNorm, postFFNNorm | 4 | 120 |
| Fused RMSNormRoPE | Q norm+rope, K norm+rope | 2 | 60 |
| RMSNorm (v_norm) | Value normalization (no learnable weight) | 1 | 30 |
| SDPA | Attention (vector mode decode) | 1 | 30 |
| Element-wise | residual add, gelu×mul (compiledGeglu), norm+residual (compiledNormResidual) | ~2 | ~60 |
| Cache update | KV cache slice-assign | 1 | 30 |
| Reshape/transpose/slice | **No dispatch** — array view only | — | 0 |
| **Total** | | **~18** | **~540** |

Note: some norms and residual adds are fused via `compiledNormResidual` and `compiledGeglu`, reducing the effective count to **~420 dispatches**.

#### Gemma4 26B-A4B (30 layers, MoE, pure attention)

Same attention ops as E2B, but MoE replaces the dense MLP:

| Op type | What it dispatches | Per layer | Total |
|---------|-------------------|-----------|-------|
| Attention ops | Same as E2B (Q/K/V/O proj, norms, RoPE, SDPA) | ~12 | ~360 |
| Shared MLP | gate, up, gelu×mul, down | 4 | 120 |
| Router | Linear projection (hidden→128 experts) | 1 | 30 |
| Top-K selection | argPartition + takeAlong | ~2 | ~60 |
| Softmax + scaling | Expert weight computation | ~2 | ~60 |
| **gatherQuantizedMM** | Fused gate_up projection (128 experts, top-8) | 1 | 30 |
| **gatherQuantizedMM** | Down projection (128 experts, top-8) | 1 | 30 |
| Expert reduce | Weighted sum + squeeze | ~2 | ~60 |
| Additional norms | preNorm2, postNorm1, postNorm2 | 3 | 90 |
| Element-wise | Residuals, expert weight scaling | ~2 | ~60 |
| **Total** | | **~30** | **~900** |

MoE adds ~360 ops per token vs dense. The `gatherQuantizedMM` dispatches are the expensive MoE-specific kernels.

#### Gemma4 31B (60 layers, dense, pure attention)

Same per-layer structure as E2B but 2× layers, larger hidden (5376), and larger MLP (21504):

| Op type | What it dispatches | Per layer | Total |
|---------|-------------------|-----------|-------|
| Attention ops | Q/K/V/O proj, norms, RoPE, SDPA, v_norm | ~12 | ~720 |
| Dense MLP | gate (5376→21504), up, gelu×mul, down (21504→5376) | ~4 | ~240 |
| Norms + residuals | inputNorm, postAttnNorm, preFFN, postFFN, fused | ~4 | ~240 |
| Cache update | KV cache | 1 | 60 |
| **Total** | | **~21** | **~1260** |

Largest op count of all models due to 60 layers. Each GEMV is also larger (5376 hidden, 21504 intermediate).

#### Qwen3.5-27B (64 layers, dense, GDN hybrid)

48 GatedDeltaNet layers + 16 full attention layers. GDN layers replace SDPA with a recurrent state update:

| Op type | Per GDN layer | ×48 | Per Attn layer | ×16 | Total |
|---------|--------------|-----|---------------|-----|-------|
| Input projection (QKV/conv) | 3 | 144 | 4 | 64 | 208 |
| Conv1d (GDN short conv) | 1 | 48 | — | — | 48 |
| **GDN Metal kernel** (fused or standard) | 1 | 48 | — | — | 48 |
| RMSNorm (q_norm, k_norm) | 2 | 96 | 2 | 32 | 128 |
| RoPE | — | — | 2 | 32 | 32 |
| SDPA | — | — | 1 | 16 | 16 |
| Output projection | 1 | 48 | 1 | 16 | 64 |
| Dense MLP (gate, up, silu×mul, down) | 4 | 192 | 4 | 64 | 256 |
| Norms + residuals | ~4 | 192 | ~4 | 64 | 256 |
| Cache update | 1 | 48 | 1 | 16 | 64 |
| **Total** | **~17** | | **~19** | | **~1120** |

GDN layers use the custom Metal kernel (standard or fused) which replaces SDPA — fewer attention ops but adds the conv1d and state update.

#### Qwen3.5-35B-A3B (40 layers, GDN hybrid + MoE)

30 GDN+MoE layers + 10 Attention+MoE layers:

| Op type | Per GDN+MoE layer | ×30 | Per Attn+MoE layer | ×10 | Total |
|---------|-------------------|-----|---------------------|-----|-------|
| Input projection/conv | ~4 | 120 | ~4 | 40 | 160 |
| GDN kernel | 1 | 30 | — | — | 30 |
| SDPA | — | — | 1 | 10 | 10 |
| Norms (Q/K/v_norm) | ~3 | 90 | ~3 | 30 | 120 |
| RoPE | — | — | 2 | 20 | 20 |
| Output projection | 1 | 30 | 1 | 10 | 40 |
| Router (hidden→256) | 1 | 30 | 1 | 10 | 40 |
| Top-K + softmax | ~3 | 90 | ~3 | 30 | 120 |
| Shared expert MLP | 3 | 90 | 3 | 30 | 120 |
| **gatherQuantizedMM** (256 experts, top-8) | 2 | 60 | 2 | 20 | 80 |
| Expert reduce + norms | ~3 | 90 | ~3 | 30 | 120 |
| Norms + residuals | ~4 | 120 | ~4 | 40 | 160 |
| **Total** | **~25** | | **~27** | | **~1020** |

Most complex architecture: GDN + MoE + shared experts. The `gatherQuantizedMM` handles 256 experts with top-8 routing.

#### GPT-OSS-20B (24 layers, MoE, pure attention)

12 sliding window + 12 global attention layers, all with MoE (32 experts, top-4):

| Op type | What it dispatches | Per layer | Total |
|---------|-------------------|-----------|-------|
| Attention: Q, K, V, O projections | Quantized GEMV | 4 | 96 |
| RMSNorm | Input, post-attn, pre-FFN norms | ~3 | 72 |
| RoPE (YarnRoPE) | Q and K rotation | 2 | 48 |
| SDPA | Vector attention (with attention sinks) | 1 | 24 |
| Router | Linear (2880→32) | 1 | 24 |
| Top-K + softmax | Expert selection + weight computation | ~2 | 48 |
| **gatherQuantizedMM** | Fused gate_up (32 experts, top-4, swiglu) | 1 | 24 |
| **gatherQuantizedMM** | Down projection (32 experts, top-4) | 1 | 24 |
| Expert reduce | Weighted sum | ~2 | 48 |
| Element-wise | Residuals, compiled swiglu | ~2 | 48 |
| Cache update | KV cache | 1 | 24 |
| **Total** | **~20** | | **~480** |

Fewest total ops due to only 24 layers. The swiglu activation uses a compiled path (`compiledSwiglu`).

#### Comparison: total ops per decode token

| Model | Layers | Ops/layer | Total ops | Weight reads | Theoretical floor |
|-------|--------|-----------|-----------|-------------|-------------------|
| **Gemma4 E2B** | 30 | ~14 | **~420** | ~1 GB | 2.5 ms |
| **Gemma4 26B-A4B** | 30 | ~30 | **~900** | ~14 GB | 35 ms |
| **Gemma4 31B** | 60 | ~21 | **~1260** | ~17 GB | 42.5 ms |
| **Qwen3.5-27B** | 64 | ~18 | **~1120** | ~14 GB | 35 ms |
| **Qwen3.5-35B-A3B** | 40 | ~25 | **~1020** | ~18 GB | 45 ms |
| **GPT-OSS-20B** | 24 | ~20 | **~480** | ~10 GB | 25 ms |

Weight reads are approximate for 4-bit quantized models. Theoretical floor = weight_size / 400 GB/s.

MoE models read fewer weights per token than the total model size because only the routed experts' weights are read (top-K out of total). The "weight reads" column accounts for this — for Gemma4 26B with 128 experts and top-8, only ~8/128 = 6.25% of expert weights are read per token, plus shared attention + router weights.

### How ops flow from CPU to GPU

```
 CPU (Swift + MLX C++)                          GPU (Metal)
 ═══════════════════                            ═══════════

 ① Model forward pass (LAZY — builds graph, no GPU work)
 ┌─────────────────────────────────────┐
 │ for layer in 0..<30:                │
 │   norm(x)        → graph node       │     (idle)
 │   qProj(x)       → graph node       │
 │   kProj(x)       → graph node       │
 │   rope(q)        → graph node       │
 │   cache.update() → graph node       │
 │   SDPA(q,k,v)    → graph node       │
 │   oProj(out)     → graph node       │
 │   mlp(h)         → graph node       │
 │ sample(logits)   → graph node       │
 └─────────────────┬───────────────────┘
                   │
 ② asyncEval(token) — walks graph, encodes into Metal command buffers
 ┌─────────────────┴───────────────────┐
 │                                     │
 │  Command Buffer 1 (ops 1-100):      │
 │  ┌────────────────────────────────┐ │
 │  │ set pipeline: qgemv_float16    │ │
 │  │ bind buffer 0: x (input)       │ │
 │  │ bind buffer 1: weights         │ │
 │  │ dispatch(grid, threadgroup)    │──────▶ GPU starts executing
 │  │                                │ │     ops 1-100 immediately
 │  │ set pipeline: rms_norm         │ │
 │  │ bind buffer 0: ...             │ │
 │  │ dispatch(grid, threadgroup)    │ │
 │  │ ... (98 more ops)              │ │
 │  └──────────────┬─────────────────┘ │
 │                 │ COMMIT            │
 │                 │                   │      ┌──────────────────┐
 │  Command Buffer 2 (ops 101-200):    │      │ GPU executing    │
 │  ┌────────────────────────────────┐ │      │ CB1 ops 1-100    │
 │  │ set pipeline: sdpa_vector      │ │      │ while CPU encodes│
 │  │ bind buffer 0: queries         │ │      │ CB2              │
 │  │ bind buffer 1: cached_keys     │ │      └────────┬─────────┘
 │  │ dispatch(grid, threadgroup)    │ │               │
 │  │ ... (99 more ops)              │ │               │
 │  └──────────────┬─────────────────┘ │               │
 │                 │ COMMIT            │               │
 └─────────────────┴───────────────────┘               │
                                                       │
 ③ .item() — CPU waits for final result               │
 ┌─────────────────────────────────────┐               │
 │ block until GPU finishes CB2  ◄─────────────────────┘
 │ read token ID from GPU buffer       │
 └─────────────────────────────────────┘
```

### Per-op encoding cost

Each op requires the CPU to:
1. **Set pipeline state** — look up the pre-compiled Metal kernel (~1μs, cached)
2. **Bind buffers** — point the kernel at input/output arrays (~2-5μs per buffer)
3. **Set parameters** — threadgroup size, grid dimensions (~1μs)
4. **Dispatch** — add to command buffer (~1μs)

Total: **~5-10μs per op**. With ~200 ops: **~1-2ms of pure encoding time**.

### Why fewer ops = faster

Every op we eliminate (through kernel fusion) removes:
- ~5-10μs of CPU encoding time
- One GPU kernel launch (with its scheduling overhead and potential idle gap)
- One set of intermediate buffer allocations

Our fused kernels so far:
- **RMSNormRoPE**: 5 ops → 3 ops per layer (saves ~60 ops total)
- **compiledNormResidual**: 2 ops → 1 op per layer (saves ~30 ops)
- **compiledGeglu**: 2 ops → 1 op per layer (saves ~30 ops)

---

## asyncEval: What "Async" Actually Means (and Where It Blocks)

### The truth about asyncEval

Despite the name, `asyncEval` is **only async in one narrow sense**: it doesn't wait for the GPU to finish executing. Everything else — graph traversal, topological sort, and Metal command encoding — happens **synchronously on the calling thread** before the function returns.

```
asyncEval(token) — what actually happens on the CPU:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Phase 1: Graph traversal (DFS)                ~0.5ms      │
│  ├─ Walk ~500 nodes                                        │
│  ├─ Build dependency counts                                │
│  └─ Identify cross-stream fences                           │
│                                                            │
│  Phase 2: Topological sort (BFS)               ~0.3ms      │
│  ├─ Build execution tape in dependency order               │
│  └─ Width-limited BFS to control memory                    │
│                                                            │
│  Phase 3: Execute tape (THE HOT LOOP)          ~8-9ms      │
│  ├─ For each of ~200 ops:                                  │
│  │   ├─ Check fences/events for dependencies               │
│  │   ├─ gpu::eval(arr) → encode into Metal command buffer  │
│  │   ├─ *** MEMORY PRESSURE CHECK ***                      │
│  │   │   if (active_memory > memory_limit):                │
│  │   │     commit all open command buffers                 │
│  │   │     scheduler::wait_for_one() ← BLOCKS!             │
│  │   │     (repeat until memory drops below limit)         │
│  │   └─ Mark as evaluated, update fences                   │
│  │                                                         │
│  │   Every max_ops_per_buffer ops:                         │
│  │     commit command buffer → GPU starts executing        │
│  │                                                         │
│  Phase 4: Finalize                             ~0.2ms      │
│  ├─ Commit remaining command buffers                       │
│  └─ Signal completion events                               │
│                                                            │
│  RETURN (GPU still executing, CPU free to continue)        │
└────────────────────────────────────────────────────────────┘

vs eval(token):
  Same as above, PLUS:
  └─ .wait() — blocks until GPU finishes all command buffers
```

### The hidden memory pressure blocker

Inside the tape execution loop, MLX checks memory on EVERY op:

```cpp
if (scheduler::n_active_tasks() > MAX_ACTIVE_TASKS ||
    (get_active_memory() > get_memory_limit() &&
     scheduler::n_active_tasks() > 0)) {
    // Commit all open streams
    gpu::finalize(s);
    scheduler::wait_for_one();   // ← CPU BLOCKS waiting for GPU
    while (get_active_memory() > get_memory_limit()) {
        scheduler::wait_for_one(); // ← keeps blocking until memory drops
    }
}
```

**If active memory exceeds the memory limit, asyncEval stops encoding and waits for the GPU to finish and free memory.** This turns the "async" eval into a synchronous stall mid-encoding.

### Memory limit: how it's set

```cpp
// At Metal allocator init (allocator.cpp line 52):
block_limit_ = min(1.5 × recommendedMaxWorkingSetSize, 0.95 × totalSystemMemory)
```

On a 64 GB M1 Max:
- `recommendedMaxWorkingSetSize` ≈ 48 GB (75% of 64 GB, set by macOS)
- `block_limit_` = min(1.5 × 48, 0.95 × 64) = min(72, 60.8) = **60.8 GB**

On a 32 GB M1 Max:
- `recommendedMaxWorkingSetSize` ≈ 22 GB
- `block_limit_` = min(33, 30.4) = **30.4 GB**

`active_memory` tracks every currently-allocated Metal buffer (model weights + KV cache + all intermediate tensors). If this exceeds `block_limit_`, asyncEval blocks.

**Why this might be triggering**: With `max_ops_per_buffer=300`, more intermediates are alive simultaneously (Metal holds them until command buffer completes). If the model weights + KV cache + intermediates approach the limit, the memory pressure check fires and asyncEval stalls mid-encoding.

### The profiler breakdown confirms this

```
CPU profiler (Gemma4 E2B, 216 tokens average):
  model forward:   2.06ms (16%)  ← lazy graph building (fast)
  asyncEval:      10.73ms (84%)  ← graph walk + encoding + potential memory stalls
  .item() sync:    0.01ms (0%)   ← GPU already done by this point
```

The 10.73ms in asyncEval is NOT GPU execution time (the GPU finishes before `.item()` needs the result). It's the CPU doing graph traversal + Metal encoding + potential memory pressure stalls.

---

## How to Make It Faster: Summary

```
                                  Current decode token time: ~12ms
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. REDUCE WEIGHT READS (attack the 2.5ms GPU compute floor)        │
│     ├─ Quantization: 4-bit = 4× less data to read                   │
│     ├─ MoE: only read top-K expert weights, not all                 │
│     ├─ Batch decode: amortize 1 GB read across B tokens             │
│     └─ Smaller models: less total weight data                       │
│                                                                     │
│  2. REDUCE OPS (attack the ~2ms encoding overhead)                  │
│     ├─ Fused kernels: NormRoPE, compiledGeglu, etc.                 │
│     ├─ Batched QKV GEMV: 3 dispatches → 1                           │
│     ├─ Fused MLP: 5 dispatches → 1                                  │
│     └─ Goal: ~420 ops → ~120 ops per token                          │
│                                                                     │
│  3. SMARTER COMMAND BUFFER MANAGEMENT                               │
│     ├─ Adaptive ops-per-buffer: high for decode, MB-limited         │
│     │   for prefill                                                 │
│     ├─ Track output buffer sizes (not just inputs)                  │
│     └─ Metal Indirect Command Buffers: pre-encode once,             │
│        replay each token (eliminates encoding entirely)             │
│                                                                     │
│  4. OVERLAP CPU AND GPU WORK                                        │
│     ├─ Incremental eval: encode + submit layer-by-layer             │
│     │   instead of building full graph then encoding                │
│     ├─ Double-buffer: build token N+1 graph while GPU               │
│     │   executes token N                                            │
│     └─ Background encoding thread: move graph walk + Metal          │
│        encoding off the main thread                                 │
│                                                                     │
│  5. FIX ASYNCEVAL MEMORY PRESSURE BLOCKING (VERIFIED NOT THE ISSUE) │
│     ├─ Profile: what is active_memory vs memory_limit?              │
│     ├─ Are intermediates being retained too long?                   │
│     ├─ Is the memory limit set appropriately for the workload?      │
│     ├─ Command buffer intermediates: Metal keeps all buffers        │
│     │   alive until CB completes — fewer ops/CB = less memory       │
│     └─ This may be the #1 source of the ~8ms stall in               │
│        asyncEval — the CPU blocks waiting for GPU to free           │
│        memory before it can continue encoding                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

     Theoretical floor         Achievable target       Current
     (bandwidth-limited)       (with all optimizations) 
     
     2.5 ms/token              4-5 ms/token             12 ms/token
     400 tok/s                 200-250 tok/s            80-90 tok/s
```

---

## Software Pipeline: Token Generation Loop

### What Happens Per Decode Token

```
              CPU                                     GPU
              ───                                     ───
         ┌──────────────────────┐
    ①    │ Build computation   │               (idle, waiting
         │ graph for token N    │                 for work)
         │ • 30 layer forwards  │
         │ • attention + MLP    │
         │ • logit processing   │
         │ • sampling           │
         │ (ALL LAZY — no GPU   │
         │  work yet)           │
         └──────────┬───────────┘
                    │
    ②    ┌──────────▼───────────┐
         │ asyncEval(token)      │──────────┐
         │ Submit graph to GPU   │          │
         │ (returns immediately) │          ▼
         └──────────┬───────────┘    ┌──────────────────┐
                    │                │ GPU evaluates    │
    ③    ┌──────────▼───────────┐    │ entire graph:    │
         │ didSample(token)      │   │ • 30 layers o    │
         │ Update penalty ring   │   │   matmul + nor   │
         │ (lazy, adds to next   │   │   + attention    │
         │  token's graph)           │ • logit softmax  │
         └──────────┬───────────┘    │ • sampling       │
                    │                │                  │
    ④    ┌──────────▼───────────┐    │   ...working...  │
         │ .item(Int.self)       │   │                  │
         │ *** CPU BLOCKS ***    │◄──┤ GPU finishes,    │
         │ waiting for token     │   │ returns result   │
         │ N-1's result          │   └──────────────────┘
         └──────────┬───────────┘
                    │
    ⑤    ┌──────────▼───────────┐
         │ Return tokenId to    │
         │ caller / detokenize  │
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

## Metal Command Buffer Lifecycle and Peak Memory

### How Metal manages buffer lifetimes

Metal keeps **every buffer referenced by a command buffer alive** until that command buffer **completes GPU execution**. This is a fundamental design constraint — the GPU might read any buffer at any point during execution, so nothing can be freed until the entire command buffer is done.

MLX batches multiple operations into each command buffer, controlled by `max_ops_per_buffer` (default: 100 for M1 Max). Each operation allocates its output buffer when it runs. The total number of buffer allocations is the same regardless of batch size — what changes is **how many are alive simultaneously**.

### The conveyor belt analogy

Think of it like a restaurant kitchen with a pass (the command buffer):

- **Small batches (ops_per_buffer=100)**: The kitchen finishes and clears plates from the first 100-order batch before the second batch's plates pile up. Peak counter space: ~100 plates.
- **One giant batch (ops_per_buffer=500)**: All 500 orders' plates sit on the counter simultaneously until the entire batch is complete. Peak counter space: ~500 plates.

The total food served is identical. But the peak counter space (memory) is dramatically different.

### Why this matters differently for decode vs prefill

**Decode (T=1)**: Each intermediate is a tiny vector.
- Per-operation output: ~5 KB (hidden_dim × 2 bytes)
- 500 ops × 5 KB = ~2.5 MB peak → **negligible**
- Higher ops_per_buffer = fewer command buffer commits = faster decode

**Prefill (T=2048)**: Each intermediate is 2048× larger.
- Per-operation output: ~11.5 MB (T × hidden_dim × 2 bytes)
- Attention score matrix: `[B, H, T, T]` = [1, 16, 2048, 2048] × 4 bytes = **256 MB per layer**
- 100 ops with several attention layers = **easily 1+ GB of simultaneously-alive buffers**
- 500 ops = **3+ GB peak** before any commit triggers reclamation

### Current heuristic gap

MLX commits a command buffer when either threshold is exceeded:
```cpp
return (stream.buffer_ops > max_ops_per_buffer_) ||
    ((stream.buffer_sizes >> 20) > max_mb_per_buffer_);
```

**The problem**: `buffer_sizes` only tracks **input** buffer sizes. Output allocations (like the 256 MB attention score matrix) are invisible to this check. During prefill, the largest allocations are outputs — so the memory-based trigger never fires when it should.

### Benchmark evidence

| ops_per_buffer | Decode (tok/s) | Impact on prefill memory |
|---------------|---------------|------------------------|
| 25 | 79.0 (-11%) | Lowest peak (most frequent commits) |
| 100 (default) | 89.3 | Moderate peak |
| 300 | 93.5 (+5%) | **High peak — prefill memory blows up** |
| 500 | 92.0 | Plateaus — all ops in one buffer anyway |

### The solution: adaptive memory-aware commits

Track **total** referenced memory (inputs + outputs) and use that as the commit trigger. Set ops_per_buffer high (500) for decode speed, but let a memory limit (200 MB) trigger early commits during prefill when large tensors accumulate. See `spec-adaptive-command-buffer-management.md` for full design.

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
