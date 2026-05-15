# 033 — Block-sparse SDPA Metal kernel

- **Status:** spec, foundational (multi-month research kernel; enables spec 034 + full FlashPrefill family)
- **Branch:** new branch off alpha; multi-repo (mlx + mlx-c + mlx-swift + mlx-swift-lm)
- **Depends on:** none directly; spec 031 should land first to validate sparse-attention infrastructure end-to-end without kernel risk
- **Origin:** Research review 2026-05-08; the unifying primitive behind [MInference](https://github.com/microsoft/MInference), [FlashPrefill](https://arxiv.org/abs/2603.06199), [FlexPrefill](https://arxiv.org/abs/2502.20766), [XAttention](https://github.com/mit-han-lab/x-attention), [TriangleMix](https://arxiv.org/pdf/2507.21526), [UniPrefill](https://arxiv.org/html/2605.06221)
- **Related:** [031](031-vertical-slash-sparse-prefill.md) (subset, ships first), [034](034-decode-side-kv-selection.md) (consumer)

## The insight

The five major sparse-prefill papers from 2024–2026 (MInference, FlexPrefill, XAttention, FlashPrefill, TriangleMix) all converge on the same kernel primitive: **block-sparse scaled-dot-product attention** where each query block attends only to a dynamically-selected subset of key blocks. The differences between papers are entirely in *how the subset is chosen* (offline patterns, JS-divergence detection, antidiagonal scoring, dynamic thresholding, decoding-time importance) — but they all output a `(query_block, key_block)` adjacency matrix and call the same kernel.

MLX exposes a fused dense SDPA (`MLXFast.scaledDotProductAttention`) but no block-sparse variant. To unlock any of the five papers' wins beyond what spec 031's two-region trick captures, we need the kernel.

This spec covers the Metal kernel, its C++ primitive, the C ABI, and the Swift wrapper — a four-repo PR chain matching the pattern established by the TurboQuant kernel and the spec-99 path-B sinks work.

## Why this is foundational, not "just another optimization"

Once this kernel exists:

- **Spec 031 phase 5** (full block-sparse pattern) drops in trivially — just a third pattern in the existing dispatcher
- **Spec 034** (Quest / RetrievalAttention-style decode-side top-k) is implementable with a one-row sparse SDPA
- **MInference / FlashPrefill / FlexPrefill / XAttention / TriangleMix** all become alternative pattern-detection front-ends on top of one kernel
- **Future research** in sparse attention is a per-experiment Swift change, not a new kernel each time

Without it: every paper that comes out is a Metal-kernel project. With it: every paper is a one-week pattern-detection spike.

## Design

### Kernel ABI

```cpp
// mlx::core::fast::block_sparse_attention
// Inputs:
//   queries:    [B, H, T_q, D]      (bf16 or fp16)
//   keys:       [B, H, T_k, D]
//   values:     [B, H, T_k, D]
//   adjacency:  [B, H, ceil(T_q/Bq), ceil(T_k/Bk)]   (bool / uint8)
//   scale:      float
//   block_q:    int (e.g. 64)
//   block_k:    int (e.g. 64)
// Outputs:
//   out:        [B, H, T_q, D]
//   lse:        [B, H, T_q]                 (optional, for fused merge with dense)
```

`adjacency[b, h, i, j] == 1` ⇒ query block `i` attends to key block `j`. Causal masking applied within blocks where adjacency is on the diagonal.

### Pass structure (FlashAttention-2 style)

Two-pass online softmax, identical math to dense `MLXFast.scaledDotProductAttention` but with the inner key-block loop iterating only over `j` such that `adjacency[b, h, i, j] == 1`:

```
for i in q_blocks:                                          # parallel over (b, h, i)
    Q_i = load(queries, block=i)                            # [Bq, D]
    m = -inf; l = 0; o = 0                                  # online softmax accumulators
    for j in active_k_blocks(adjacency[b, h, i, :]):        # sparse iteration
        K_j = load(keys, block=j)                           # [Bk, D]
        V_j = load(values, block=j)                         # [Bk, D]
        S = (Q_i @ K_j^T) * scale                           # [Bq, Bk]
        if causal and j == i: apply_causal_mask(S)
        m_new = max(m, rowmax(S))
        P = exp(S - m_new); l = l*exp(m - m_new) + rowsum(P)
        o = o*exp(m - m_new) + P @ V_j
        m = m_new
    out[block i] = o / l
    lse[block i] = m + log(l)                                # if requested
```

Adjacency packing: pre-sort active `j` indices per `(b, h, i)` row into a CSR-like `(row_ptr, col_idx)` representation on host or with a one-time Metal pass before the main kernel. Avoids a per-block adjacency lookup-and-skip cost in the inner loop.

### Block size choice

- `Bq = 64, Bk = 64` matches MLX's existing dense SDPA block size on Metal — reuses tuning intuition
- Adjacency at this granularity gives ~1% sparsity resolution (1 block of 64 ≈ 1% of 6.4K context)
- Block sizes are kernel template parameters; keep `(64, 64)` for V1, parameterize in V2

### Causal mask handling

When the adjacency includes any `(i, j)` with `i == j` or `i > j`, the kernel emits a within-block triangular mask. Adjacency lower-triangular is the caller's responsibility (the pattern detector emits causal-respecting adjacency).

### LSE output for compose-with-dense

Returning per-query `lse = m + log(l)` lets us *split* attention into a sparse-block partition + a dense "remainder" partition, computed by separate kernel calls, then merged via spec-030's `merge_lse` utility. Critical for the FlexPrefill / FlashPrefill threshold mechanisms that always-attend to a vertical stripe + a sparse remainder.

### Memory layout / quantization compatibility

Phase 1 ships fp16 / bf16 only. Phase N can extend to:

- TurboQuant-packed K (for compressed-domain block-sparse — composes with TurboQuant path B)
- Affine-quantized K (for `AffineQuantizedKVCache` consumers)

Each is a kernel variant, same adjacency interface.

## Implementation phases

1. **Phase 1 — Reference Python kernel.** Triton or pure JAX/PyTorch reference. Verify per-block equivalence with dense SDPA when adjacency is all-ones. Verify against MInference's reference on identical adjacency. ~2 weeks. Goal: have a numerical oracle.

2. **Phase 2 — Naive Metal port.** Port the two-pass kernel to Metal Shading Language using `simdgroup_matrix`. Optimize for correctness first. ~3 weeks. Goal: kernel runs, matches reference within fp16 tolerance, prefill wall-clock parity-or-better with dense SDPA at 50% sparsity.

3. **Phase 3 — Adjacency CSR + scheduling.** Replace dense adjacency tensor with row-CSR; add a one-shot "compaction" kernel that converts adjacency → CSR. Tune threadgroup partitioning. ~2 weeks. Goal: 2× over phase 2 at 90% sparsity.

4. **Phase 4 — LSE output + return-tuple plumbing.** Extend the kernel and Swift wrapper to optionally return per-query LSE. Wire through spec 031's `merge_lse`. ~1 week.

5. **Phase 5 — Multi-repo PR chain.** mlx (kernel + C++ primitive), mlx-c (C ABI), mlx-swift (Swift wrapper), mlx-swift-lm (consumer call sites). ~2 weeks for the chain landing including review cycles. Pattern matches the TurboQuant landing.

6. **Phase 6 — MInference pattern-detection front-end.** Port MInference's offline pattern-assignment + online sparse-index construction. Produces the adjacency this kernel consumes. ~2 weeks.

7. **Phase 7 — FlashPrefill / FlexPrefill / XAttention pattern detectors.** Each is a distinct adjacency-emitter; same kernel back-end. Pick one to start (FlashPrefill is the most recent and has the strongest results). ~1 week each. Order: FlashPrefill → FlexPrefill → XAttention.

8. **Phase 8 — TurboQuant variant.** Compressed-K block-sparse kernel. Composes the two memory-bandwidth wins. ~3 weeks. Goal: TurboQuant path B + sparse prefill on long context.

## Expected impact

End-to-end speedups stack as roughly:

```
spec 031 (vertical-slash, 70% of heads)             → 3-4× at 64K
spec 033 phase 6 (MInference, all 3 patterns)       → 5-7× at 64K
spec 033 phase 7 (FlashPrefill threshold dispatch)  → 8-15× at 64K, scales to 27× at 256K per paper
spec 032 (speculative prefill) composed with above  → multiplicative 2-3× on top
```

At 128K context with all three composed, projected target: **20-30× prefill** vs current dense baseline. PFlash claims 10× from speculative-prefill alone; full FlashPrefill claims 27.78× at 256K from kernel alone; the math checks out for stacking.

## Risk register

1. **Metal block-sparse kernel performance vs dense SDPA at low sparsity.** If the adjacency is denser than ~30% non-zero, the per-block branch overhead can outweigh the saved compute. Mitigation: kernel emits a "dense-fallback" recommendation in its profile output; pattern detectors fall back to dense when adjacency density exceeds a threshold (default 0.3).

2. **`simdgroup_matrix` doesn't expose sparse-friendly primitives.** May need to drop to scalar SIMD ops in the inner loop. Phase 2 is the de-risk: if the naive kernel can't beat dense at 90% sparsity on a target shape, the entire spec is at risk. Alternative: use the existing `MLXFast.scaledDotProductAttention` with a packed-K rearrangement (gather active blocks into a contiguous tensor, one dense SDPA call, scatter result). Slower than a fused sparse kernel but a viable fallback.

3. **MLX upstream may add block-sparse SDPA before we ship.** Track [ml-explore/mlx](https://github.com/ml-explore/mlx) for related work; if Apple lands a primitive, we use it instead and reduce this spec to phases 6–8 only.

4. **bf16 numerical drift in the online-softmax accumulators at very high sparsity.** Mitigation: do `m`, `l`, `o`-scale accumulators in fp32 within the kernel (single kernel-side conversion), output bf16. Same trick used in `MLXFast.scaledDotProductAttention`.

5. **Adjacency construction itself becomes the bottleneck.** MInference's online adjacency construction is O(T·H·D/B). At 128K this is itself non-trivial. Mitigation: phase 3's CSR compaction includes an explicit benchmark gate — adjacency-construction overhead must be < 5% of dense SDPA cost at the target context length, or the front-end is mis-designed.

## Acceptance criteria

- Kernel lands across mlx / mlx-c / mlx-swift / mlx-swift-lm via 4-repo PR chain
- Numerical equivalence with dense SDPA at adjacency=all-ones, within fp16 tolerance (rel err ≤ 1e-3)
- Wall-clock speedup ≥ 3× at 90% sparsity vs dense SDPA, on Gemma4-26B-A4B at 16K context (representative shape)
- LSE output validated against the reference (rel err ≤ 1e-3)
- TurboQuant variant lands with same numerical guarantees
- ≥ 2 pattern-detection front-ends (MInference + FlashPrefill) working end-to-end
- PPL regression ≤ +1% on `wikitext-2` for any front-end shipped
- NIAH retention ≥ 95% at 128K for any front-end shipped
- Documentation: `documentation/BLOCK-SPARSE-ATTENTION.md` covering kernel ABI, front-end protocol, per-model results

## Cross-cutting work tracked under this spec

- `documentation/sparse-attention-patterns/<model>.json` — calibration sidecars (shared with spec 031)
- `BenchmarkSignpost` phases: `bsa_compact`, `bsa_pass1`, `bsa_pass2`, `bsa_merge`
- `MLX_BSA_DEBUG=1` env var for kernel-level tracing

## What this spec deliberately does NOT do

- **No fused QKV projection + sparse attention.** Issue [#115](https://github.com/ekryski/mlx-swift-lm/issues/115) covers QKV fusion; orthogonal.
- **No per-token-quantized K.** TurboQuant variant comes in phase 8 once base kernel is stable.
- **No decode-time pattern detection.** Decode-side selection is spec 034, which consumes this kernel.
- **No paged-attention compatibility in V1.** [#127](https://github.com/ekryski/mlx-swift-lm/issues/127) tracks paged kernel separately; integration is a future spec.
