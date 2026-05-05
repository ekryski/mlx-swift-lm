# 028 — Quadratic / chunkwise WY GatedDeltaNet prefill parallelization

**Status:** spec, exploratory (lower priority — research-grade, prior attempt regressed)
**Branch:** new branch off alpha
**Depends on:** none directly, but [#123](https://github.com/ekryski/mlx-swift-lm/issues/123) (adaptive evalInterval) covers complementary tuning
**Origin:** [`fused-gdn-kernel-prefill-regression-analysis.md`](/Users/eric/Development/personal/sam/planning/performance-notes/fused-gdn-kernel-prefill-regression-analysis.md), [`moe-prefill-decode-bottleneck-analysis.md`](/Users/eric/Development/personal/sam/planning/performance-notes/moe-prefill-decode-bottleneck-analysis.md) §11

## The insight

GatedDeltaNet (GDN) layers in Qwen 3.5 / 3.6 / Nemotron-H / Jamba run their prefill **sequentially** — each token's recurrent state depends on the previous token's, so the recurrence can't be parallelized via the obvious approach. **GDN prefill dominates total prefill time on these models.** Decode tok/s is already competitive (~62–75 tok/s on Qwen 3.5 35B-A3B); prefill is the open bottleneck.

Two known parallelization techniques exist in the literature for linear recurrences:

1. **Quadratic attention reformulation** — for short-to-medium contexts, the recurrent SSM update can be reformulated as a matmul-style attention computation. Trades recurrence-depth for matmul-width. Fundamentally O(T²) work but parallelism-friendly. The Flash-Linear-Attention codebase has reference implementations.

2. **Chunkwise WY representation** — split the recurrence into chunks; within each chunk, use the Woodbury-Young identity to express the state evolution as a matmul-friendly form; chain chunks sequentially. Hybrid of (1) and the sequential approach: O(T·chunk) work but chunks parallelize internally.

A previous quadratic-attention experiment **regressed** at `Dk=128` (commit reference in `fused-gdn-kernel-prefill-regression-analysis.md` — not because the math is wrong, but because the Metal kernel was naive). A purpose-built fused Metal quadratic kernel + chunked WY hybrid might still recover the prefill speedup.

## Why this is "research"

Three things make this exploratory rather than a clear-win optimization:

1. **Prior attempt failed.** The doc explicitly notes the previous quadratic-attention experiment regressed. We don't know whether the regression was kernel-tuning, numerical precision, or fundamental — the analysis document was inconclusive.

2. **The work is open-ended.** A "fused Metal quadratic GDN kernel" is a multi-month research project, not a bounded engineering task. It involves kernel design, numerical-stability validation against the sequential reference, chunked-WY math derivation, and per-model tuning.

3. **Projected impact is wide-ranging.** Per `moe-prefill-decode-bottleneck-analysis.md`, the upside ranges from **5× to 15× prefill on Qwen 3.5** if it works. The downside is "doesn't work and we spent weeks." Classic research bet.

## Design

### Approach 1 — Quadratic reformulation (short context only)

For context lengths ≤ ~2K, fully materialize the GDN recurrence as `O(T²)` matmuls:

```
Sequential GDN:
  s_{t+1} = (A_t * s_t) + (β_t * v_t * k_t^T)        [recurrent, O(T·D²)]
  o_t     = q_t * s_t                                  [O(D²) per token]

Quadratic equivalent (for context ≤ T_thresh):
  M = lower-triangular gating mask × A products
  S = build full state-history tensor [T, D, D]
  o = q ⊛ S                                            [O(T²·D²) work]
```

Trade-off: O(T²) work but trivially parallelizable. Wins for `T·D < threshold` where the parallelism dominates the extra work.

### Approach 2 — Chunkwise WY (any context)

Split the T-token sequence into chunks of size C (e.g., C=64 or C=128). Within each chunk, use the WY representation to express the chunk's state evolution as a sequence of matmuls. Chain chunks sequentially.

Per-chunk work is O(C²·D + C·D²) but parallelizes inside the chunk. Total work is O(T/C × C·D²) = O(T·D²) — same complexity as sequential, but with much better wall-clock parallelism for large C.

This is the technique used in production Flash-Linear-Attention for Mamba and friends.

### Approach 3 — Hybrid (recommended starting point)

Quadratic for `T ≤ T_thresh` (~2K), chunked-WY for `T > T_thresh`. Switch dispatched per-layer based on input length.

## Implementation phases

1. **Phase 1 — Reference Python port.** Port the Flash-Linear-Attention chunkwise-WY GDN kernel from PyTorch to a JAX/MLX-Python reference. Verify per-token equivalence with the sequential MLX-Swift reference at fp32 precision. ~2 weeks. Goal: have a working algorithm we can compare against.

2. **Phase 2 — Naive Metal port.** Port the chunkwise-WY algorithm to a Metal kernel using straightforward `simdgroup_matrix` calls. Optimize for correctness first; expect this to be at parity with the sequential path or worse. ~3 weeks. Goal: prove the algorithm runs on MLX-Metal.

3. **Phase 3 — Fused-kernel optimization.** Combine the chunk's gate, A_t, β_t computation with the WY matmul into a single dispatch. Tune `C` (chunk size), warp/threadgroup layout. ~3 weeks. Goal: prefill speedup over sequential.

4. **Phase 4 — Quadratic mode + dispatch logic.** Add the short-context quadratic path. Add per-layer length-based dispatch. ~1 week. Goal: hybrid path delivers the projected speedup at all context lengths.

5. **Phase 5 — Cross-model integration.** Verify on Qwen 3.5 / 3.6, Nemotron-H, Jamba. Each may have slightly different state shapes; per-model kernel variants. ~1–2 weeks per model. Goal: prefill-bottleneck removed on the supported hybrid model zoo.

## Expected impact

If approach 3 works as projected:
- **+5× to +15× prefill** on Qwen 3.5 family (the dominant prefill bottleneck per `moe-prefill-decode-bottleneck-analysis.md`)
- **Comparable on Nemotron-H** — same architecture, same bottleneck
- **No decode change** — decode is already at parity (T=1 is just one chunk)

If approach 3 regresses again:
- We learn whether the prior failure was kernel-tuning (recoverable) or fundamental (unsolved research problem)
- The reference Python port is reusable for further experiments

## Risks

1. **Numerical precision under bf16.** GDN's gating involves `exp` of negative values that decay state; bf16 can lose enough precision in long chains that the chunkwise approximation diverges from sequential. Phase 1's per-token equivalence test must be tight (tolerance < 1e-5 in fp32; relaxed but bounded in bf16). The dflash-mlx codebase uses fp32 accumulators on the rollback kernel for this reason; same trick applies here.

2. **Memory pressure.** Chunkwise WY needs working buffers of size `[B, C, D, D]` per layer per chunk. At C=128, D=128, that's 16M elements × 2 bytes = 32 MB per chunk per layer. For 30-layer GDN models at long context this adds up. Cap C adaptively based on memory budget.

3. **Per-model kernel variants.** Qwen 3.5 GDN, Nemotron-H "cascade", Jamba partial-Mamba — each has slightly different state shapes. Phase 5 may turn into 3 separate kernel variants, each needing its own tuning pass.

4. **Apple Metal `simdgroup_matrix` constraints.** The matmul dimensions need to match Metal's matrix-engine tile sizes (8×8 on M1, 16×16 on M5+). If GDN's state dim doesn't align cleanly, padding overhead may eat the parallelism win.

## Files touched

| File | What |
|---|---|
| `tools/gdn-reference-python/` (new) | Phase 1 Python reference port |
| `Sources/Cmlx/mlx-generated/metal/gdn_chunkwise.metal` (new) | Phase 2-3 Metal kernel |
| `Libraries/MLXLLM/Models/GatedDelta.swift` | Dispatch logic (sequential / chunkwise / quadratic) |
| `Libraries/MLXLLM/Models/Qwen35.swift` | Per-model integration |
| `Libraries/MLXLLM/Models/Qwen3Next.swift` | Per-model integration |
| `Libraries/MLXLLM/Models/NemotronH.swift` | Per-model integration |
| `Tests/MLXLMTests/GdnEquivalenceTests.swift` (new) | Phase 1 numerical-equivalence regression test |
| `benchmarks/notes/gdn-prefill-2026-MM-DD.md` (new) | Phase-by-phase results |

## Why this is Tier 4

XL scope, multi-month research, **prior attempt regressed**. The projected upside is genuinely large (5–15× prefill on Qwen 3.5 family), but research bets at this scale should only be made when the high-confidence work has stabilized.

Specifically: spec-decode (specs 013–025) targets 2–4× decode wins on the same model families with phase-1 scaffolds already landed. Land those first; if Qwen 3.5 prefill is still the dominant complaint after spec-decode is stable, take the research bet on 028.
