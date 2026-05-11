# 031 — Vertical-slash sparse prefill attention

**Status:** spec, ready (low-risk first deliverable in the sparse-attention family)
**Branch:** new branch off alpha
**Depends on:** none (uses existing `MLXFast.scaledDotProductAttention`)
**Origin:** Research review 2026-05-08; subset of MInference / FlashPrefill family that is implementable without a custom Metal kernel
**Related:** [032](032-speculative-prefill.md) (composes), [033](033-block-sparse-sdpa-metal.md) (supersedes for full FlashPrefill coverage)

## The insight

[MInference](https://github.com/microsoft/MInference) (NeurIPS'24) and [FlashPrefill](https://arxiv.org/abs/2603.06199) (Fan'26) both observe that long-context attention matrices empirically decompose into three patterns: **A-shape / streaming** (attend to first N sink tokens + recent window), **vertical-slash** (a few "vertical stripe" tokens get attended-to globally + a diagonal "slash" recency band), and **block-sparse** (square blocks of high attention scattered through the matrix). MInference assigns one pattern per head offline and dispatches the matching kernel online.

The first two patterns — A-shape and vertical-slash — are **literally a small dense region plus a diagonal band**. They can be computed with two ordinary dense-SDPA calls and a max-merge. **No block-sparse kernel needed.** Only the third pattern (random block-sparse) requires the Metal kernel work in spec 033.

Empirically, A-shape + vertical-slash heads cover **~70% of attention heads** in modern transformer models per the MInference paper. So implementing only those two patterns captures most of the achievable sparsity gain without writing a new kernel.

## Why this is the right first step

1. **Zero kernel work.** Both patterns reduce to existing `MLXFast.scaledDotProductAttention` calls on sliced tensors plus a softmax-merge (logsumexp combine).
2. **Validates the infrastructure.** Building head-pattern dispatch, calibration data collection, and accuracy-gating harness is reusable for spec 033 and spec 034.
3. **Bounded effort.** Phase 1 is implementable in ~1 week with no external dependencies.
4. **Composes with spec 032.** Speculative prefill compresses 128K → 6.4K; vertical-slash on the residual 6.4K compounds the win.
5. **Falls back cleanly.** Any head where pattern detection fails routes to dense SDPA — zero quality risk for those heads.

## Design

### Pattern definitions

For prefill with Q ∈ [B, H, T, D] and K, V ∈ [B, H, T, D]:

- **A-shape (streaming):** head attends only to the first `s` "sink" tokens (the absolute prefix) and the last `w` "window" tokens. Replaces full O(T²) compute with O(T·(s+w)).
- **Vertical-slash:** head attends to a fixed sparse set of `v` "vertical" token indices `V_idx ⊆ [0, T)` plus a diagonal slash band of width `b` around each query position. Both `V_idx` and `b` are detected per-head via a calibration pass.
- **Dense (fallback):** unchanged dense SDPA.

### Compute path for A-shape

```
sink_K, sink_V = K[:, :, :s, :], V[:, :, :s, :]
recent_K, recent_V = K[:, :, -w:, :], V[:, :, -w:, :]
out = MLXFast.scaledDotProductAttention(
    queries=Q, keys=concat(sink_K, recent_K), values=concat(sink_V, recent_V), scale=...
)
```

Causal mask handled by truncating `recent_K`/`recent_V` to positions ≤ each query's index — implemented as a precomputed mask once per chunk, not per-query.

### Compute path for vertical-slash

Two SDPA calls + merge:

```
# Vertical: gather v selected indices once
V_K = K[:, :, V_idx, :]                  # [B, H, v, D]
V_V = V[:, :, V_idx, :]                  # [B, H, v, D]
out_v, lse_v = sdpa_with_lse(Q, V_K, V_V)

# Slash: diagonal band of width 2b+1 around each query position
slash_mask = abs(q_pos - k_pos) <= b
out_s, lse_s = sdpa_with_lse(Q, K, V, mask=slash_mask)

# Merge via log-sum-exp
out = merge_lse(out_v, lse_v, out_s, lse_s)
```

The `merge_lse` step combines two partial-attention outputs into the equivalent of one full softmax. Standard FlashAttention online-softmax math; small Swift utility, no kernel. (Note: `MLXFast.scaledDotProductAttention` returns the output but not LSE — phase 2 below addresses this.)

### Pattern detection (offline calibration)

Per model, run a small calibration corpus (~32 prompts × 4K tokens) and per-head measure:

```
sink_recall(s, w)    = sum of attention weights to (first s ∪ last w) / total
slash_recall(b)      = sum of weights within |q - k| ≤ b / total
vertical_recall(v)   = sum of weights to top-v columns by global magnitude / total
```

For each head, assign the pattern with the highest recall at the smallest budget that exceeds a threshold (default 0.95). Cache per-head assignments in a sidecar JSON keyed by `(model_id, head_idx)`.

### Pattern detection (online — phase 2)

MInference's "online" mode re-detects pattern per query block. FlashPrefill / FlexPrefill use a "block search" of mean(K) projections to detect vertical/slash per-query in O(T·H·D/B) work. Phase 2 ports this. Phase 1 ships with offline-only.

## Implementation phases

1. **Phase 1 — Calibration harness + offline pattern assignment + A-shape dispatch.** Build `scripts/calibrate-attention-patterns.swift` that runs a corpus through a model with attention-weight capture, computes per-head recalls, emits sidecar JSON. Add `SparsePrefillRouter` that consumes the sidecar and dispatches A-shape vs dense per-head. ~1 week. Goal: A-shape working on Gemma4 / Qwen3.5 with measurable prefill speedup at ≥ 8K context.

2. **Phase 2 — LSE-aware SDPA + vertical-slash dispatch.** Add `MLXFast.scaledDotProductAttentionWithLSE` (multi-repo: needs mlx, mlx-c, mlx-swift bumps — pattern same as TurboQuant kernel landings). Add `merge_lse` Swift utility. Wire vertical-slash path. ~2 weeks. Goal: two-pattern dispatch working at 16K+ context.

3. **Phase 3 — Online pattern detection.** Port FlashPrefill's mean(K) block-search. Eliminates calibration sidecar. ~1 week. Goal: zero-config sparse prefill for any model.

4. **Phase 4 — Cross-model rollout.** Calibrate Gemma4 / Qwen3.5 / Qwen3.6 / Mistral / GPT-OSS / Nemotron-H. Per-model accuracy gate (PPL within +0.5% on `wikitext-2`, NIAH retrieval ≥ 95%). ~1 week per model.

## Expected impact

Per the MInference paper, A-shape + vertical-slash captures ~70% of heads. Combined with the patterns' own per-head speedup at long context:

- **8K context:** 1.3–1.8× prefill (small win — overhead dominates)
- **32K context:** 2.5–4× prefill
- **128K context:** 4–7× prefill (dominant heads now in O(T·s+w) regime)

Speedup is bounded by remaining dense heads (~30% of compute). Spec 032 picks up that remainder via block-sparse for the final 27.78× FlashPrefill claim.

## Risk register

1. **Pattern detection mis-assigns a head → quality regression.** Mitigation: calibration gate (recall ≥ 0.95) plus runtime per-head fallback to dense if detected pattern's recall on the actual prompt drops below threshold. A small "watchdog" runs on every Nth chunk during prefill.

2. **`merge_lse` numerical drift on bf16.** Mitigation: do the LSE merge in fp32 (single scalar op per head, no perf cost).

3. **Sidecar sprawl.** Per-model JSON files. Mitigation: store under `documentation/sparse-attention-patterns/` and version with the model checkpoint id, similar to the TurboQuant codebook sidecar pattern.

## Acceptance criteria

- A-shape + vertical-slash dispatch wired, behind `MLX_SPARSE_PREFILL=1` env var initially
- Calibration sidecar for ≥ 2 models (Gemma4-26B-A4B, Qwen3.5-35B-A3B)
- Prefill speedup measured at 4K / 16K / 64K / 128K on calibrated models
- PPL regression ≤ +0.5% on `wikitext-2`; NIAH retrieval ≥ 95%
- Per-prompt sweep mode (spec 018 harness) extended with `--method sparse-prefill`
- Documentation: `documentation/SPARSE-PREFILL.md` covering calibration, per-model results, when to disable

## What this spec deliberately does NOT do

- **No block-sparse pattern.** That's spec 033 (needs Metal kernel).
- **No dynamic per-query pattern routing on TurboQuant path B.** Compose later if both ship.
- **No decode-side sparsification.** That's spec 034.
