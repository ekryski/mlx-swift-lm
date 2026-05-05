# 026 — Profile-guided Morton-order expert weight reordering

**Status:** spec, exploratory (lower priority — experimental, not validated)
**Branch:** new branch off alpha
**Depends on:** none — pure offline + sanitize() work
**Origin:** [`sam/planning/performance-notes/morton-order-expert-reorder-spec.md`](/Users/eric/Development/personal/sam/planning/performance-notes/morton-order-expert-reorder-spec.md), [`option-d-l2-prefetching-analysis-2026-04-06.md`](/Users/eric/Development/personal/sam/planning/performance-notes/option-d-l2-prefetching-analysis-2026-04-06.md)

## The insight

`SwitchLayers.sortThreshold = 128` sets the boundary where MoE expert dispatch switches between sorted and unsorted modes. Below the threshold, **the unsorted path costs ~25% extra at T=128 prefill** because experts called in random order have poor L2 locality on their quantized weight tiles. Above it, the sort overhead dominates. There's no setting that wins everything.

Profile-guided reordering eliminates the trade-off: at calibration time, record which experts are co-selected together (per-token expert pairs / triples), then **permanently reorder the expert weights at `sanitize()` time** so frequently co-selected experts are physically adjacent in memory. The sort overhead becomes irrelevant because the layout itself encodes the locality, and L2 reuses across consecutive expert calls happen naturally.

This is offline work — zero runtime cost. The win is permanent for any deployment using the reordered checkpoint.

## What "Morton-order" means here

Morton order (Z-order curve) is a specific space-filling curve that interleaves bit-level coordinates of indexed data. Adapted to expert reorder: treat each expert's weights as living at a 2D address (expert ID, weight tile), then permute the expert IDs so the linearized memory layout is Morton-ordered relative to the expected co-selection graph.

In practice, it doesn't have to be Morton specifically — what we want is *some* spatial locality preservation under the realistic co-selection distribution. Spectral graph layout (Fiedler ordering on the co-selection graph) is the principled alternative; greedy nearest-neighbor merging is the simple alternative. All three are candidates for the calibration step.

## Design

### 1. Calibration pass

Run a representative corpus (~10K tokens across summarization, code, chat, agent loops) through the model at inference, dumping the full `(layer, token, selected_experts)` tuple per step. Output: a co-selection matrix `C[expert_i, expert_j]` per layer (count of how often experts i and j fire on the same token).

### 2. Permutation generation

For each MoE layer independently:
1. Build the co-selection graph (nodes = experts, edge weights = co-selection counts).
2. Run a layout algorithm to assign linear positions: candidates are
   - Greedy: start with the most-frequently-fired expert, append next-most-correlated, repeat.
   - Spectral: Fiedler vector of the graph Laplacian gives a 1D coordinate per expert.
   - Morton: explicit Z-order on a 2D grid of co-selection clusters.
3. Output a per-layer `permutation: [Int]` mapping original expert ID → new position.

Permutations are tiny (e.g., 32 ints per layer × 30 layers = ~1KB) and ship as a sidecar `.json` alongside the model weights.

### 3. `sanitize()` integration

When loading a model that has a permutation sidecar:
1. Read the per-layer permutation.
2. In the model's `sanitize()` hook, apply the permutation to the expert weight tensors (gate, up, down) for that layer.
3. Also apply to the gate / router weights so `argmax` over the gate output produces correctly-permuted expert IDs.
4. Cache the permuted weights as a new sanitized version so re-loads skip step 2.

`sanitize()` is the right hook because it runs once at model load, before any inference. The permuted weights are functionally equivalent — the permutation is invertible by definition, applied to both the gate and the experts.

### 4. Sort threshold tuning post-reorder

After reorder, the sort overhead becomes a much smaller relative cost (because there's less to sort meaningfully). Re-sweep the `SwitchLayers.sortThreshold` to find the new optimum — it may go up to ~256 or higher.

## Implementation phases

1. **Phase 1 — Calibration harness.** Write a standalone tool that runs a corpus through a target model and dumps the `(layer, token, experts)` tuples. Output: per-layer co-selection matrices saved as `.npy` or `.json`. Run on Qwen3.5-35B-A3B and Gemma4-26B-A4B (the two MoE models with concrete prefill cliffs).

2. **Phase 2 — Permutation generator.** Three algorithm options (greedy, spectral, Morton). Run all three; pick the best per-layer empirically (graph-conductance metric). Output: per-model `permutations.json` sidecar.

3. **Phase 3 — `sanitize()` hook.** Per-model integration in `Qwen35MoE.swift` and `Gemma4.swift` MoE blocks. Detection: if a `permutations.json` exists alongside the checkpoint, apply it; otherwise no-op. Add a CLI flag (`--ignore-expert-permutation`) for A/B.

4. **Phase 4 — Bench + threshold re-tune.** Measure prefill at T=128, T=512, T=1024, T=4096 against the unsorted-path baseline. Re-sweep sort threshold post-reorder.

## Expected impact

- **+25% prefill at T=128** on MoE models (recover the unsorted-path penalty entirely).
- **+0–5% at T≥1024** where the sorted-path baseline already wins.
- **+0–3% at decode** (T=1, single-expert calls per layer — minor L2 benefit on the expert weight tile being adjacent to neighbors that fire often).
- **Zero runtime cost** — all work is offline.

## Risks

1. **Calibration corpus may not generalize.** Co-selection patterns are workload-dependent. A corpus that overrepresents one task (e.g., all summarization) may produce a permutation that loses on others.
2. **Routing entropy may be too high.** If experts are nearly uniformly co-selected (low mutual information), no permutation helps and the work is wasted. This is the load-bearing risk — needs measuring before committing to phase 3.
3. **Permutation sidecar workflow.** Models distributed without our sidecar miss the optimization. Either ship a tool that generates the sidecar from any HF checkpoint, or vendor permutations for the popular models.

## Files touched

| File | What |
|---|---|
| `tools/expert-calibration/main.swift` (new) | Phase 1 calibration harness |
| `tools/expert-permutation/main.swift` (new) | Phase 2 permutation generator (greedy/spectral/Morton) |
| `Libraries/MLXLLM/Models/Qwen35MoE.swift` | sanitize() integration for permutation application |
| `Libraries/MLXLLM/Models/Gemma4.swift` | sanitize() integration |
| `Libraries/MLXLLM/Models/SwitchLayers.swift` | post-reorder sort threshold tuning |
| `benchmarks/notes/expert-reorder-2026-MM-DD.md` (new) | results across phases |

## Why this is Tier 4

Real win, but **per-model**, **per-checkpoint**, and **per-corpus** — every supported MoE model + every HF checkpoint variant + any deployment with sufficiently different traffic patterns needs its own calibration. That's a lot of moving parts for a +25% prefill gain at one specific T size.

The speculative-decoding work (specs 013–025) targets 2–4× wins on entire model families with no per-checkpoint setup. Defer 026 until those have stabilized and we're hunting for the next 10% on MoE prefill specifically.
