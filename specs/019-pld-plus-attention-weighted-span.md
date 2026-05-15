# 019 — PLD+ attention-weighted span selection

**Status:** 🚧 Phase 1 scaffold landed ([PR #148](https://github.com/ekryski/mlx-swift-lm/pull/148) — selector protocol + cosine helper). Phase 2 (per-model conformance + iterator integration) **not started.**
**Branch:** Phase 1 merged via PR #148; Phase 2 gets a fresh branch off alpha.
**Depends on:** spec 013 (n-gram iterator). Composes with multi-candidate (`MLX_NGRAM_MULTI_CANDIDATE`).

## Problem

Our current `proposeDraft` picks the candidate continuation by either **most-recent occurrence** (legacy) or **most-frequent first token** (multi-candidate). Neither uses the model's own state — they're purely token-frequency heuristics. PLD+ (Singh et al., 2024, [arXiv:2412.01447](https://arxiv.org/abs/2412.01447)) shows that **using the target model's own attention to score candidate spans** picks better drafts than frequency alone. The reported gain is measurable on input-grounded tasks (summarisation, QA, code edit) — exactly the regime where our PLD wins.

The mechanism PLD+ uses: a small set of pre-identified **"induction heads"** (attention heads that perform prefix matching and copying — see Olsson et al., "In-context Learning and Induction Heads") at specific layers. At each draft position, aggregate the attention scores from those heads across the candidate span's *first token* in the prompt. The candidate whose first prompt-position has the highest aggregated attention wins.

PLD+ also explores a **hidden-state similarity** alternative: cosine similarity between the layer-L hidden state at position `j-1` (just before each candidate) and the layer-L hidden state at the current decode position `t-1`. Doesn't require attention plumbing; works at early-to-mid layers.

## Design

### 1. Hidden-state similarity (Phase 1 — easier, model-agnostic)

```swift
public protocol AttentionAwareSpanSelector {
    /// Score each candidate occurrence position. Higher = more likely the
    /// model is "thinking about" that occurrence next.
    func score(
        candidates: [Int],            // candidate first-token positions in lookup history
        currentPos: Int,              // last accepted position
        hiddenStates: [Int: MLXArray] // layer id → [seq_len, hidden_dim]
    ) -> [Float]
}

public struct HiddenStateCosineSelector: AttentionAwareSpanSelector {
    public let layerId: Int
    // PLD+ ablation: layers 9-13 work best across model sizes.

    public func score(...) -> [Float] {
        let hl = hiddenStates[layerId]!
        let h_t = hl[currentPos - 1, .all]
        return candidates.map { j in
            let h_j = hl[j - 1, .all]
            return cosineSimilarity(h_j, h_t)
        }
    }
}
```

The iterator already runs forward passes; we just need to **capture** the hidden state at the configured layer at each step. Add a hook that the model can opt into:

```swift
public protocol HiddenStateCapture {
    /// If non-empty, the model writes hidden states from these layer IDs
    /// into a side channel that the iterator reads after each forward.
    var captureLayerIDs: Set<Int> { get set }
    func consumeCapturedHidden() -> [Int: MLXArray]
}
```

Each model in `Libraries/MLXLLM/Models/*` already has the per-layer activations on its forward path; we just need to optionally tap them. dflash-mlx already does this (`captureLayerIDs` in `DFlashTargetModel`); we'd port the same pattern.

### 2. Induction-head attention scoring (Phase 2 — model-specific)

Phase 1 scores candidates with cosine on hidden states. Phase 2 ports the actual PLD+ idea:

```swift
public struct InductionHeadAttentionSelector: AttentionAwareSpanSelector {
    public let inductionHeads: [(layer: Int, head: Int)]  // pre-identified for this model
    // PLD+ ablation: 379 heads in Vicuna-7B; top 50 work best.

    public func score(...) -> [Float] {
        // Aggregate (max, not sum) over selected heads of the attention
        // weight from `currentPos` toward each candidate's first token.
        ...
    }
}
```

This needs a hook into `MLXFast.scaledDotProductAttention` that exposes per-head attention probabilities — currently it returns only the SDPA output. Two paths:

- **Path A** — duplicate the attention computation in CPU-readable form for the configured heads only. Costly but localised; only fires for the few selected heads.
- **Path B** — use the existing FlashAttention-style fused kernel and add an optional `output_weights: MLXArray?` argument. Requires MLX-level changes; out of scope for this spec.

Path A first; defer Path B to a follow-up.

### 3. Induction-head identification

PLD+ identifies induction heads via a one-shot offline analysis: generate outputs on a calibration set, find which input positions could have been "copied", check which heads attend to those positions. The output is a `(layer, head)` set per model.

Ship a tool: `swift run mlx-find-induction-heads --model qwen3-9b --calibration ngram-sweep-prompts/qa-requote --output ~/.cache/mlx-swift-lm/induction-heads/qwen3-9b.json`. Run once per model. The result file lives in `Resources/induction-heads/{model}.json` and ships with the package for the supported model set.

### 4. Integration with multi-candidate

When `useMultiCandidate` produces multiple candidate first-tokens, today's tiebreak is recency. With PLD+ enabled (`MLX_NGRAM_PLDPLUS=1`), tiebreak becomes the attention-aware score:

```swift
// Inside NGramLookup.proposeDraft, when multiple candidates exist:
let scored = candidates.map { c in (c, selector.score(c, ...)) }
let best = scored.max(by: { $0.1 < $1.1 })!
```

When PLD+ disabled, fall through to current behaviour. When PLD+ enabled but the model has no induction-head map, fall back to `HiddenStateCosineSelector` with a default layer (layer 11 — middle of most pre-trained transformers per PLD+'s ablation).

## Implementation phases

1. **Phase 1** — `HiddenStateCosineSelector` + `HiddenStateCapture` protocol + Qwen 3 / Gemma 4 / GPT-OSS conformance. Behind `MLX_NGRAM_PLDPLUS=1`.
2. **Phase 2** — `InductionHeadAttentionSelector` + offline tool + bundled head maps for the supported model set.
3. **Phase 3** — Auto-tuning: select the best layer per model from a small calibration step. Requires Phase 1's hidden-state capture.

## Expected impact

PLD+ paper reports ~20-30% improvement in accept rate on Vicuna-7B over plain PLD on summarisation / QA. Our setting is similar (multi-candidate is already on; PLD+ is the next step up the ladder). Expected accept-rate lift on input-grounded prompts: 30-40% → 50-60% region, which on Gemma 4 26B A4B should turn today's +25% speedup into +35-45%.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/AttentionAwareSpanSelector.swift` (new) | Protocol + cosine + induction-head impls. |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | Wire selector into `proposeDraft`. |
| `Libraries/MLXLLM/Models/*` | Hidden-state capture conformance per supported model. |
| `Sources/Tools/MLXFindInductionHeads/` (new) | Offline calibration tool. |
| `Resources/induction-heads/*.json` (new) | Per-model induction-head maps. |
| `Tests/MLXLMTests/PLDPlusTests.swift` (new) | Selector unit tests. |

## Out of scope

- Model-internal Path B (FlashAttention exposing attention weights). Plumbing through MLX is a separate project.
- Self-attention-distance heuristics (e.g. Lookahead Decoding). PLD+ is sufficient for our regime.

## References

- [PLD+: Accelerating LLM inference by leveraging Language Model Artifacts (arXiv:2412.01447)](https://arxiv.org/abs/2412.01447)
- [In-context Learning and Induction Heads (Anthropic, 2022)](https://arxiv.org/abs/2209.11895) — definition + behaviour of induction heads.
- [SuffixDecoding (NeurIPS 2025, arXiv:2411.04975)](https://arxiv.org/abs/2411.04975) — frequency-based scoring as an alternative to attention-based; complementary, not competitive.
