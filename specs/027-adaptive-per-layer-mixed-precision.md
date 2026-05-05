# 027 — Adaptive per-layer mixed-precision quantization framework

**Status:** spec, exploratory (lower priority — experimental, generalizes existing ad-hoc work)
**Branch:** new branch off alpha
**Depends on:** none — pure infrastructure work; pairs naturally with [issue #74](https://github.com/ekryski/mlx-swift-lm/issues/74) (Unsloth UD-MLX) and finding from [`lm-head-quantization-analysis-2026-04-08.md`](/Users/eric/Development/personal/sam/planning/performance-notes/lm-head-quantization-analysis-2026-04-08.md)
**Origin:** [`lm-head-quantization-analysis-2026-04-08.md`](/Users/eric/Development/personal/sam/planning/performance-notes/lm-head-quantization-analysis-2026-04-08.md) §"Per-Layer Adaptive Quantization", [issue #74](https://github.com/ekryski/mlx-swift-lm/issues/74)

## The insight

The repo already supports per-layer mixed-precision quantization in **two places**, but neither is general:

1. **Unsloth Dynamic 2.0 (`*-UD-MLX-*bit` checkpoints)** — different bit-widths assigned to different modules within a single checkpoint. Tracked as [issue #74](https://github.com/ekryski/mlx-swift-lm/issues/74) for correctness/loading; concrete motivator `unsloth/Qwen3.6-27B-UD-MLX-4bit`.
2. **Gemma 4 26B** — the `mlx-community/gemma-4-26b-a4b-it` checkpoint already ships with 8-bit MLP / 4-bit attention from Hugging Face. The model loader handles the per-layer bit-widths because the metadata is in the safetensors header.

Beyond these two ad-hoc paths, the codebase has no first-class concept of "per-layer bit-width recipes." The architecture is: one global quantization config, applied uniformly. That precludes:

- **Quantizing only the LM head** ([per `lm-head-quantization-analysis-2026-04-08.md`](/Users/eric/Development/personal/sam/planning/performance-notes/lm-head-quantization-analysis-2026-04-08.md): Int4 LM head is **20.8× faster** than FP16 on Gemma4's 262K vocab — 30 ms → 1.5 ms per token).
- **Quality-vs-speed tradeoff knobs** like "4-bit attention + 8-bit MLP + Int4 LM head" as a recipe users can opt into.
- **Recipe-driven user customization** — without writing a custom checkpoint sanitize hook.

This spec generalizes the two ad-hoc paths into a recipe-driven framework.

## Design

### 1. Recipe schema

```swift
public struct QuantizationRecipe: Codable {
    /// Bit-width per module-path glob pattern. First match wins.
    /// Glob examples: "lm_head", "*.attention.*", "*.mlp.gate_proj", "embed_tokens"
    public let layerBits: [(pattern: String, bits: Int)]

    /// Group size per pattern. Optional — defaults to 64.
    public let layerGroupSize: [(pattern: String, groupSize: Int)]

    /// Modules that should not be quantized at all.
    public let skipPatterns: [String]
}
```

Recipes ship as JSON sidecars (e.g., `recipe-int4-lm-head.json`) or are embedded in safetensors headers per the Unsloth pattern.

### 2. Sanitize() integration

The model's `sanitize()` hook applies the recipe at load time:

```swift
extension Module {
    public func applyQuantizationRecipe(_ recipe: QuantizationRecipe, weights: inout [String: MLXArray]) {
        // For each weight tensor:
        //   1. Find first matching pattern in recipe.layerBits
        //   2. Quantize to that bit-width with the matching group size
        //   3. Store the quantized weight + scales/zeros under the expected MLX naming
        //   4. Skip if matched by a skipPattern
    }
}
```

This is the load-time equivalent of writing a custom per-model sanitize hook for each recipe — but instead of N hooks, there's one generic applier driven by the recipe.

### 3. Recipe library

Ship a starter library of recipes:

| Recipe | Description | Use case |
|---|---|---|
| `int4-lm-head` | bf16 model + Int4 LM head only | 20× LM head speedup on bf16 models with no quality impact (per the analysis doc) |
| `attn4-mlp8` | 4-bit attention, 8-bit MLP | Quality-preserving quantization where attention is more sensitive |
| `attn8-mlp4` | 8-bit attention, 4-bit MLP | Speed-leaning |
| `unsloth-ud-2.0` | Reproduces Unsloth's UD-MLX layout from a generic 4-bit checkpoint | Equivalence with #74 |

### 4. CLI surface

```
./scripts/benchmark.sh --model gemma-4-e2b --recipe int4-lm-head ...
```

Defaults to whatever the model's safetensors header says, or "uniform 4-bit" if no recipe is specified.

## Implementation phases

1. **Phase 1 — Recipe schema + library.** JSON parsing + 4 starter recipes. ~150 lines.

2. **Phase 2 — `applyQuantizationRecipe` generic implementation.** Pattern matching + per-tensor quantization dispatch. Test against a known-equivalence checkpoint (load `mlx-community/gemma-4-e2b-it-4bit` with a "uniform 4-bit" recipe; verify byte-equivalent to current loader output). ~300 lines.

3. **Phase 3 — Sanitize() integration.** Wire into per-model sanitize hooks for Gemma 4, Qwen 3.5, GPT-OSS first. Verify Unsloth UD-MLX checkpoints load correctly via the `unsloth-ud-2.0` recipe — closes [issue #74](https://github.com/ekryski/mlx-swift-lm/issues/74) as a side effect.

4. **Phase 4 — Bench across recipes.** PPL + tok/s matrix for the 4 starter recipes × 4 representative models. Document quality-vs-speed tradeoffs in `benchmarks/notes/recipes-2026-MM-DD.md`. Determine which recipes are worth promoting as user-facing defaults.

## Expected impact

- **Closes [#74](https://github.com/ekryski/mlx-swift-lm/issues/74)** as a side effect (Unsloth UD-MLX support comes for free once the recipe framework exists).
- **+20× LM head decode** for users who opt into `int4-lm-head` on bf16 models — only matters for bf16-weight users (small audience but huge impact for them).
- **Recipe-driven A/B for quantization research** — turns "what's the right bit allocation?" from a checkpoint-modification chore into a one-flag experiment.
- **Quality-preserving quantization paths** like `attn4-mlp8` get a clean home — currently buried in per-checkpoint sanitize hooks.

## Risks

1. **Pattern-matching ambiguity.** Glob patterns can have multi-match cases ("first match wins" handles it but recipes are easy to write wrong). Mitigation: ship a `--validate-recipe` mode that enumerates exactly which weights match each pattern.

2. **Group size mismatches.** Different layers may have shapes that don't divide cleanly by a uniform group size. The schema supports per-pattern group sizes, but recipe authors need to know the constraints. Mitigation: validation step that pre-checks shape compatibility before quantizing.

3. **Quality regression isn't free.** Mixed-precision recipes are quality trade-offs by definition. Phase 4's bench needs to flag any recipe that's a Pareto loser (slower AND worse quality than uniform).

4. **Recipe library maintenance.** Each new model architecture may need recipe-pattern updates if module naming differs. Track in `recipes/README.md`.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/QuantizationRecipe.swift` (new) | Recipe schema + applier |
| `recipes/int4-lm-head.json` (new) | Recipe library |
| `recipes/attn4-mlp8.json` (new) | |
| `recipes/attn8-mlp4.json` (new) | |
| `recipes/unsloth-ud-2.0.json` (new) | |
| `Libraries/MLXLLM/Models/Gemma4.swift` | sanitize() integration |
| `Libraries/MLXLLM/Models/Qwen35.swift` | sanitize() integration |
| `Libraries/MLXLLM/Models/GPTOSS.swift` | sanitize() integration |
| `scripts/benchmark.sh` | `--recipe <name>` CLI flag |
| `benchmarks/notes/recipes-2026-MM-DD.md` (new) | Phase 4 bench results |

## Why this is Tier 4

Lower priority because: (a) it's a framework, not an end-user feature — value depends on what recipes ship on top; (b) the headline win (Int4 LM head) only matters for bf16-weight users which is a small audience; (c) closing #74 is real but Unsloth UD-MLX support could ship as a one-off custom sanitize hook in less time than the full framework. The framework wins long-term, but spec-decode (specs 013–025) gets bigger numbers per engineer-week.

Worth doing once spec-decode stabilizes and we want to ship quality-vs-speed knobs as a product feature.
