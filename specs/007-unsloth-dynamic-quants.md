# Spec 007 — Unsloth Dynamic Quant (UD-MLX) compat

**Status:** draft
**Target:** Unsloth "Dynamic 2.0" MLX checkpoints — repos tagged `*-UD-MLX-*bit`. Concrete motivator: [`unsloth/Qwen3.6-27B-UD-MLX-4bit`](https://huggingface.co/unsloth/Qwen3.6-27B-UD-MLX-4bit). Generalizes to every Unsloth UD variant shipped for Qwen / Gemma / Llama / etc.
**Owner:** TBD
**Expected gain:** Correctness — load + run Unsloth UD checkpoints with no shape-mismatch errors. Secondary: lower memory footprint vs uniform 4-bit (2-bit for layers Unsloth deemed robust to more aggressive quant).

## Motivation

Unsloth's "Dynamic 2.0" quantization assigns **different bit-widths to different modules** within a single checkpoint. On MLX, these ship as `*-UD-MLX-4bit` repos where the headline "4-bit" is an average, not a uniform layer width. Observed on [`Qwen3.6-27B-UD-MLX-4bit`](https://huggingface.co/unsloth/Qwen3.6-27B-UD-MLX-4bit) during spec 001 benchmarks:

```
mismatchedSize(
    path: ["language_model", "model", "embed_tokens", "weight"],
    modules: [… "Qwen35TextModelInner", "QuantizedEmbedding"],
    expectedShape: [248320, 640],   // 4-bit: 5120 / 8 = 640 packed cols
    actualShape:   [248320, 1280])  // 2-bit: 5120 / 4 = 1280 packed cols
```

The checkpoint stores `embed_tokens` at 2-bit while the MLX Swift loader builds a 4-bit `QuantizedEmbedding` based on the default quantization block in `config.json`. The per-layer override exists in the config (at least for Gemma 4 26B A4B, which uses the same `PerLayerQuantization` plumbing and does work), but for Unsloth UD checkpoints **the perLayerQuantization isn't being threaded to `embed_tokens`**, and — once that's fixed — may well hit additional mismatches on `lm_head`, attention projections, or MoE routers that Unsloth also packs at non-default bit widths.

The infrastructure (`BaseConfiguration.PerLayerQuantization`, `Load.swift`'s `effectivePerLayerQuantization` lookup, per-model `sanitize(perLayerQuantization:)` hooks) already exists. What's missing is **(a) config-format parsing** for Unsloth's specific layout, **(b) coverage at every quantizable module construction site** (especially embeddings, not just Linears), and **(c) per-model path-remapping sanitize overrides** for the models Unsloth ships UD variants of.

## Non-goals

- Building new quantization kernels. Unsloth UD relies on the same mlx `quantized` primitive with `bits ∈ {2, 3, 4, 8}` that MLX Swift already supports.
- Reproducing Unsloth's quality-preserving layer-selection heuristics. We consume the shipped mix, we don't redesign it.
- Non-MLX Unsloth formats (GGUF, exl2, etc.) — those are handled elsewhere or out of scope here. Only `UD-MLX-*` repos are in scope.
- Matching the accuracy / calibration of Unsloth's Python loader bit-for-bit — load, correctness, and reasonable KLD are the gates. Perplexity numbers are a follow-up once load is working.

## Investigation (do first, before implementing)

1. **Dump the Unsloth config.** Pull `config.json` from `unsloth/Qwen3.6-27B-UD-MLX-4bit` and compare the `quantization` block against an MLX-community reference (`mlx-community/Qwen3.6-27B-4bit`). Key questions:
   - Is the mixed layout encoded as `{"model.embed_tokens": {"bits": 2, "group_size": 32}, ...}` inside the `quantization` dict, matching MLX Swift's existing `QuantizationContainer` decoder?
   - Or is it encoded differently (Unsloth-custom key names, e.g. `dynamic_quantization_overrides`, a separate top-level field, a quantization "map" keyed by index rather than path)?
2. **Enumerate the per-layer-quant entries.** Print every `(key, bits, groupSize)` pair in the Unsloth config. Categorize by module type: embedding / lm_head / q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj / router / expert.* / norm. Which categories deviate from the headline 4-bit? (Expect: embeddings at 2-bit, routers at 8-bit, some norms skipped — exact mix is what we're discovering.)
3. **Pick a second repo for cross-validation.** `unsloth/Qwen3.6-27B-UD-MLX-4bit` and one other UD-MLX repo from a different architecture (e.g. an Unsloth Gemma 4 UD-MLX if it exists, or a Llama-family UD-MLX) to make sure the config format is stable across architectures before we build the parser.
4. **Record findings** in a short doc (`benchmarks/notes/unsloth-ud-format-<date>.md`) so the next pass has it. Do this *before* writing any code.

Only after investigation: pick design path.

## Design (post-investigation — to be tightened once format is known)

### If Unsloth matches the existing `QuantizationContainer` format

Then the config parses correctly today and the gap is at module-construction time: `QuantizedEmbedding` (and any other quantizable module whose construction doesn't consult `PerLayerQuantization`) needs to be wired through the same `loadWeights` path that already works for `QuantizedLinear`.

Work:
- Audit `loadWeights(modelDirectory:model:quantization:perLayerQuantization:)` in [Load.swift](Libraries/MLXLMCommon/Load.swift) to confirm `QuantizedEmbedding`'s `group_size`/`bits` come from the per-layer map, not the container default.
- Add the `sanitize(perLayerQuantization:)` override on every model Unsloth ships UD variants of. At minimum: Qwen3.5/3.6, Gemma 4, Llama, Mistral. Mirror the Qwen3.5 pattern landed in spec 001 (strip `language_model.` prefix on the outer, remap fused keys on the inner).
- Extend the Qwen3.5 override from spec 001 to also remap quantized-embedding-specific keys if Unsloth uses different naming.

### If Unsloth uses a different config layout

Extend `QuantizationContainer` to parse the Unsloth form in addition to the MLX form. Unknown keys should be tolerated (log + ignore), so an unseen UD schema doesn't hard-fail.

Work:
- Add an `UnsloftUDQuantization` decoder path triggered by a marker key (TBD from investigation step 1) or by repo naming convention (less reliable; avoid).
- Translate to the canonical `[String: QuantizationOption]` dict, then continue down the existing code path. No changes past the parser.

### Either way — generalize the path-remap

Every model that lands a `sanitize(weights:)` transformation (fusion like spec 001, key renames, prefix strips) needs a matching `sanitize(perLayerQuantization:)`. Today only Gemma 4 (and, post-001, Qwen3.5) does this. Audit and extend:
- Grep for `sanitize(weights:)` overrides; for each, check whether it renames any `.weight` keys.
- If it does and the model isn't in `sanitize(perLayerQuantization:)`, add an override. Unsloth UD exposes every such gap because it's the only format that routinely uses path-keyed quantization beyond Gemma 4's 4-bit/8-bit split.

## Acceptance criteria

1. `unsloth/Qwen3.6-27B-UD-MLX-4bit` loads and produces coherent summarization output at ctx=1024 with `--quant 4bit-ud --kv none`.
2. Decode tok/s on `unsloth/Qwen3.6-27B-UD-MLX-4bit` (M1 Max, ctx=1024): **within ±5% of `mlx-community/Qwen3.6-27B-4bit`** under the same config. (Lower bit layers may trade compute for memory bandwidth — small deltas in either direction are OK; large regressions mean we're not hitting the GEMV fast path for the 2-bit Linears and need a follow-up.)
3. GPU peak memory on the UD variant is **demonstrably lower** than the uniform 4-bit mlx-community variant at the same context (confirms the mixed-bit layout is actually active).
4. KL divergence vs the `mlx-community/Qwen3.6-27B-4bit` baseline at a fixed ctx is comparable (no more than ~2× higher). This is not a correctness gate in the strict sense — Unsloth UD is lossier by design — but catches "we silently dropped a per-layer override and are running uniform 4-bit" errors.
5. One additional UD-MLX repo from a different architecture loads without mismatch errors. Picks the actual second test repo during investigation.
6. No regression on `mlx-community/Qwen3.6-27B-4bit` or any other uniform-quantized checkpoint (the perLayerQuantization override must short-circuit cleanly when there are no UD-specific entries).

## Measurement plan

```bash
# Baseline (no UD)
scripts/benchmark.sh --model qwen36-27b --quant 4bit \
    --method summarization --kv none --context 1024,4096,8192

# UD variant
scripts/benchmark.sh --model qwen36-27b --quant 4bit-ud \
    --method summarization --kv none --context 1024,4096,8192

# KLD check (add --kld once the KLD harness supports cross-variant comparison;
# otherwise, manually diff PPL on a fixed wikitext2 slice)
scripts/benchmark.sh --model qwen36-27b --quant 4bit-ud \
    --method wikitext2 --context 1024 --kld
```

Report peak GPU, decode tok/s, and PPL / KLD for each.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Unsloth ships config schema changes without notice | Version-detect from a header field inside the `quantization` block if one exists, else pin to a specific commit hash per tested repo. Fall back with a clear error if the schema looks new. |
| 2-bit Linears drop off MLX's GEMV fast path and regress decode | Measure first. If real, the mitigation is to dequantize 2-bit layers to 4-bit at load time (costs memory, restores speed) and surface an env flag to choose: `MLX_UD_DEQUANT_2BIT=1`. |
| UD variants exist for models whose MLX Swift port doesn't have a `sanitize(perLayerQuantization:)` override | Block the load with an actionable error message listing the missing override path, rather than crashing deep inside weight-shape comparison. |
| LoRA adapters trained against the UD variant target 2-bit layers | LoRA compat with UD is a follow-up — flag clearly if an adapter's target module paths hit UD-quantized weights. |
| Silent fallback to uniform 4-bit (overrides parsed but not applied) | Acceptance criterion #3 (peak-memory drop) catches this. |

## Out of scope / follow-ups

- **GGUF-via-MLX ingestion of Unsloth Dynamic GGUFs**: separate format, different loader stack. Different spec.
- **Quality evaluation of Unsloth's UD mixes vs uniform 4-bit** on standard evals (MMLU, HumanEval). Interesting but not a correctness gate for "does it load and run."
- **Auto-detection from repo-id**: do we special-case `*-UD-MLX-*` in the model factory? No — prefer config-based detection so out-of-convention repos still work. Revisit only if config parsing has too many false negatives.
- **VLM Unsloth UD variants** (vision towers): in-scope conceptually but add the LLM path first.

## Open questions

1. Does Unsloth's config declare per-layer quantization using the same `{"model.X.weight": {"bits": N, "group_size": G}}` shape as MLX-community / Gemma 4 mixed-quant, or something else? — **Answer via investigation step 1.**
2. Are any Unsloth UD-MLX layers stored at 3-bit? MLX supports `bits ∈ {2, 3, 4, 6, 8}` but not every combination has a fast GEMV kernel. If 3-bit appears, we need a perf follow-up.
3. Unsloth sometimes quantizes `lm_head` differently from `embed_tokens` even when they're tied. For tied-embedding models (most Qwen), do we need to detect a de-tied UD layout and suppress the tie?
