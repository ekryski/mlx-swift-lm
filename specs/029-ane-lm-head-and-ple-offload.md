# 029 — ANE-offloaded LM head + Gemma 4 PLE projection

- **Status:** spec, exploratory (lower priority — depends on spec 025's concurrent-execution measurement passing)
- **Branch:** new branch off alpha
- **Depends on:** [spec 025](./025-ane-gpu-concurrency-primitives.md) Phase 1 measurement harness passing (concurrent ANE+GPU execution achievable on Apple Silicon for this workload)
- **Origin:** [`m-series-architecture-neural-accelerator-support.md`](/Users/eric/Development/personal/sam/planning/performance-notes/m-series-architecture-neural-accelerator-support.md) §3a/§3b, [`ane-kernel-example-distilbert.md`](/Users/eric/Development/personal/sam/planning/performance-notes/ane-kernel-example-distilbert.md), [`lm-head-quantization-analysis-2026-04-08.md`](/Users/eric/Development/personal/sam/planning/performance-notes/lm-head-quantization-analysis-2026-04-08.md)

## The insight

Spec 025 builds the primitives + measurement harness for race-free ANE+GPU concurrent execution. Spec 021 uses those primitives to run a draft model on ANE while the target verifies on GPU (Mirror SD).

This spec uses the same primitives for a different class of workload: **per-layer projections that can run on ANE while the GPU starts the next layer's attention.** Two concrete candidates:

1. **LM head projection** — last-step computation per token. On Gemma 4's 262K vocab at FP16 it's ~30 ms per token. Per [`lm-head-quantization-analysis-2026-04-08.md`](/Users/eric/Development/personal/sam/planning/performance-notes/lm-head-quantization-analysis-2026-04-08.md), Int4 LM head on GPU is 1.5 ms — but on ANE it could overlap with the next-token forward, hiding the latency entirely.

2. **Gemma 4 E2B `perLayerModelProjection`** — runs once per forward pass to compute per-layer PLE inputs. While GPU runs the first layer's attention, ANE computes the PLE projection for layers 2…35. Hides the projection cost.

Both are bounded fixed-shape projections — the **ideal shape for Core ML**, which compiles its graph once and runs predictions repeatedly. Different from Mirror SD's draft-loop in workload but the same primitives.

## Why ANE for these specifically

ANE has three properties that make these workloads a fit:

1. **Compiled fixed-shape graph.** LM head and PLE projection have the same shape every call — `[B, T, hidden] @ [hidden, vocab]` for LM head; `[B, T, hidden] @ [hidden, layers·plDim]` for PLE. No KV cache, no variable seq length, no per-step routing. Core ML compiles once.

2. **Concurrent with GPU work.** While ANE runs LM head for token N, GPU can be doing forward for token N+1's prefill or the next layer's attention. The **wall-clock cost of LM head goes from "30 ms exposed" to "30 ms hidden behind GPU work that was happening anyway."**

3. **Massive vocab makes this winnable.** Gemma 4's 262K vocab is ~10× normal. The bigger the projection, the more there is to overlap. For models with smaller vocabs (Qwen at 152K) the win is smaller; not worth the integration cost.

## Why this is exploratory

Three reasons it's spec 029 not spec 022 or 023:

1. **Hard dependency on spec 025 measurement.** If Apple Silicon doesn't actually run ANE+GPU concurrently for this workload (decision point #2 in IMPLEMENTATION-PLAN), this spec dies. Don't start phase 1 here until spec 025 phase 1 passes.

2. **Per-model integration cost.** Each candidate workload (Gemma 4 LM head, Gemma 4 PLE, Qwen LM head, etc.) is its own integration. Not a universal optimization.

3. **Tooling is in-flight.** Core ML LLM Swift Package, ANE measurement infra, IOSurface-backed cross-device buffers — all of these are spec 021 / spec 025 prerequisites. Spec 029 reuses them.

## Design

### 1. LM head ANE-offload pattern

```swift
// Inside the model's forward(), at the LM head step:
let hiddenState = ...  // [B, T, hidden] from the last decoder layer

// Submit ANE prediction asynchronously
let lmHeadFuture = aneLMHead.predictAsync(input: hiddenState)

// Meanwhile, GPU is free to:
//   - finish any pending KV cache writes
//   - start prefill of the next chunk (if streaming)
//   - run any post-processing not on the LM-head critical path

// Await the result when the next token is needed
let logits = await lmHeadFuture.value  // [B, T, vocab]
```

The `aneLMHead` is a Core ML model wrapper produced from the safetensors LM head weights at model-load time. Compiled once, predicted many times.

### 2. Gemma 4 PLE projection ANE-offload pattern

```swift
// Inside Gemma4ModelInner.callAsFunction, after embedding the input tokens:
let h = embed(inputs) * embedScale  // GPU computes embedding

// Kick off PLE projection on ANE while GPU works on layer 0
let pleFuture = anePLEProjection.predictAsync(input: h)

// GPU runs first layer's attention (which doesn't need PLE input)
let layerOutput = layers[0](h, ...)

// PLE result is needed starting at layer 1
let pleInputs = await pleFuture.value
```

The PLE projection is a single dense layer: `[B, T, hidden] → [B, T, layers·plDim]`. Concrete shape on Gemma 4 E2B: `[B, T, 1536] → [B, T, 35·256]`. ~13M params; quick to compile and predict.

### 3. Core ML model construction at load time

At model load, after weights are decoded:
1. For each ANE-targetable projection (LM head, PLE, etc.), extract the weight tensor.
2. Build a Core ML model with that weight tensor as a constant.
3. Compile (Core ML caches the compiled artifact across loads).
4. Store the compiled `MLModel` reference on the LLM struct.

This adds ~1–3 seconds to first-time model load (Core ML compilation). Subsequent loads use the cache. The cost is amortized over millions of inference calls.

### 4. Race-free input/output handoff

Use the patterns from spec 025:
- **CPU → ANE input:** fresh `MLArray` per `predict()` call from the GPU-resident hidden state. Copy CPU-side or via IOSurface (TBD per spec 025 phase 2 measurement).
- **ANE → CPU output:** Core ML's prediction completion gives a fresh output buffer, retained until completion handler runs. Copy logits out before the scope ends.

## Implementation phases

1. **Phase 1 — Block on spec 025 phase 1 + 2.** Do not start phase 1 of this spec until spec 025's measurement harness passes (concurrent ANE+GPU execution achievable) and the buffer-lifecycle audit completes. Hard dependency.

2. **Phase 2 — LM head ANE wrapper for Gemma 4.** Build `ANELMHead` Swift class. Compile the LM head as a Core ML model at model load. Replace the GPU `Linear` call with the async ANE `predict`. Use spec 025's `ANEDraftLoop` patterns. ~1 week. **Measure: decode tok/s with vs without ANE-offloaded LM head on Gemma 4 E2B.** Gate is +5% or more decode improvement (smaller wins aren't worth the integration cost).

3. **Phase 3 — PLE projection ANE wrapper for Gemma 4 E2B.** Build `ANEPLEProjection` Swift class. Compile the PLE projection as a Core ML model. Kick off async at the start of `Gemma4ModelInner.callAsFunction`. ~1 week. **Measure: prefill + decode delta on Gemma 4 E2B.**

4. **Phase 4 — Generalize to other models (if phase 2/3 pass).** LM head ANE-offload for Qwen 3.5, GPT-OSS, etc. Each is a per-model integration. Skip models where the win is <5%.

## Expected impact

If Apple Silicon truly runs ANE+GPU concurrently for these workloads:
- **+5–15% per-token** on Gemma 4 E2B (LM head + PLE both offloaded)
- **+3–8% per-token** on other large-vocab models (LM head only)
- **No decode change on Qwen 3.5** (smaller vocab, weaker case)

If concurrency doesn't work (spec 025 phase 1 fails):
- This spec dies; the whole ANE-offload class of optimizations is closed off
- Spec 021 (Mirror SD) also dies — same root cause
- We fall back to GPU-only optimizations

## Risks

1. **Spec 025's measurement gate fails.** Hard dependency — see §"Implementation phases" item 1. If the gate fails, this spec doesn't start.

2. **Core ML compilation latency at model load.** First-time load adds 1–3 seconds. Mitigation: warm-up step at app launch + Core ML's built-in compilation cache. Worth measuring before declaring it acceptable.

3. **ANE / GPU thermal coupling.** On sustained workloads, both compute units share the same thermal envelope. May reduce sustained throughput vs GPU-only. Phase 2 measurement needs to include sustained-throughput, not just burst.

4. **Per-model integration cost compounds.** 10+ supported models × 2 candidate projections each = 20 integration sites if generalized fully. Mitigation: only do the high-vocab models where the win is meaningful; skip the rest.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/ANELMHead.swift` (new) | ANE-offloaded LM head wrapper |
| `Libraries/MLXLMCommon/ANEPLEProjection.swift` (new) | ANE-offloaded PLE projection wrapper (Gemma 4-specific) |
| `Libraries/MLXLMCommon/CoreMLModelBuilder.swift` (new) | Helper to build Core ML models from MLX weight tensors at load time |
| `Libraries/MLXLLM/Models/Gemma4.swift` | Replace GPU LM head + PLE projection with ANE wrappers |
| `Libraries/MLXLLM/Models/Qwen35.swift` | LM head only (phase 4) |
| `Libraries/MLXLLM/Models/GPTOSS.swift` | LM head only (phase 4) |
| `benchmarks/notes/ane-offload-2026-MM-DD.md` (new) | Per-model results |

## Why this is Tier 4

Three reasons:

1. **Hard dependency on spec 025**, which is itself blocked on Core ML LLM Swift Package availability + ANE measurement infrastructure (Tier 3 prework).
2. **Per-model effort**, not universal — each model + each projection is its own integration with its own perf measurement gate.
3. **Lower confidence than spec-decode wins** — spec-decode (013–025) targets 2–4× wins on entire model families; this targets 5–15% on a per-model basis. Worse Pareto.

Defer until spec 025 has measured concurrent execution on Apple Silicon and spec 021 phase 1A has shipped (or definitively failed). At that point, the integration patterns are proven and this becomes a straightforward bounded effort per model.
