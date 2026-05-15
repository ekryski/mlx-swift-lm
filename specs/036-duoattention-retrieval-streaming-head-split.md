# 036 — DuoAttention: retrieval / streaming head split via calibration

- **Status:** spec, ready to issue (requires per-model calibration; modest kernel work)
- **Branch:** new branch off alpha
- **Depends on:** PR #186 windowed eviction (turbo windowed-turbo eviction); spec 020 tape-replay rollback at the cache layer (for spec-decode composition); ideally lands after `KVCache` per-head-shape support is sorted (the load-bearing technical risk — see §"What we already need to figure out").
- **Origin:** [`papers/beyond-quadratic-attention-on-apple-silicon.md`](../papers/beyond-quadratic-attention-on-apple-silicon.md) §3.2; [DuoAttention, MIT, ICLR 2025 (arXiv 2410.10819)](https://arxiv.org/abs/2410.10819); [reference impl](https://github.com/mit-han-lab/duo-attention); [project page](https://hanlab.mit.edu/projects/duo-attention)

## The insight

In any pretrained attention layer, **only a fraction of heads do long-range retrieval**. The rest are "streaming" heads — they only attend to a sink prefix + a recent window. DuoAttention's claim, validated across Llama-2/3 + Mistral 7B–70B in the original paper, is that:

1. Most MHA heads are streaming (~50%); GQA models are denser (~30–40% streaming).
2. The split is *stable across inputs* — a head's role is a learned property of the pretrained weights, not the prompt.
3. A short calibration pass (synthetic needle-in-haystack tasks, single-pass scoring) classifies every head once at deploy time.

At runtime, retrieval heads keep the full KV; streaming heads keep only `keep` (sink) + `maxSize` (window) tokens. **The streaming-head plumbing already exists in `StandardKVCache(maxSize:keep:)` and the new windowed turbo path from PR #186** — what's missing is the per-head dispatch.

Headline numbers from the paper: **2.18× decode / 1.50× decode** (MHA / GQA), **1.73× / 1.63× prefill**, ~half the KV memory on MHA. Llama-3-8B at 3.3 M context on a single A100 with quant. Lossless within calibration tolerance — no perplexity regression on RULER when calibration is well-tuned.

Critically, **this is the technique that most directly slots into the windowed-KV hierarchy that PR #186 just landed.** The streaming-head case *is* a windowed cache; the retrieval-head case is unbounded. The contribution of this spec is making the per-attention-layer construction return one of these two cache types per head, and making the SDPA dispatch handle the ragged shape that produces.

## Why Apple Silicon specifically

DuoAttention's wins are bandwidth-driven (fewer K/V reads per query) and memory-driven (smaller per-head caches). Both translate ~1:1 on M-series. Beyond that:

- **Composes with TurboQuant.** Streaming heads can use `TurboQuantizedKVCache(maxSize:)`; retrieval heads use unbounded `TurboQuantizedKVCache()`. Both code paths exist.
- **Composes with windowed eviction (PR #186).** Streaming heads use a windowed cache — the rotating window is exactly the streaming pattern. The windowed-turbo path that just landed is the streaming-head storage type for free.
- **Composes with Quest (spec 035).** Quest applies to retrieval heads only (streaming heads have no full cache to score against). The two specs partition the heads cleanly.

## What we already need to figure out

The load-bearing question for whether DuoAttention is a 2-week project or a 6-week project: **how do we hold per-head caches with different shapes inside one layer?**

Three options, in order of preference:

### Option A — Two caches per layer, head-permuted Q/K/V

Rearrange Q/K/V so retrieval heads come first, streaming heads second:

```
[H_r, H_s] heads, where H_r heads are retrieval and H_s are streaming
keys = [keys_retrieval, keys_streaming]  // along head axis
```

Hold two `KVCache` instances per attention layer:

```swift
class DuoAttentionCache: BaseKVCache {
    let retrieval: any KVCache    // unbounded
    let streaming: any KVCache    // windowed (sink + maxSize)
    let retrievalHeadCount: Int   // computed at calibration time
    let streamingHeadCount: Int
}
```

At write time: split K/V by head, dispatch to two `update(...)` calls. At read time: two SDPAs, concat outputs along head axis. **This is the path that lets us reuse `StandardKVCache` / `TurboQuantizedKVCache` unchanged.**

Cost: two SDPA dispatches per attention layer instead of one. Mitigation: in the steady-state windowed-turbo path the streaming SDPA is much cheaper (smaller K/V), so the two dispatches in parallel are still a net win. Phase 1 measures this.

### Option B — Single ragged cache, masked SDPA

Keep one cache with full storage; at read time, build a per-head mask that zeros out everything beyond `sink + maxSize` for streaming heads. **No memory savings**, only bandwidth-via-mask savings (and only if the SDPA kernel can short-circuit masked tokens, which it can't on the current `MLXFast.scaledDotProductAttention` path).

Reject Option B unless A is somehow infeasible — it gives up the headline memory win.

### Option C — Custom Metal kernel that handles ragged per-head shapes natively

Write an SDPA kernel that takes `[H, T_r_or_s, D]` per-head ragged K/V. Biggest engineering project, biggest measurable win (single dispatch, no concat). Defer to a phase 4-5 optimization once option A's win is measured.

**Recommendation: Phase 1 ships Option A.** The two-dispatch overhead is the trade for being able to ship without writing a new attention kernel. If the bench shows the dispatch overhead eats >25% of the projected win, escalate to Option C.

## What this composes with

- **PR #186 windowed turbo eviction.** Streaming heads use `TurboQuantizedKVCache(maxSize:)`. Already works.
- **Quest (spec 035).** Retrieval heads use paged + Quest. Already works (post-035).
- **Spec 020 tape-replay rollback.** Both retrieval and streaming caches need rollback. Spec 020's per-cache-type rollback already covers `StandardKVCache` and `TurboQuantizedKVCache`; `DuoAttentionCache` rollback is just dispatching to both inner caches.
- **MoE expert paths.** Untouched — DuoAttention is a per-attention-layer split, MoE is a per-FFN-layer split. Orthogonal.
- **Spec-decode (013, 014, 019, 021, 023).** Speculative decode through a duo cache works as long as both inner caches are tape-replay-conformant. The drafts and verifies share the per-head split.

## What this does NOT compose with

- **Pure GDN / Mamba layers.** Those don't have a softmax cache to split. Models like Qwen 3.5 with hybrid GDN+attention layers get DuoAttention only on their attention layers, and only those layers benefit. Hybrid models like Granite 4.0-H benefit on the few attention layers they have — small relative win.
- **Sliding-window-only models** (Gemma 4 has alternating sliding/full layers). On the sliding layers DuoAttention is a no-op (already streaming). On the full layers the win is real. Net: ~half the layers benefit.
- **Native sparse-attention models (NSA / DSA / MoBA).** These already learned the head-routing pattern at pretraining. DuoAttention overlays a coarser, lower-quality split — don't apply.

## Design

### Phase 1 — `DuoAttentionCache` + dual-SDPA dispatch

Add a new cache type that holds two inner caches and a head split:

```swift
public class DuoAttentionCache: BaseKVCache {
    public let retrievalCache: any KVCache      // unbounded
    public let streamingCache: any KVCache      // windowed (sink + maxSize)
    public let retrievalHeads: Int              // first H_r heads are retrieval
    public let streamingHeads: Int              // remaining H_s heads are streaming
    public let permutation: [Int]?              // optional head re-permutation if model
                                                // doesn't natively store retrieval-first
}
```

`update(...)` splits K/V by head along `axis=1` (assuming permuted layout) and forwards to each inner cache. Read path returns `(retrievalK, retrievalV, streamingK, streamingV)` and the model's attention call site issues two SDPAs and concatenates outputs.

Per-head split is a deploy-time artifact baked into the loaded weights — head permutation, if used, runs at `sanitize()` so the runtime sees a model where heads `[0..H_r)` are retrieval and `[H_r..H)` are streaming. **This avoids per-token gather/scatter at decode.**

### Phase 2 — Calibration script + mask format

Provide a calibration script that, for any pretrained model:

1. Runs synthetic needle-in-haystack at 8 K, 16 K, 32 K context.
2. For each attention head, measures the *attention-shift score* — how much the model's output for the needle question changes when the head's KV is replaced with the streaming version.
3. Threshold the scores; produce a per-layer head mask:

```json
{
  "model": "Qwen/Qwen3-9B",
  "calibration": {
    "ruler_score": 0.94,
    "tested_at": "2026-MM-DD"
  },
  "per_layer_streaming_heads": [
    [3, 7, 12, 15, 19, 22, 28, 31],   // layer 0: 8 of 32 heads are streaming
    [1, 4, 9, 11, 18, 23, 26, 29],    // layer 1
    ...
  ]
}
```

These ship as JSON sidecars under `recipes/duoattention/<model>.json`, parallel to spec 027's recipe library. **One calibration per model, runs in ~20 minutes on a server GPU**, results check into the repo. Apple-Silicon users don't re-calibrate.

### Phase 3 — Per-model `sanitize()` integration

In each supported model's `sanitize()`, if a duoattention recipe is loaded:

1. Read the per-layer streaming-head set.
2. Permute Wq / Wk / Wv / Wo so retrieval heads come first.
3. Wire `newCache(parameters:)` to construct `DuoAttentionCache` with the right split per layer.

The model's attention `forward` call site grows a duo-aware branch that runs two SDPAs and concats. ~50 lines per model.

### Phase 4 — Bench + budget tuning

Run RULER + LongBench + needle-in-haystack at 8/16/32/64/128 K context with and without DuoAttention on:

- **Qwen 3.5-9B** (priority — main user-facing model)
- **Gemma 4 26B-A4B** (MoE)
- **GPT-OSS-20B**
- **Optionally: Qwen 3.5-26B**

Document in `benchmarks/notes/spec-036-duoattention-2026-MM-DD.md`. Tune the streaming/retrieval threshold per model if the default loses >2% on RULER.

### Phase 5 (optional) — Fused per-head ragged SDPA Metal kernel

If Phase 4 measurement shows the dual-SDPA dispatch overhead eats >25% of the projected win, write a single SDPA kernel that takes ragged per-head K/V shapes natively. Otherwise skip — the projected win already lands.

## Implementation phases

1. **Phase 1 — `DuoAttentionCache` infrastructure** (1 week). New cache class + `update`/`read`/`makeMask`/`copy`/`trim` conformance. Composes with both `StandardKVCache` and `TurboQuantizedKVCache` as inner cache types. ~250 lines + tests.

2. **Phase 2 — Calibration script** (1–2 weeks). Standalone Python script (PyTorch reference for portability — runs on any GPU; result JSON ships in repo). Implements the attention-shift scoring from the paper. Generates a recipe per supported model. Lives in `tools/duoattention-calibrate/`.

3. **Phase 3 — Per-model integration** (1 week per model). `sanitize()` hook + attention `forward` dual-dispatch path. Start with Qwen 3.5-9B; copy pattern to Gemma 4 / GPT-OSS.

4. **Phase 4 — Bench sweep** (1 week). RULER + LongBench × 3 models × 5 contexts = 15 cells. Tune thresholds. Decision gate: ship per-model only when no >2% RULER regression at the chosen budget.

5. **Phase 5 (optional) — Ragged SDPA Metal kernel** (3–4 weeks). Only if Phase 4 says the dispatch overhead matters.

## Expected impact

**On supported models, post-calibration, at long context:**

- **Qwen 3.5-9B (GQA, 32 heads, 8 KV heads):** ~30–40% of heads streaming → projected **+25–40% decode at 32 K**, **+30–50% at 64 K**, ~30% KV-memory reduction, +10–20% prefill.
- **Gemma 4 26B-A4B:** alternating sliding+full layers complicate things. On the full layers the win matches Qwen 3.5; on the sliding layers it's a no-op. Net: **+15–25% decode at 32 K** (half the layers benefit).
- **GPT-OSS-20B:** projected **+30–40% decode at 32 K** (assuming GQA streaming-head fraction ~35% by analogy to Llama 3).
- **Memory:** ~30% KV-cache footprint reduction at long context on GQA models, enabling longer context windows on the same hardware.

- **Quality:** lossless within calibration tolerance. RULER loss <1% at 32 K post-calibration in the paper. Watch reasoning-chain workloads (multi-hop QA, math) — these stress retrieval more than the calibration set, may need a tighter streaming threshold.

**Where this DOESN'T help:**

- Short context (<4 K). The streaming heads' `maxSize` cap doesn't bind; we just pay the dispatch overhead.
- Pure GDN/Mamba models. No softmax cache to split.

## Risks

1. **Calibration generalisation.** The per-head retrieval/streaming label is calibrated on synthetic needle-in-haystack; production workloads may stress different heads. Mitigation: Phase 4 includes a real-prompt regression set; if a head class systematically fails (e.g., reasoning chains regress >3% on GSM8K) the threshold gets reset.

2. **Head permutation correctness.** Permuting Wq / Wk / Wv / Wo at sanitize() must be commutative with the rest of the attention math (RoPE applies per-head, q/k norms apply per-head, residual stream is unaffected — all fine in principle). Risk is per-model implementation bugs. Mitigation: Phase 3 includes a bit-exact-output regression test (run a calibrated model with `streaming_heads = []` — should match baseline forward exactly).

3. **Two-SDPA-dispatch overhead.** On M-series the kernel-launch cost is ~10–30 μs; doubling it per attention layer per token is real. At 32 layers × 50 tok/s × 2 dispatches × 20 μs = ~64 ms/s of overhead, ~6% slowdown. Mitigation: Phase 1 measures actual overhead; if it eats >25% of the projected win, Phase 5 fuses.

4. **Composition with PR #186 windowed turbo on alternating-sliding-window models.** Gemma 4's mixed sliding/full layers + DuoAttention's per-head split + TurboQuant's Path A/B + windowed eviction is a four-way interaction that needs explicit verification. Bug-class is silent quality regression. Mitigation: Phase 3 starts with Qwen 3.5 (uniform layers); Gemma 4 lands only after Qwen 3.5 ships and the unit-test bar is high.

5. **The MIT reference implementation uses retraining-time learned masks; we use post-hoc calibration only.** The paper's training-free calibration variant works but with slightly worse quality. Mitigation: Phase 2 implements both — the scoring-based variant for portability across all models, with a `--retrain-friendly` mode that the user can opt into if they have access to gradient access on a fine-tuned variant. Default: pure post-hoc.

6. **Spec-decode through a duo cache.** Tape-replay rollback (spec 020) needs to roll back both inner caches consistently. If a draft causes one cache to advance and the other to fail to advance (impossible in practice, but defensible against), state diverges. Mitigation: Phase 1's `DuoAttentionCache` implements `update`/`trim` as transactional — both inner caches advance or neither does.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/DuoAttentionCache.swift` (new) | New cache type, dual-inner-cache management |
| `Libraries/MLXLMCommon/DuoAttentionRecipe.swift` (new) | JSON schema for per-layer streaming-head sets |
| `Libraries/MLXLMCommon/KVCacheTypes.swift` | Extend `makeAttentionCache(...)` to construct `DuoAttentionCache` when a recipe is loaded |
| `Libraries/MLXLLM/Models/Qwen35.swift` | sanitize() head permutation + dual-SDPA forward branch |
| `Libraries/MLXLLM/Models/Gemma4.swift` | Same (Phase 3, after Qwen 3.5 ships) |
| `Libraries/MLXLLM/Models/GPTOSS.swift` | Same |
| `recipes/duoattention/qwen35-9b.json` (new) | Phase 2 calibration output, checked into repo |
| `recipes/duoattention/gemma4-26b-a4b.json` (new) | |
| `recipes/duoattention/gpt-oss-20b.json` (new) | |
| `tools/duoattention-calibrate/` (new) | Standalone Python calibration script (PyTorch reference) |
| `scripts/benchmark.sh` | `--duoattention <recipe>` CLI flag |
| `Tests/MLXLMCommonTests/DuoAttentionCacheTests.swift` (new) | Cache conformance + bit-exact-with-empty-streaming test |
| `benchmarks/notes/spec-036-duoattention-2026-MM-DD.md` (new) | Phase 4 sweep results |

## Why this is Tier 3-adjacent

Three reasons it's a higher-priority spec than 027–029:

1. **Cleanest composition with PR #186 windowed eviction** — the streaming-head case *is* the windowed cache; the infrastructure already landed.
2. **Memory + bandwidth + prefill all win at once** — most other Tier 4 specs win one axis. DuoAttention wins three.
3. **Long-context is where users actually notice slowness.** 4 K → 32 K is the biggest UX cliff; this spec attacks it directly.

It's behind Tier 1 (#5 spec 020 tape-replay rollback) because spec-decode composition matters; behind Quest (#035) because the head-split makes more sense once paged + Quest are in place to handle the retrieval heads. But it's strictly above Mixture-of-Depths-style speculation (dead at scale) and above the Tier 4 quantization framework (small audience).

Suggested ordering: **after PR #186 ships, after spec 020 phase 2-3 lands, after spec 035 phase 1-3 lands, this is the next sub-quadratic-attention bet.**
