# 016 — Cross-request n-gram cache (llama.cpp's `nc_dynamic`)

**Status:** 🚧 Phases 1–2 landed ([PR #146](https://github.com/ekryski/mlx-swift-lm/pull/146) — registry + tiered cache). Phase 3 (hybrid path for ngram cache — unblocked by [spec 020](020-tape-replay-rollback-generalised.md) shipping) and Phase 4 (three-tier draft selection in iterator) **not started.**
**Branch:** Phases 1–2 merged via PR #146; later phases get fresh branches off alpha.
**Depends on:** spec 013 (n-gram iterator) shipped, spec 015 (DFlash) optional

## Problem

Today `NGramLookup` is rebuilt from scratch at every iterator construction. The build cost is small for a single short prompt (~hundreds of µs at 256 tokens, ~few ms at 32K tokens), but it's wasted work in two patterns:

1. **Multi-turn chat / agentic loops.** Turns N and N+1 share most of the prompt; we rebuild the n-gram tables for the shared prefix on each turn.
2. **Repeating-template workloads.** A user generates ten emails / RFCs / recipes from the same template; the lookup table built for "the template + first N outputs" is the most useful state for the next request.

llama.cpp's `common_ngram_cache` solves both with **two persistent caches** that survive across calls (and optionally across processes via save/load):

- `nc_context` — built from the current request's prompt + generation. Lives for the request.
- `nc_dynamic` — built from *previous* user generations across this session. Persists across requests.
- `nc_static` — pre-built from a large corpus (e.g. WikiText, training data slice). Persists on disk.

When drafting, llama.cpp tries primary caches with **lax thresholds first**, falls back to the static cache as a validator, and only emits a draft if at least one cache produces a confident candidate. The dynamic cache is the one that turns multi-turn into a long stream from PLD's perspective.

## What we want

A persistent `NGramCache` that:

1. Is keyed on a stable `(targetModelId, ngramSize)` pair so different models / configs don't collide.
2. Retains tokens across multiple iterator constructions on the same `ModelContainer` (i.e. same process, same loaded model).
3. Optionally persists to disk under `~/.cache/mlx-swift-lm/ngram/{model}/{ngram}.bin` so it survives restarts (mirroring llama.cpp's `save`/`load`).
4. Is dropped/rotated when memory pressure rises (LRU on token count).
5. Plays nicely with the auto-routing in `MLXLMCommon.generate(...)` so callers don't need to thread the cache through every call.

## Design

### 1. Cache types

```swift
public final class NGramCache {
    public enum Tier {
        case context   // current request's history; ephemeral
        case dynamic   // accumulated across prior requests in this process
        case `static`  // loaded from disk; read-only at runtime
    }

    public init(
        modelId: String,
        ngramRange: ClosedRange<Int>,
        maxTokens: Int = 100_000,
        diskPath: URL? = nil
    )

    public func extend(_ tokens: [Int])             // append to dynamic
    public func snapshotForRequest() -> NGramLookup // build context+dynamic+static
    public func commitContext(_ tokens: [Int])      // promote context → dynamic at end of request
    public func save() throws                       // write dynamic to diskPath
    public func load() throws                       // hydrate dynamic from diskPath
    public func evict(maxTokens: Int)
}
```

### 2. Three-cache draft selection (port of llama.cpp logic)

When the iterator's `proposeDraft` runs, it consults three lookup objects:

```swift
// In NGramSpeculativeTokenIterator.speculateRound:
let drafts = lookup.tryDraft(thresholds: .lax)              // context cache
    ?? lookup.tryDraft(thresholds: .strict, in: dynamic)    // dynamic cache
    ?? lookup.tryDraft(thresholds: .static, in: staticCache) // static cache
```

Each `tryDraft` runs the existing multi-candidate / dominance / longest-match ladder, parameterised by per-tier thresholds. From `ngram-cache.cpp`:

```
draft_min_sample_size_lax    = {2,  2,  1,  1}
draft_min_percent_lax        = {66, 50, 50, 50}
draft_min_sample_size_strict = {4,  3,  2,  2}
draft_min_percent_strict     = {75, 66, 66, 66}
```

The static cache is used as a validator: when the primary cache produces a candidate, multiply its count by the static cache's count for the same continuation. Penalises tokens that the primary saw a lot but the static thinks are rare (typos, hallucinated boilerplate).

### 3. Process-level singleton

Add an `NGramCacheRegistry` keyed by `(modelId, ngramRange)`. The auto-routing in `MLXLMCommon.generate(...)` looks up or creates the cache for the current model on entry:

```swift
let cache = NGramCacheRegistry.shared.cache(for: context.modelId, ngramRange: range)
let iterator = try NGramSpeculativeTokenIterator(
    input: input, mainModel: context.model,
    parameters: parameters, ngramCache: cache)
```

At end of request, the iterator commits its context tokens into the registry's dynamic cache.

### 4. Disk persistence (optional)

Match llama.cpp's binary format:

```
struct NGramRecord {
    uint16_t ngram_tokens[LLAMA_NGRAM_MAX];
    uint32_t  num_continuations;
    struct { uint32_t token; uint32_t count; } continuations[];
}
```

Stream-write at `commit` time when disk path is set; stream-read at registry construction. Atomic via `tmp + rename`.

### 5. Static-cache bootstrap

Single one-shot tool: `swift run mlx-ngram-build-static --corpus path/to/wikitext --model qwen3-9b --output ~/.cache/mlx-swift-lm/ngram/qwen3-9b/static.bin`. Reads ~100 MB of text, tokenises with the model's tokenizer, builds the cache, writes to disk. Done once per model.

Static cache is purely opt-in — runtime works fine with it absent.

## Implementation phases

1. **Phase 1 — context-only persistence.** Just keep the `NGramLookup` alive across iterator constructions on the same `ModelContainer`. No tier hierarchy, no disk. Single env knob `MLX_NGRAM_CACHE_PERSIST=1`.
2. **Phase 2 — dynamic tier.** Add the `NGramCacheRegistry` singleton + `commit` lifecycle. Promote context tokens → dynamic at request end. Multi-turn + repeated-template workloads start winning here.
3. **Phase 3 — static tier from disk.** Save/load + the small CLI tool. Probably a separate spec depending on size.
4. **Phase 4 — three-tier draft selection.** Port llama.cpp's `try_draft` logic.

Ship phases independently — each one is a measurable win on its own.

## Expected impact

llama.cpp reports ~1.3–1.5× additional speedup over the no-cache PLD baseline on agentic / multi-turn workloads. SuffixDecoding, which is essentially this idea taken to its logical extreme (suffix tree instead of fixed-size n-gram tables, frequency-based scoring), reports up to 2.9× over SpecInfer on AgenticSQL.

For our setting, with PLD already winning ~+25% on input-grounded prompts on Gemma 4 26B A4B, expect another **+10-30%** on multi-turn chat because the dynamic cache turns "two turns of a chat" into "one accumulated history" for lookup purposes. Net effect: PLD becomes consistently 1.3–1.5× faster than baseline on chat workloads instead of the current 0–1.3×.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/NGramCache.swift` (new) | `NGramCache`, `NGramCacheRegistry`, three-tier scoring. |
| `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift` | New `init` accepting `ngramCache`; commit at `deinit`. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Auto-routing wires registry lookup before iterator construction. |
| `Tests/MLXLMTests/NGramCacheTests.swift` (new) | Unit tests for `extend`, `commit`, evict, save, load, three-tier scoring. |
| `Sources/Tools/MLXNgramBuildStatic` (new) | One-shot CLI to bake a static cache from a text corpus. |

## Open questions

1. **Privacy.** Dynamic cache accumulates user content. For a serving deployment, callers should be able to opt out per-request (or the cache should be per-session, not process-wide). Mirror vLLM's `cache_salt` design? Default behaviour for a desktop CLI is fine to share; for `mlx_lm.server`-style deployments, default off.
2. **Cache invalidation.** When the user changes the system prompt or chat template, the dynamic cache contains tokens from a different distribution. Detect this? Or just let LRU handle it?
3. **Memory budget.** What's a sensible default `maxTokens`? llama.cpp uses unbounded; for a long-running agent that's a problem. Probably 100K tokens default with LRU rotation.
4. **Save cadence.** Every commit (write-heavy) vs. on shutdown (lossy on crash) vs. periodic background flush? Probably commit-on-end with a soft 32MB cap before forcing a flush.

## References

- [llama.cpp `ngram-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-cache.cpp) — `nc_context`/`nc_dynamic`/`nc_static`, `try_draft` thresholds, save/load.
- [SuffixDecoding (NeurIPS 2025)](https://arxiv.org/abs/2411.04975) — the suffix-tree generalisation; same idea taken further.
