# 017 — Cross-request prefix KV cache

**Status:** spec, ready to issue
**Branch:** new branch off `main`
**Depends on:** none — orthogonal to 013/014/015/016

## Problem

Multi-turn chat and agentic workloads send a prompt whose **prefix is identical** to the previous turn's prompt: same system message, same chat template boilerplate, often identical earlier user/assistant turns. Today every turn re-runs prefill over that prefix. For a 4K-token context that's the dominant TTFT cost.

vLLM's "Automatic Prefix Caching" (APC) and llama.cpp's `--prompt-cache` solve this with a key-value snapshot of the target's KV state at a stable prefix boundary. dflash-mlx implements the same in Python under `dflash_mlx/cache/`. We don't.

This spec is **decoder-agnostic**: the prefix cache helps `TokenIterator` baseline decoding, `NGramSpeculativeTokenIterator`, and the future `DFlashSpeculativeTokenIterator` equally, because they all start from the same target-model KV state.

## Design

### 1. Snapshot

A `PrefixSnapshot` captures the target KV state at the end of a prefill:

```swift
public struct PrefixSnapshot: Sendable {
    let key: PrefixKey                  // (model id, layer count, head dims)
    let tokens: [Int]                   // the exact prefix, byte-stable
    let layerStates: [LayerCacheState]  // per-layer K, V (or quantised equivalents)
    let lastHidden: MLXArray?           // last hidden state (used by DFlash, optional)
    let createdAt: Date
}
```

`LayerCacheState` is a sum type covering `KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, `TurboQuantKVCache`, and `MambaCache`. Each cache type already has a `.state: [MLXArray]` accessor; serialisation is trivial.

### 2. Cache

```swift
public final class PrefixKVCache {
    public init(maxBytes: Int = 8 * 1024 * 1024 * 1024,    // 8 GB default
                maxEntries: Int = 4)
    public func lookup(prefix: [Int], key: PrefixKey) -> (matchedLen: Int, snapshot: PrefixSnapshot)?
    public func insert(_ snapshot: PrefixSnapshot)
    public func clear()
    public var stats: PrefixCacheStats { get }
}
```

Two-level keying:
- **Token-prefix match** — find the longest snapshot whose tokens match the request's prompt token-for-token from position 0.
- **Configuration key** — refuse to restore if model id, layer count, head dims, or KV bit-width differ. Snapshots are not interchangeable across models or quantisation settings.

### 3. Stable prefix policy

Chat-template suffix tokens (`<|im_start|>assistant\n`) change every turn, but they're stable in shape. dflash-mlx solves this with `compute_stable_prefix_len` which trims the trailing assistant-opener tokens from the candidate prefix before lookup. We do the same:

```swift
public protocol StablePrefixPolicy {
    /// Returns the largest `n` such that `tokens[0..<n]` is stable across
    /// turns of the same conversation. The trailing window typically
    /// contains chat-template boilerplate that will be regenerated next
    /// turn.
    func stablePrefixLen(_ tokens: [Int], tokenizer: Tokenizer) -> Int
}
```

Default implementations:

- `LastAssistantOpenerPolicy` — looks for the last `<|im_start|>assistant` (or model-family equivalent) and trims everything from that boundary onward. Works for Qwen, Gemma 4, GPT-OSS chat templates.
- `IdentityPolicy` — returns `tokens.count`. For non-chat workloads (tool calls, completion).

### 4. Restore semantics

```swift
public func generate(input: LMInput, ...) {
    if let snap = prefixCache.lookup(prefix: input.tokens, key: ourKey) {
        hydrateCache(snap)
        let suffix = input.tokens[snap.tokens.count...]
        // Run prefill only over the suffix.
    } else {
        // Standard prefill over the whole prompt.
    }
    // ... iterator runs as today
}
```

Hydration is a deep copy of the snapshot's MLXArrays into a fresh `[KVCache]` so subsequent generation cannot mutate the snapshot. Cost: GPU-side memcpy of ~`O(prefix_len × layers × kv_dim × bits)`, much cheaper than re-running the prefix forward.

### 5. Insert at end

After generation completes, the iterator's cache holds the full request state. We snapshot at the **stable prefix boundary** for the *next* request — that is, we trim the assistant response from the cache before snapshotting.

For trimmable caches, this is `trimPromptCache(cache, numTokens: cache.offset - stablePrefixLen)`. For hybrid caches with non-trimmable Mamba layers, see spec 020 (tape-replay rollback) — without that, hybrid models can only snapshot at request *start* (before the assistant turn) which still helps.

### 6. Eviction

LRU within a byte budget. `PrefixSnapshot.bytes` accumulates `MLXArray` data sizes. On insert, evict the oldest entries until the new snapshot fits.

A future refinement (out of scope for v1): dominance-based eviction — keep snapshots whose prefix is a substring of more queued prefixes. dflash-mlx hints at this but doesn't implement it.

## Telemetry

`PrefixCacheStats` exposes:
- hits / misses / partial hits
- bytes used / bytes budget / entries
- average matched length on hit
- per-key hit rate

The bench harness should print one `[PREFIX-CACHE]` line per request when the cache is enabled.

## Expected impact

vLLM reports 2–10× TTFT improvement on multi-turn chat with APC enabled. dflash-mlx reports ~1.5–2× wall-clock improvement on agentic workloads (TTFT compounds with per-turn decode speedup). Our setting is per-process not per-cluster, so the headline numbers will be smaller — but for a long chat session on Gemma 4 26B A4B (4K-token system prompt + accumulated history), we should see TTFT drop from ~7s to ~1s on turn N+1.

## Files touched

| File | What |
|---|---|
| `Libraries/MLXLMCommon/PrefixKVCache.swift` (new) | `PrefixKVCache`, `PrefixSnapshot`, `LayerCacheState`. |
| `Libraries/MLXLMCommon/StablePrefixPolicy.swift` (new) | Policy protocol + default Qwen/Gemma 4/GPT-OSS impls. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Lookup at `generate` entry; hydrate; insert at end. |
| `Libraries/MLXLMCommon/KVCache.swift` | `KVCache.serialise()` / `KVCache.hydrate(from:)` for each concrete cache type. |
| `Tests/MLXLMTests/PrefixKVCacheTests.swift` (new) | Hit/miss/partial; cross-model isolation; eviction. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | New `multi-turn-cached` bench method. |

## Phasing

1. **Phase 1** — `KVCacheSimple` + `RotatingKVCache` only; no Mamba. Token-exact prefix matching, no stable-prefix trim. Validates the snapshot/restore plumbing on Gemma 4 / GPT-OSS / Llama / Phi.
2. **Phase 2** — `LastAssistantOpenerPolicy` + chat-aware stable prefix. Where multi-turn wins start.
3. **Phase 3** — `MambaCache` snapshot via spec 020's tape-replay rollback (or a much simpler full-state checkpoint if 020 isn't shipped yet — Mamba state is small).
4. **Phase 4** — Disk persistence (write snapshots to `~/.cache/mlx-swift-lm/prefix/`). Optional; mostly useful for bench reproducibility.

## Open questions

1. **Cross-process sharing.** vLLM's APC is per-process; the same prefix served across multiple instances is recomputed. For our single-process desktop / iOS setting this isn't a concern.
2. **Quantised-cache snapshot fidelity.** Restoring a snapshot with a different KV quantisation than the runtime configures should fail loudly. Plumb through `PrefixKey`.
3. **Rotating cache wrap.** If the snapshot was taken when offset > maxKVSize (sliding-window layers wrapped), the snapshot semantics get tricky — replay would diverge from a fresh prefill. Easiest fix: refuse to snapshot wrapped rotating caches. Document.

## References

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) — hash-based, page-granular, cross-request.
- [dflash-mlx prefix cache](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2/dflash_mlx/cache) — Python reference; we'd port the design to Swift.
- [llama.cpp `--prompt-cache`](https://github.com/ggml-org/llama.cpp) — single-snapshot variant of the same idea.
