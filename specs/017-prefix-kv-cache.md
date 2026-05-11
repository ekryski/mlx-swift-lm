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

`LayerCacheState` is a sum type covering the post-spec-006 cache hierarchy: `StandardKVCache` (unbounded or windowed), `AffineQuantizedKVCache`, `TurboQuantizedKVCache`, and `SSMStateCache`. Each cache type already has a `.state: [MLXArray]` accessor; the typed-surface exposed via `KVStorageKind` (raw / affineQuantized / turboCompressed / ssm / composite) lets the dispatch decide per-cache serialisation cost up front.

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

For trimmable caches, this is `trimPromptCache(cache, numTokens: cache.offset - stablePrefixLen)`. For hybrid caches with non-trimmable `SSMStateCache` layers (Qwen 3.5 / Qwen 3 Next / Nemotron-H / Jamba), see spec 020 (state-replay rollback) — without that, hybrid models can only snapshot at request *start* (before the assistant turn) which still helps.

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

1. **Phase 1** — `StandardKVCache` (unbounded + windowed) only; no `SSMStateCache`. Token-exact prefix matching, no stable-prefix trim. Validates the snapshot/restore plumbing on Gemma 4 / GPT-OSS / Llama / Phi.
2. **Phase 1B** — Per-class `serialise()` / `hydrate(from:)` on each concrete cache type (`StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`). Wires `Evaluate.generate` lookup + hydrate + insert.
3. **Phase 2** — `LastAssistantOpenerPolicy` + chat-aware stable prefix. Where multi-turn wins start.
4. **Phase 3** — `SSMStateCache` snapshot via spec 020's state-replay rollback (or a much simpler full-state checkpoint if 020 isn't shipped yet — SSM state is small).
5. **Phase 4** — Disk persistence (write snapshots to `~/.cache/mlx-swift-lm/prefix/`). Optional; mostly useful for bench reproducibility. **Realised upstream as `prefix_l2.py`** — see References.

## Phase 1 status (PR #144, ready for review)

Phase 1 lives in [PR #144](https://github.com/ekryski/mlx-swift-lm/pull/144), rebased onto current `alpha` (post spec-006 / WS-A–D / consolidation sprint):

- `Libraries/MLXLMCommon/PrefixKVCache.swift` (332 lines) — `PrefixKey`, `LayerCacheState` (opaque `[MLXArray]`), `PrefixSnapshot`, `PrefixCacheLookupResult`, `PrefixCacheStats`, `PrefixKVCache` class with `lookup` / `insert` / `clear` / `resetStats` + LRU eviction + byte-budget cap.
- `Libraries/MLXLMCommon/StablePrefixPolicy.swift` (56 lines) — `StablePrefixPolicy` protocol + `IdentityPolicy` + `FixedTrimPolicy`.
- `Tests/MLXLMTests/PrefixKVCacheTests.swift` — 21 tests covering exact/partial match, cross-model isolation, mismatch miss, LRU bump, byte-budget eviction, entry-count eviction, hitRate, meanMatchedLength.

**Open work for phase 1B (per-class `serialise` / `hydrate`):**

- `LayerCacheState.arrays: [MLXArray]` is **opaque** in `PrefixKVCache.swift` — phase 1B replaces this with a typed `LayerCacheSnapshot` enum so hydration validates shape + dtype per cache class. The enum dispatches via `KVStorageKind` (`.raw` / `.affineQuantized(bits:groupSize:)` / `.turboCompressed(keyBits:valueBits:)` / `.ssm` / `.composite`) — same axis the runtime already exposes for cache-typed dispatch.
- No `serialise()` / `hydrate(from:)` methods yet on `StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache`. Each cache's existing `state: [MLXArray]` accessor is the starting point: `serialise()` returns the cache's `state` plus enough metaState (`maxSize`, `keep`, `groupSize`, `bits`, `keyBits`, `valueBits`, `compressedWriteOffset`, `rotatingIdx` etc.) to round-trip. `hydrate(from:)` is the constructor-shaped inverse. Defer `SSMStateCache` to phase 3 (depends on spec 020 phase 2).
- `Evaluate.swift` is **not wired** — no calls to `prefixCache.lookup` or `.insert` in the generate path. Wire-in callsites: `generate(input:cache:parameters:context:wiredMemoryTicket:)` (currently around line 2065 in alpha; symbol-level edit) and the analogous draft-model variant.

**Open work for phase 2 (chat-aware stable prefix):**

- `LastAssistantOpenerPolicy` is **not** implemented — only `IdentityPolicy` + `FixedTrimPolicy` ship in phase 1. Phase 2 adds it with sentinel encodings for Qwen (`<|im_start|>assistant\n`), Gemma 4 (`<start_of_turn>model\n`), GPT-OSS (`<|start|>assistant<|channel|>`), pre-encoded at construction since the protocol's `stablePrefixLen(_:)` takes only `[Int]`.

**Built against the post-spec-006 KV-cache hierarchy:** phase 1B's per-class `serialise()` / `hydrate(from:)` methods target the post-spec-006 cache classes (`StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`) — the legacy names (`KVCacheSimple` / `RotatingKVCache` / `QuantizedKVCache` / `TurboQuantKVCache`) are gone. `SSMStateCache` (renamed from `MambaCache`) gets its round-trip in phase 3 once spec 020 phase 2 lands the state-replay infrastructure.

## dflash-mlx upstream updates since spec drafted

This spec was drafted against `bstnxbt/dflash-mlx@engine-v2`. Commit [`bc24ab0`](https://github.com/bstnxbt/dflash-mlx/commit/bc24ab0) (2026-04-27, on `main`) reshapes the upstream prefix-cache design — phase 1B / phase 2 should adopt these:

- **Cache subpackage** at [`dflash_mlx/cache/`](https://github.com/bstnxbt/dflash-mlx/tree/main/dflash_mlx/cache) replaces the single `prefix_cache.py`. Files we mirror conceptually:
  - [`fingerprints.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/cache/fingerprints.py) — `DFlashPrefixKey` with `target_model_id`, `draft_model_id`, `capture_layer_ids: tuple[int, ...]`, `draft_sink_size`, `draft_window_size`, `target_fa_window`, `format_version: int = 2` (bumped 1→2 in [`463d722`](https://github.com/bstnxbt/dflash-mlx/commit/463d722), see "post-`8d8545d` deltas" below). Phase 1B extends our `PrefixKey` with `captureLayerIds: [Int]?` (nil = all layers cached) and `formatVersion: Int = 2`.
  - [`snapshot.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/cache/snapshot.py) — `DFlashPrefixSnapshot` carries chunked `target_hidden_chunks` + `target_hidden_chunk_spans` (sink + tail trim, not full hidden state) for the DFlash draft. Phase 1B reserves the shape: replace `lastHidden: MLXArray?` with `targetHiddenChunks: [(MLXArray, ClosedRange<Int>)]?` + `targetHiddenTotalLen: Int?`. Defaults to `nil` since we don't ship DFlash yet — reserves the on-wire shape without forcing a `formatVersion` bump later.
  - [`policies.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/cache/policies.py) — `target_cache_is_serializable(...)` returns `False` for any cache whose Python equivalent is a `RotatingKVCache`. Our equivalent is **a windowed `StandardKVCache` (`eviction == .window(...)`) whose `offset` exceeds `maxSize`** — the rotating buffer has wrapped, so the serialised state would no longer be a faithful prefix snapshot. Confirms our open-question 3: refuse to snapshot wrapped windowed caches. Document explicitly + assert in `serialise()`.
  - [`prefix_l1.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/dflash_mlx/cache/prefix_l1.py) — `DFlashPrefixCache` with richer stats: `exact_hits`, `prefix_hits`, `misses`, `insertions`, `evictions`, `byte_budget_evictions`, `skipped_too_long`, `prefix_prunes`, `cross_kind_prunes`, `prefill_tokens_saved`, `fingerprint_rejects`, `l2_hits`, `l2_misses`. Phase 1B extends `PrefixCacheStats` with `exactHits`, `byteBudgetEvictions`, `skippedTooLong`, `fingerprintRejects`, `prefillTokensSaved`. Defer L2 fields until phase 4.
  - `prefix_l2.py` (26 KB) — disk persistence layer, our phase 4 realised upstream. Recent commit [`8d8545d`](https://github.com/bstnxbt/dflash-mlx/commit/8d8545d) "keep MLX work out of async L2 writer" sets the discipline for phase 4: L1 sync read/write; L2 async-write only, sync-read on miss; no MLX work on the async writer.
- **Multi-turn benchmark template**: [`benchmark/bench_prefix_cache_multiturn.py`](https://github.com/bstnxbt/dflash-mlx/blob/main/benchmark/bench_prefix_cache_multiturn.py) — turn 1 cold (warmup), turns 2-N measured for hit-rate + saved prefill tokens. Mirror this in `Tests/Benchmarks/InferenceBenchmark.swift`'s new `multi-turn-cached` method.

## dflash-mlx post-`8d8545d` deltas (2026-05-08 sync)

Three upstream commits land between `8d8545d` (drafted-against) and the current `main` HEAD that further sharpen the design surface. They do not invalidate phase 1; phases 1B / 2 / 4 incorporate them:

- **`463d722` (2026-05-10) — "perf: persist and reuse stable prefix snapshots"**
  - Bumped `DFlashPrefixKey.format_version` 1→2 (**backward-incompatible**). Our `PrefixKey.formatVersion` ships at `2` from phase 1B; v1 snapshots are rejected as mismatched on load.
  - New snapshot invariants enforced at `insert()` / `hydrate()`: (a) **FA cache offset must equal token-prefix length** — guards against capturing mid-cycle state. (b) **Target hidden chunks are truncated to match prefix length** — guards against dangling tail across the stable-prefix boundary. Phase 1B asserts both inside `serialise()`; phase 2 (stable-prefix trim) re-asserts after the trim.
  - **L1 lookup signature**: added `record: Bool = true` parameter (non-recording lookups for diagnostics / probing without bumping LRU). Phase 1B's `PrefixKVCache.lookup(prefix:key:record:)` matches.
  - **L2 lookup signature**: added `minTokenLen: Int = 0` parameter (filter candidates by minimum token length). Defer to phase 4 with the L2 backing store.
  - **Combined L1+L2 lookup policy**: prefer the **longer prefix match** even if it crosses a tier boundary (L2 longer-match beats L1 shorter-match). Document in phase 4; phase 1B is L1-only so non-issue.
  - **L2 insert dedup**: check for existing snapshots before queueing async writes. Phase 4 contract; mirror as "no-op on identical-key insert" in `PrefixKVCache.insert()`.
  - Internal rename `target_hidden` → `draft_context` in breakdown reporting. Our equivalent (`targetHiddenChunks`) stays as-is for clarity; rename is upstream's accounting-field-name choice, not a wire-format change.

- **`8c29e3e` (2026-05-05) — "test: add prefix-cache survival gate"**
  - Defines the **survival-gate test methodology** that phase 2 / phase 4 PRs must adopt. Five behavioural contracts:
    1. **Budget fit**: survival case sits within token budget (e.g. 176–220 of 220), needle record placement deterministic.
    2. **Cold→warm prefix preservation**: warm turn keeps the cold turn's messages as an immutable prefix; only the new request nonce is appended.
    3. **Reuse ratio floor**: warm turn must restore **≥ 80%** of the prompt from the cache. Below that, the gate fails.
    4. **Staleness guard**: a wrong-answer query must get **zero** cache benefit — protects against cache-poisoning across semantically distinct prompts that happen to share a token prefix.
    5. **Non-speculative fallback gate**: warm turn must not fall back to non-speculative AR; cache reuse must be physical (i.e. measured by saved prefill tokens, not just hit-count).
  - Phase 2 ships these as `Tests/MLXLMTests/PrefixCacheSurvivalTests.swift` (or extends the existing tests file with the five-contract suite). Phase 4 re-runs the suite end-to-end through L2.
  - The "new prefill accounting fields" in this commit map to our `PrefixCacheStats.prefillTokensSaved` (introduced in phase 1B per the prior section).

- **`4bc72c8` (2026-05-10) — "fix: harden runtime and cache contract failures"**
  - Establishes the **fail-fast contract philosophy** for cache lifecycle and snapshot validation:
    - New error type `RuntimeCacheManagerClosed` raised by every public method after retirement (`_ensure_open_locked()` precondition). Our Swift equivalent: throw a typed `PrefixKVCacheError.closed` from every `lookup` / `insert` / `clear` after `close()` is called. Idempotent shutdown.
    - `maybe_insert_snapshot()` now raises `ValueError("prefix snapshot requires last_logits")` instead of silently returning `0.0`. Our equivalent: phase 1B `insert()` throws `PrefixKVCacheError.missingLogits` when a DFlash-mode snapshot is inserted without `lastHidden`. (Pure-attention/SSM mode snapshots don't require it.)
    - L2 exception narrowing: `OSError` instead of bare `Exception` on init failure. Phase 4 adopts the same — `try?` is forbidden across cache boundaries; specific error types surface upward.
  - **Adopt from day 1**: phase 1B introduces `PrefixKVCacheError` (cases `.closed`, `.missingLogits`, `.formatVersionMismatch`, `.wrappedWindowedCache`, `.snapshotInvariantViolation(String)`). Every `serialise()` / `hydrate(from:)` is `throws`. No silent fallbacks across the cache contract.

**Reference commits (all on `bstnxbt/dflash-mlx@main`, post-`8d8545d`):** `463d722`, `8c29e3e`, `4bc72c8`. Cross-relevant but not prefix-cache-shaping: `ce36f62` / `0972afb` / `e2be8a4` (runtime config / ownership / observability refactors — Swift equivalents live in `Evaluate.swift`'s generate path, no spec change), `05cc456` / `2274b67` (Gemma4 DFlash backend — relevant to spec 015, not 017).

## Open questions

1. **Cross-process sharing.** vLLM's APC is per-process; the same prefix served across multiple instances is recomputed. For our single-process desktop / iOS setting this isn't a concern.
2. **Quantised-cache snapshot fidelity.** Restoring a snapshot with a different KV quantisation than the runtime configures should fail loudly. Plumb through `PrefixKey`.
3. **Windowed cache wrap.** If the snapshot was taken when `offset > maxSize` (sliding-window layers wrapped under `eviction == .window(...)`), the snapshot semantics get tricky — replay would diverge from a fresh prefill. Easiest fix: refuse to snapshot wrapped windowed caches in `StandardKVCache.serialise()`. Document.

## References

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) — hash-based, page-granular, cross-request.
- **Primary upstream (post-refactor):** [`bstnxbt/dflash-mlx@main`](https://github.com/bstnxbt/dflash-mlx) HEAD `8d8545d` (2026-05-04) — see "dflash-mlx upstream updates since spec drafted" above for the file-level breakdown of [`dflash_mlx/cache/`](https://github.com/bstnxbt/dflash-mlx/tree/main/dflash_mlx/cache).
- [dflash-mlx engine-v2 prefix cache](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2/dflash_mlx/cache) — original Python reference (superseded by main).
- [llama.cpp `--prompt-cache`](https://github.com/ggml-org/llama.cpp) — single-snapshot variant of the same idea.
- [Issue #73 / spec 006](https://github.com/ekryski/mlx-swift-lm/issues/73) — KVCache refactor that lands before phase 1B; provides the `StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache` surface that the per-class `serialise()` / `hydrate(from:)` extensions target.
