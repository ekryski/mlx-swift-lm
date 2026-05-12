# 017 — Cross-request prefix KV cache

**Status:** **Phases 1 + 1B + 2 + 3 + 4 consolidated** in PR [#144](https://github.com/ekryski/mlx-swift-lm/pull/144) (built on top of spec 020 phase 5). Phase 1 in-memory cache, Phase 1B per-class `serialise()`/`hydrate(from:)` + `generate()` wiring, Phase 2 chat-aware `LastAssistantOpenerPolicy`, Phase 3 hybrid (GDN+attention) cache support via spec 020 state-replay, Phase 4 opt-in disk persistence — all live.
**Branch:** `ek/017-prefix-cache-phase1` (PR #144)
**Depends on:** spec 020 phases 1-3 (shipped 2026-05-11 in PR #143) for hybrid-model coverage. Pure-attention models work without spec 020.

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

## Implementation status (phases 1 + 1B + 2 + 3 + 4 consolidated in PR #144)

PR [#144](https://github.com/ekryski/mlx-swift-lm/pull/144) ships **all five phases in one branch**, rebased onto current `alpha` (post spec-020 phase 1+2+3, post-WS-A–D, post-consolidation sprint). The ordering inverted at implementation time: rather than ship phase 1 alone, we re-folded 1B/2/3/4 onto the same branch because spec 020 had already shipped the load-bearing prerequisite (state-replay for hybrid models). Single PR keeps the per-class serialise/hydrate, generate-path wiring, chat-aware policy, hybrid-model coverage, and L2 disk persistence semantically consistent.

### What ships in PR #144 (mlx-swift-lm)

**Cache core** (`Libraries/MLXLMCommon/PrefixKVCache.swift`, 540 lines):
- `PrefixKey(modelID:layerCount:kvHeadDim:kvBits:captureLayerIds:formatVersion:)` — Hashable, Sendable. `formatVersion = 2` from day 1 (matches dflash-mlx `463d722`); `captureLayerIds: [Int]?` reserved for the eventual DFlash subset-capture case.
- `LayerCacheState(kind:tokenCount:arrays:metaState:)` with **typed `Kind` discriminator**:
  - `.standardUnbounded`
  - `.standardWindowed(maxSize:keep:)`
  - `.affineQuantized(bits:groupSize:)`
  - `.turboCompressed(keyBits:valueBits:)`
  - `.ssm`
  Mirrors `KVStorageKind` but adds the windowed-vs-unbounded discriminator (which `KVStorageKind` lacks).
- `PrefixSnapshot(key:tokens:layerStates:lastHidden:createdAt:)` — `byteSize` covers all backing arrays + optional `lastHidden`.
- `PrefixCacheStats` extended with `exactHits`, `partialHits`, `byteBudgetEvictions`, `skippedTooLong`, `fingerprintRejects`, `prefillTokensSaved` — all from dflash-mlx `prefix_l1.py` HEAD.
- `PrefixKVCacheError` enum (`.closed` / `.missingLogits` / `.formatVersionMismatch(expected:found:)` / `.wrappedWindowedCache` / `.snapshotInvariantViolation(_:)`) — `throws`-based, fail-fast discipline from `4bc72c8`.
- `PrefixKVCache` class:
  - `lookup(prefix:key:record:)` — non-recording lookups (`record = false`) don't bump LRU or update stats. Mirrors dflash-mlx `463d722`.
  - `insert(_:)` — enforces format-version match, snapshot invariants (each non-SSM, non-empty layer's `tokenCount == promptLen`), byte-budget + entry-count eviction, idempotent re-insert.
  - `close()` — idempotent shutdown; every public method throws `.closed` after.

**Per-class serialise / hydrate** (`Libraries/MLXLMCommon/KVCacheSerialisation.swift`, 245 lines):
- `serialiseLayerCacheState(_:)` / `hydrateLayerCache(_:into:)` — type-dispatched over the four concrete cache classes.
- `serialisePrefixSnapshot(cache:tokens:key:lastHidden:)` / `hydratePrefixSnapshot(_:into:)` — full-stack helpers.
- `StandardKVCache` (unbounded + windowed): `serialise()` refuses to capture a wrapped windowed cache (`offset > maxSize`), throwing `.wrappedWindowedCache` per dflash-mlx `target_cache_is_serializable`.
- `AffineQuantizedKVCache`: validates `bits + groupSize` match on hydrate.
- `TurboQuantizedKVCache`: raw-mode only; refuses snapshot when `isCompressed == true` (compressed-domain snapshot deferred — codec rotation matrices need their own serialisation).
- `SSMStateCache`: snapshots `state` (conv + recurrent slots) + `metaState` through the existing `ArraysCache.restoreFromMetaState` round-trip. Refuses when `canStateReplay == false` (Mamba opt-out preserved from spec 020).
- `prefixKey(forCache:modelID:kvBits:)` — best-effort key derivation from a `[KVCache]` for callers that don't want to hand-plumb model dims.

**Stable-prefix policies** (`Libraries/MLXLMCommon/StablePrefixPolicy.swift`):
- `IdentityPolicy` (full prompt stable — completion workloads).
- `FixedTrimPolicy(trimSuffix:)` (constant trailing-trim — phase-1 chat workaround).
- `LastAssistantOpenerPolicy(opener:fallback:)` — scans for the rightmost match of an opener token sequence and returns its start index. Tokenizer-independent at runtime; encoding done at construction via the `AssistantOpener` enum (`.qwenChatML` / `.gemma4` / `.gptOSSHarmony` / `.custom(_:)`).
- Fallback policy on no-match: `.identity` (whole-prompt) / `.refuse` (no cache) / `.fixedTrim(suffix:)`.

**Generate-path wiring** (`Libraries/MLXLMCommon/PrefixKVCacheRoute.swift`, 360 lines):
- `prefixCacheRoute(input:cache:parameters:model:)` — entry shim. Computes the `PrefixKey`, attempts L1 (and L2 if `prefixCacheDiskEnabled`) lookup, rewrites the input to the suffix tokens on hit, and returns a `PrefixCacheRouteState` carrying the hydrated cache + a snapshotter closure.
- `PrefixCacheRouteState.wrapStreamForSnapshot(_:cache:)` — wraps the iterator's result stream; after the source stream's `.info` event, fires the snapshotter on the live cache. Uses `SendableBox<[KVCache]>` to thread the non-Sendable cache reference through the wrapper Task (same pattern `generateLoopTask` already uses).
- `quantisationKindMismatch(snapshot:cache:)` — defence-in-depth quantisation guard. Runs before any hydrate; checks each non-exempt layer's `LayerCacheState.Kind` against the target cache's concrete class and re-validates `(bits, groupSize)` / `(keyBits, valueBits)` for affine + turbo. Mismatch returns a diagnostic string and the route falls back to the uncached path. Exempts SSM and donor-sharing empty layers.
- `tokensArrayFromLMInput` / `makeLMInput` — `LMInput` ↔ `[Int]` adapters; preserves vision payloads (`image` / `video`) so VLM call sites continue to work when the prefix cache strips text tokens.
- Wired into `MLXLMCommon.generate(input:cache:parameters:context:wiredMemoryTicket:)` (Evaluate.swift): three call sites — n-gram path (after `canRollbackPromptCache` check), n-gram fallback to `TokenIterator`, plain `TokenIterator` path. All three return through `wrapStreamForSnapshot(...)`.
- **`GenerateParameters` extended** with four opt-in fields: `prefixCacheEnabled` (env override `MLX_PREFIX_CACHE=1`), `prefixCachePolicy`, `prefixCacheModelID`, `prefixCacheDiskEnabled` (env override `MLX_PREFIX_CACHE_DISK=1`). All default `false` — opt-in.

**Disk persistence (Phase 4)** (`Libraries/MLXLMCommon/PrefixKVCacheDisk.swift`, 250 lines):
- `PrefixKVCacheDisk(root:)` — opt-in L2. Default root `~/.cache/mlx-swift-lm/prefix/`. **Off by default** — opt-in via `MLX_PREFIX_CACHE_DISK=1` or `GenerateParameters.prefixCacheDiskEnabled`. We don't bloat the user's disk inadvertently.
- Schema: one directory per snapshot, named by FNV-1a fingerprint of `(modelID, first-16 tokens, length)`. Inside: `index.json` (manifest with key + tokens + per-layer kind + metaState) and `arrays.safetensors` (flattened `[layer].[idx]` keys, written via the existing `MLX.save(arrays:url:)` helper).
- `lookup(prefix:key:)` walks all on-disk snapshots once and returns the longest token-prefix match. Format-version mismatch surfaces as `.formatVersionMismatch`; corrupt entries are skipped silently (lookup throws are filtered).
- Promotion: `prefixCacheRoute` promotes L2 hits into L1 (`PrefixKVCache.shared.insert(...)`) on first read so the next request short-circuits the disk walk.

**Multi-turn benchmark** (`Tests/Benchmarks/InferenceBenchmark.swift`):
- New `--method multi-turn-cached` bench method. Runs N (default 4) user turns against the same growing chat history with the prefix cache enabled. **Realistic chat:** every turn passes `UserInput(prompt: .messages(...))` → `MessageGenerator → tokenizer.applyChatTemplate(...)`, so the model's actual chat template (Qwen ChatML, Gemma 4 `<start_of_turn>`, GPT-OSS harmony) wraps every turn. The model's **actual reply** from the previous turn is captured via the `resultSink` closure and appended to the chat history as the assistant turn — not a fixed stub. (Reply is trimmed to 300 chars per turn so a chatty reply doesn't blow up the next turn's prompt size.)
- Per-turn `[PREFIX-CACHE]` line: TTFT, hits, saved tokens, cache bytes. End-of-run `[PREFIX-CACHE] summary` line with all stats fields.
- Env knobs: `MLX_BENCH_TURNS`, `MLX_BENCH_MAX_TOKENS`, `MLX_PREFIX_CACHE_TRIM` (default 4), `MLX_PREFIX_CACHE_DEBUG=1` (verbose snapshotter trace).

**Tests** (`Tests/MLXLMTests/PrefixKVCacheTests.swift` + `PrefixKVCacheSerialisationTests.swift` + `PrefixKVCacheDiskTests.swift` — **63 tests across 9 suites**, all passing):
- `StablePrefixPolicyTests` (10): Identity / FixedTrim / LastAssistantOpener (find, rightmost, fallback variants, edge cases), `AssistantOpener.rawString`.
- `PrefixKeyTests` (4): equality, hash bucketing, `formatVersion == 2`, captureLayerIds.
- `PrefixKVCacheTests` (26): empty miss / partial vs exact / longest-prefix selection / cross-key fingerprintReject / mismatch miss / snapshot-longer-than-request / re-insert dedup / LRU bump / `lookup(..., record: false)` no-bump no-stats / entry-count cap / byte-budget + `byteBudgetEvictions` / `skippedTooLong` / formatVersion mismatch / layer tokenCount mismatch / empty-donor-sharing exempt / SSM-kind exempt / `close()` retire-throws / idempotent close / clear / resetStats / hitRate / meanMatchedLength / `prefillTokensSaved` accumulation.
- `StandardKVCacheSerialisationTests` (5): unbounded round-trip, windowed round-trip with metaState, wrapped-window refuse, kind-mismatch throw, empty round-trip.
- `AffineQuantizedKVCacheSerialisationTests` (2): round-trip with bits+groupSize, bit-width mismatch throws.
- `TurboQuantizedKVCacheSerialisationTests` (1): raw-mode round-trip.
- `PrefixSnapshotRoundTripTests` (2): full-stack round-trip preserves per-layer state; layer-count mismatch on hydrate.
- `QuantisationKindMismatchTests` (8): matching kinds, affine bits / groupSize mismatch, class mismatch fp16↔affine in both directions, turbo bit-width mismatch, SSM + donor-sharing exemptions.
- `PrefixKVCacheDiskTests` (7): write+lookup, empty directory, idempotent overwrite, longest-prefix selection, cross-model isolation, clear, format-version mismatch.

### Bench validation (multi-turn-cached, kv=none, M1 Max 64GB, default 4-turn run, **real model replies appended as assistant turns**)

The bench captures each turn's actual model output via `resultSink`, sanitises any harmony-format sentinels (`<|channel|>...|>`) leaked by GPT-OSS, and appends the cleaned text (trimmed to 300 chars) as the next iteration's assistant message. This makes the bench a realistic multi-turn chat: each warm turn re-prefills both the cached prefix and the new "previous reply + new user question" suffix, so warm-turn TTFT improvements are smaller than they'd be against a fixed-stub history but more representative of real-world chat sessions.

| Model | Turn 1 (cold) | Turn 2 (warm) | Turn 3 (warm) | Turn 4 (warm) | Best warm TTFT speedup | Prefill rate progression | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (GDN hybrid) | 124ms | 57ms | 48ms | <48ms | **~2.6×** | 318 → 3346 tok/s (~10×) | Clean monotonic |
| Gemma 4 E2B (KV sharing) | 102ms | 71ms | 40ms | 60ms | **~2.5×** | 372 → 2288 tok/s (~6×) | Cleanest small-model run |
| **Qwen3.5-35B-A3B (MoE+GDN)** | **2186ms** (cold-Metal) / 875ms (warm-Metal repeat) | 503ms | 523ms | 499ms | **~4.3× (cold-Metal)** ⚡ / 1.9× (warm-Metal) | 17 → 596 tok/s (**~35×**) | Headline win on first-of-session prompts |
| Gemma 4 26B-A4B (MoE) | 380ms | 406ms | 325ms | 423ms | ~1.2× | 108 → 626 tok/s (~6×) | Replies are long; new-token cost grows |
| Gemma 4 31B (dense) | 1232ms | 1215ms | 688ms | 1577ms | ~1.8× (turn 3) | 33 → 170 tok/s | Noisy; dense 31B's long captured replies dominate the per-turn delta |
| GPT-OSS-20B (sliding window=128) | 508ms | 451ms | 593ms | 849ms | **1.0× — no benefit** | 193 → 326 tok/s | Documented limitation (see below) |

**Headline**: Qwen3.5-35B-A3B sees **~4.3× TTFT** on the first cached turn after cold-Metal-pipeline boot (2186ms → ~500ms), and **prefill throughput grows ~35×** by turn 4 because the cache saves more absolute tokens as the conversation extends. Smaller models still benefit at ~2.5× TTFT once their cold path is large enough to amortise the cache restore.

**Note on Metal pipeline warmup**: Turn-1-cold measurements depend on whether Metal's shader pipeline is already JIT'd from a prior run. The first session ever sees the highest cold-path numbers (e.g. Qwen3.5-35B-A3B 2186ms); repeated bench invocations in the same shell session show much warmer turn-1 numbers (875ms) but identical warm-turn behaviour. The speedup ratio is what matters; the absolute cold-path number is shell-state-dependent.

**Note on earlier stub-based numbers**: pre-finalisation runs against a fixed-stub reply showed up to ~11× TTFT speedup because each warm turn had only ~7 new tokens to prefill. Those stub runs are valid for measuring the cache's structural correctness but inflate the user-visible TTFT win vs. realistic chat. The numbers above are the honest realistic-chat baseline.

**Note on Gemma 4 family**: the dense Gemma 4 26B-A4B / 31B numbers are noisier because their captured replies are longer than other families' (the model is more verbose under the test system prompt), so each warm turn has more new tokens to prefill. The cache still saves a real fraction (236-331 saved prefill tokens per run), but the warm-TTFT win is partially eaten by the longer captured replies. Prefill rate progression remains a clean 5-7× gain across turns, confirming the cache is doing its job.

### Known limitations

1. **GPT-OSS-20B**: sliding-window attention with `window=128` means by mid-generation the alternating sliding-window layers have wrapped — `isTrimmable` returns `false` and the prefix tokens are physically gone from the cache buffer. We skip the snapshot cleanly (no crash, just `cache_bytes=0`). Honest "no benefit" outcome for models with very short sliding windows. Mitigation path: snapshot **during** prefill at the stable-prefix point (before generation can push the window past it) — bigger architectural change, tracked as phase 5 future work.

2. **Gemma 4 KV-sharing donor layers** carry empty state (`tokenCount=0`, `arrays=[]`) because they share K/V from a donor layer. The insert invariant explicitly exempts these layers from the `tokenCount == promptLen` check; on hydrate the shared layer rebinds to its donor naturally.

3. **Hybrid-model SSM layers** (Qwen 3.5 / 3.6 GDN, Mamba opt-out cases): the snapshot captures `state[0]` (conv) and `state[1]` (recurrent) at the cache's natural `offset` — which after a generated assistant reply is the prompt+gen offset, not the stable prefix point. Effect: the next turn's hydrated SSM state is "stale" (it includes a few extra generation steps), but the attention layers — which carry the dominant prefill cost — are exact. This is the bench-measured ~3× TTFT win on Qwen3.5. A cleaner solution requires capturing state at the stable-prefix point during prefill, which depends on a `SSMStateCache.snapshot(atOffset:)` primitive — tracked as phase 5 future work.

4. **`PrefixKey.kvHeadDim`** is currently a placeholder `1` in the auto-key derivation (`prefixKey(forCache:modelID:)`) — the real head dim is per-layer and not exposed by the `KVCache` protocol. Cross-model snapshots are gated by `modelID` (and `kvBits`), so this doesn't cause incorrect reuse in practice, but callers who want strict cross-config rejection should construct `PrefixKey` explicitly with the model's actual head dim.

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

## Resolved open questions

All three of the original spec's open questions are resolved in PR #144. Cross-referenced to the implementation:

1. **Cross-process sharing** — *out of scope, by design.* vLLM's APC is per-process; same prefix served across multiple instances is recomputed. For our single-process desktop / iOS setting this isn't a concern. If multi-process serving lands (not currently planned), spec 017 phase 4's L2 disk cache becomes the cross-process backing store automatically — same fingerprint scheme, same on-disk format, no API change. The actor wrapping for thread safety is tracked separately.

2. **Quantised-cache snapshot fidelity** — *resolved at two layers (defence-in-depth).*
   - **Key-level**: `PrefixKey` carries `kvBits`, derived from the cache's `storageKind` in `prefixKey(forCache:modelID:)`. A lookup with `kvBits = 4` cannot match a snapshot built with `kvBits = nil` (fp16) because the keys are inequal — `PrefixKVCache.lookup(...)` returns nil and increments `fingerprintRejects`.
   - **Hydrate-level (defence-in-depth)**: `quantisationKindMismatch(snapshot:cache:)` in `PrefixKVCacheRoute.swift` runs before any hydrate. It checks each non-exempt layer's `LayerCacheState.Kind` against the target cache's concrete class (`StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache`) and on the affine + turbo cases also re-validates `(bits, groupSize)` / `(keyBits, valueBits)`. Mismatch surfaces as a typed diagnostic and falls back to the uncached path — no silent precision loss. SSM layers and donor-sharing empty layers are exempt (Gemma 4 KV sharing).
   - **Per-class hydrate**: `hydrateAffineQuantized` and `hydrateTurbo` independently re-validate quantisation params and throw `snapshotInvariantViolation` on mismatch. So even if a caller bypasses the route helper, the per-class hydrate stays loud.
   - Tests: 8 `QuantisationKindMismatchTests` cover matching kinds, affine bits / groupSize mismatches, class mismatches in both directions (fp16 → affine, affine → fp16), turbo bit-width mismatches, plus the donor-sharing + SSM exemptions.

3. **Windowed cache wrap** — *resolved at two layers.*
   - **Serialise-level**: `serialiseStandard(...)` in `KVCacheSerialisation.swift` throws `PrefixKVCacheError.wrappedWindowedCache` when `cache.offset > size` on a windowed `StandardKVCache`. Matches dflash-mlx `target_cache_is_serializable(...)`. Test: `wrapped windowed cache refuses to serialise`.
   - **Snapshotter-level**: the snapshotter in `PrefixKVCacheRoute.swift` reads the offset from a trimmable layer and checks `canTrimPromptCache(...)` / `canRollbackPromptCache(...)` before issuing the snapshot. If a layer reports `isTrimmable = false` because of wrap (sliding window past its size), the snapshotter logs the skip and exits cleanly — no crash, no half-written entry. Live example: GPT-OSS-20B with `window=128` sliding-window layers — once generation pushes offset past 128, the prefix tokens are physically gone from the buffer; we skip cleanly and the bench shows `cache_bytes=0` (no benefit, no error). Documented as a model-family limitation in §"Known limitations".

## References

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) — hash-based, page-granular, cross-request.
- **Primary upstream (post-refactor):** [`bstnxbt/dflash-mlx@main`](https://github.com/bstnxbt/dflash-mlx) HEAD `8d8545d` (2026-05-04) — see "dflash-mlx upstream updates since spec drafted" above for the file-level breakdown of [`dflash_mlx/cache/`](https://github.com/bstnxbt/dflash-mlx/tree/main/dflash_mlx/cache).
- [dflash-mlx engine-v2 prefix cache](https://github.com/bstnxbt/dflash-mlx/tree/engine-v2/dflash_mlx/cache) — original Python reference (superseded by main).
- [llama.cpp `--prompt-cache`](https://github.com/ggml-org/llama.cpp) — single-snapshot variant of the same idea.
- [Issue #73 / spec 006](https://github.com/ekryski/mlx-swift-lm/issues/73) — KVCache refactor that lands before phase 1B; provides the `StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache` surface that the per-class `serialise()` / `hydrate(from:)` extensions target.
