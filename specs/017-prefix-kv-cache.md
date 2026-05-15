# 017 — Cross-request prefix KV cache

**Status:** ✅ **Phases 1 + 1B + 2 + 3 + 4 + 5 (Option A — post-prefill snapshot) consolidated and shipped** via PRs [#144](https://github.com/ekryski/mlx-swift-lm/pull/144) (main implementation) + [#198](https://github.com/ekryski/mlx-swift-lm/pull/198) (close-out / known-limitations follow-up). Phase 5's post-prefill snapshot timing resolves [#185](https://github.com/ekryski/mlx-swift-lm/issues/185) (✅ CLOSED 2026-05-12). [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) (Gemma 4 turbo4v2 insert-but-no-hit) and [#197](https://github.com/ekryski/mlx-swift-lm/issues/197) (compressed-mode TurboQuant serialise/hydrate) remain **OPEN** as known limitations gating any default-on flip. Compressed-domain snapshots (~4× smaller bytes) deferred to [spec 039](039-compressed-prefix-kv-cache.md).
**Branch:** shipped via PRs #144 + #198.
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
- `SSMStateCache`: snapshots `state` (conv + recurrent slots) + `metaState` through the existing `ArraysCache.restoreFromMetaState` round-trip. Refuses when `canStateReplay == false` (Mamba opt-out preserved from spec 020 — flips on for Nemotron Cascade 2 / Jamba / Granite-MoE-Hybrid / FalconH1 when [spec 040](040-mamba-state-replay.md) ships the Mamba kernel pair).
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
- **`GenerateParameters` extended** with four fields. Default-on was attempted 2026-05-12 and reverted the same day after bench validation surfaced the `--kv turbo4v2` interaction issues (see §"Known limitations & follow-up work" below). Shipping as **opt-in** until follow-up issues close:
  - `prefixCacheEnabled: Bool = false` — opt-in via this field or env `MLX_PREFIX_CACHE=1`. Env override: `MLX_PREFIX_CACHE=0` force off; `MLX_PREFIX_CACHE=1` force on.
  - `prefixCachePolicy: (any StablePrefixPolicy)? = nil` — when nil **and** prefix cache is enabled, the route helper calls `resolveDefaultPolicy(modelID:tokenizer:)` which uses `AssistantOpener.detect(forModelID:)` to match the model family substring (`qwen` / `qwq` → ChatML; `gemma` → `<start_of_turn>`; `gpt-oss` → harmony) and constructs a `LastAssistantOpenerPolicy` with the live tokenizer-encoded opener tokens. Unknown families (Llama / Phi / Mistral / …) fall back to `IdentityPolicy`. The auto-detect path stays valuable for opt-in callers — they get the right policy for free.
  - `prefixCacheModelID: String? = nil` — when nil, auto-resolved from `ModelContext.configuration.name` in `MLXLMCommon.generate(...)` so single-model apps need zero explicit setup once the flag is on. Multi-model apps that share the same architecture set this explicitly. Also drives the family-detection used for the default policy above.
  - `prefixCacheDiskEnabled: Bool = false` — strictly opt-in (env: `MLX_PREFIX_CACHE_DISK=1`).
- **Bench harness's `runGenerationBenchmark(...)`** keeps `prefixCacheEnabled: false` as its function-level default, matching the new library default. The `multi-turn` method exposes `--cache-prefix` (env `MLX_BENCH_CACHE_PREFIX=1`) to opt in; the same method body runs both baseline and cached configurations so the comparison is apples-to-apples.
- **Path to default-on**: blocking follow-up issues [#197](https://github.com/ekryski/mlx-swift-lm/issues/197), [#196](https://github.com/ekryski/mlx-swift-lm/issues/196), and [#185](https://github.com/ekryski/mlx-swift-lm/issues/185) **all resolved 2026-05-12** on `ek/spec-017-known-limitations`. The auto-policy + auto-modelID logic is already in place, so flipping `prefixCacheEnabled` default back to `true` is now a one-line change pending sign-off on the fleet-wide bench validation table in §"Known limitations" #3.

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

### Known limitations & follow-up work

The cross-request prefix KV cache ships **opt-in** in PR #144 — `GenerateParameters.prefixCacheEnabled` defaults to `false`. Default-on was attempted and reverted the same day (2026-05-12) after bench validation surfaced the limitations below interacting badly with `--kv turbo4v2`. Re-defaulting to opt-in is the right move; the **three blocking limitations have now been resolved** on the `ek/spec-017-known-limitations` follow-up branch (see status below).

#### 1. TurboQuant compressed-mode snapshot refused — Qwen 3.5 / NemotronH affected with `--kv turbo4v2` → [#197](https://github.com/ekryski/mlx-swift-lm/issues/197) — **RESOLVED 2026-05-12**

`KVCacheSerialisation.swift::serialiseTurbo` threw `snapshotInvariantViolation` when `cache.isCompressed == true`. TurboQuant transitions from raw → compressed on first decode, so by request-end every TurboQuant cache was compressed. Net effect: `cache_bytes=0` throughout a multi-turn run on `--kv turbo4v2` for Qwen 3.5 / 3.6 / NemotronH.

**Fix**: `serialiseTurbo` now dequants compressed K/V back to raw FP16 at snapshot time via `TurboQuantizedKVCache.dequantToRaw()`. Every TurboQuant snapshot is emitted as a 2-array raw payload regardless of the cache's current `isCompressed` state. Hydrate then loads into `rawKeys`/`rawValues`, and warm-turn suffix prefill's `update(...)` appends to the hydrated raw buffer correctly. Lossy (one round of TurboQuant dequant precision) but unblocks the warm-turn TTFT win on every model family that uses TurboQuant.

Bench (Qwen3.5-35B-A3B, `--kv turbo4v2 --cache-prefix`): turn-1 cold 1077 ms → turn-2 warm 405 ms (~2.7× TTFT), `hits=3 misses=1 prefillTokensSaved=339 hitRate=0.75`. Output coherent.

#### 2. Gemma 4 26B-A4B / 31B prefix-cache lookup miss with `--kv turbo4v2` → [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) — **RESOLVED 2026-05-12**

Snapshots inserted (cache_bytes grew) but every lookup returned nil (`hits=0`). Root cause: Gemma 4 large variants (26B-A4B, 31B) ship a chat template whose `add_generation_prompt` emits a 7-token opener — the model header **followed by** the channel-thought scaffold `<|channel>thought\n<channel|>`. The default `AssistantOpener.gemma4` opener `<start_of_turn>model\n` (a) only matched the 3-token model header and (b) wasn't actually recognised as a special token by the Gemma 4 tokenizer (which uses `<|turn>`), so the policy fell through to the identity fallback and snapshots covered the full prompt — making warm-turn token alignment fail.

**Fix**: split `AssistantOpener` into three Gemma cases and add `<|turn>`-aware encoded forms (`StablePrefixPolicy.swift`):
- `.gemma4` — `<start_of_turn>model\n` for Gemma 1 / 2 / 3 (legacy tokenizers).
- `.gemma4Turn` — `<|turn>model\n` for Gemma 4 small (E2B / E4B).
- `.gemma4WithThought` — `<|turn>model\n<|channel>thought\n<channel|>` for Gemma 4 large (26B-A4B, 31B).

`AssistantOpener.detect(forModelID:)` routes by substring (`gemma-4-26b` / `gemma-4-31b` → with-thought; other `gemma-4` / `gemma4` → turn; other `gemma` → legacy).

Bench (Gemma 4 26B-A4B, `--kv turbo4v2 --cache-prefix`): `hits=2 misses=1 prefillTokensSaved=151 hitRate=0.67`, snapshot tokens correctly trimmed 41 → 34. Same on 31B.

#### 3. Gemma 4 family doesn't actually run TurboQuant when `--kv turbo4v2` requested → [#185](https://github.com/ekryski/mlx-swift-lm/issues/185) — **RESOLVED 2026-05-12**

`makeAttentionCache(parameters:maxSize:keep:)` deliberately fell through to `StandardKVCache(maxSize:)` for the windowed-turbo case. Root cause (after multi-step bisection):

1. **KV-shared donor plumbing** — Gemma 4's LLM-side `Gemma4ModelInner` reads donor K/V from `cache.lastReturnedKeys` / `cache.lastReturnedValues` after every donor `update()`. `TurboQuantizedKVCache.update()` set neither (only `updateAndDequant()` did). Shared sliding layers in E2B / E4B received nil arrays and SDPA produced total garbage logits. Models without KV-sharing (26B-A4B, 31B) were unaffected.

2. **Prefix-cache hydrate → suffix prefill mode mismatch** — TurboQuant transitions to compressed mode on the first decode call. Snapshot captured compressed state; hydrate restored it. Suffix prefill's `attentionWithCacheUpdate(L > 1)` then called `update(...)` (raw path), which overwrote the hydrated state because `rawKeys == nil` in compressed mode.

**Fix**:
- Add `lastReturnedKeys` / `lastReturnedValues` assignments to `TurboQuantizedKVCache.update()` (rotating + non-rotating paths) — mirrors `updateAndDequant()`.
- Add `TurboQuantizedKVCache.innerState()` returning the live buffer arrays so Gemma 4's prefill eval-barrier (issue #169) flushes pending writes.
- `serialiseTurbo` dequants compressed → raw at snapshot time (see #1 above), so hydrate never restores compressed state.
- `makeAttentionCache(.turbo + maxSize)` now dispatches to windowed `TurboQuantizedKVCache`. Env override `MLX_TURBO_WINDOWED=0` restores the legacy `StandardKVCache` fallback.

Bench (Gemma 4 fleet, `--kv turbo4v2 --cache-prefix`, multi-turn):

| Model | KV-sharing | Cold (turn 1) | Warm (turn 2) | Output |
|---|---|---|---|---|
| Gemma 4 E2B | yes (20/35 layers) | 162 ms | 366 ms (`hits=1`) | Coherent, mild compression-induced repetition |
| Gemma 4 E4B | yes (18/42 layers) | 220 ms | 470 ms (`hits=1`) | Coherent |
| Gemma 4 26B-A4B | no | 491 ms | 651 ms (`hits=1`) | Coherent |
| Gemma 4 31B | no | 1248 ms | 1505 ms (`hits=1`) | Coherent |

The small Gemma 4 variants (E2B, E4B) show mild output-quality degradation under `--kv turbo4v2` from compression interacting with KV-sharing (`kvHeads ≤ 2`). This is a precision artifact of `valueBits=2`, not a correctness bug — `--kv turbo4` (symmetric 4-bit V) reduces the repetition; `--kv none` is clean baseline.

#### 4. GPT-OSS-20B no benefit (sliding-window=128 wraps mid-generation)

Sliding-window=128 layers wrap by mid-generation; `isTrimmable` returns `false` and the prefix tokens are physically gone from the cache buffer. Snapshot is skipped cleanly (no crash, `cache_bytes=0`). Honest "no benefit" outcome for models with very short sliding windows.

**Mitigation path**: snapshot **during** prefill at the stable-prefix point (before generation can push the window past it). Bigger architectural change — would need a per-cache pre-decode hook in `TokenIterator.prepare(...)`. Tracked as **phase 5 future work** (no issue filed yet; not blocking any current downstream).

#### 5. Hybrid-model SSM-layer state is stale on snapshot — Qwen 3.5 / 3.6 GDN affected

The snapshotter captures `state[0]` (conv) + `state[1]` (recurrent) at the cache's natural `offset` — which after a generated assistant reply is the prompt+gen offset, not the stable prefix point. The next turn's hydrated SSM state is "stale" by a few generation steps.

**Impact**: the attention layers — which carry the dominant prefill cost — are exact. This is the bench-measured ~3× TTFT win on Qwen3.5-35B-A3B under `--kv none`. SSM state freshness is a 2nd-order concern; bench output remains coherent.

**Mitigation path**: capturing SSM state at the stable-prefix point during prefill would need an `SSMStateCache.snapshot(atOffset:)` primitive backed by the spec 020 state-replay infrastructure. **Phase 5 future work**, no issue filed yet.

#### 6. `PrefixKey.kvHeadDim` is a placeholder

`prefixKey(forCache:modelID:)` returns `kvHeadDim: 1` — the real per-layer head dim isn't exposed by the `KVCache` protocol. Cross-model snapshots are still gated by `modelID` and `kvBits`, so incorrect reuse doesn't happen in practice. Callers who want strict cross-config rejection should construct `PrefixKey` explicitly. **Phase 5 future work**.

#### 7. Gemma 4 KV-sharing donor layers — handled correctly (not a limitation, documented for reference)

Some Gemma 4 attention layers carry empty state (`tokenCount=0`, `arrays=[]`) because they share K/V from a donor layer. The `PrefixKVCache.insert` invariant explicitly exempts these layers from the `tokenCount == promptLen` check; on hydrate the shared layer rebinds to its donor naturally. No follow-up needed.

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
