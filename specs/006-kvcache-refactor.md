# 006 — KVCache architecture refactor

- **Status:** ✅ Shipped — KV cache type consolidation landed via PRs [#163](https://github.com/ekryski/mlx-swift-lm/pull/163)–[#166](https://github.com/ekryski/mlx-swift-lm/pull/166). `StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache` / `SSMStateCache` hierarchy is the current state. Closes [#73](https://github.com/ekryski/mlx-swift-lm/issues/73).
- **Branch:** shipped via PRs #163–#166 (see Migration strategy)
- **Tracked in:** [GitHub issue #73](https://github.com/ekryski/mlx-swift-lm/issues/73) (CLOSED)
- **Owner:** Eric

## Goals

1. Kill `maybeQuantizeKVCache`; make each cache responsible for its own lifecycle.
2. Make `kv=turbo*` / `kv=affine*` / sliding-window / sink-token combinations declarative and safe.
3. Compose along **two axes** (storage, eviction) — not three levels of inheritance.
4. Land **before** specs 020 + 017 phase 1B so their cross-cutting concerns (`TapeReplayCache` extension, per-class `serialise()` / `hydrate(from:)`) target the cleaned surface.

One-liner of the new API:

```swift
let cache = makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2),
                        eviction: .window(size: 1024))
```

## Why this lands before specs 020 + 017 phase 1B

Both downstream specs add per-cache-class methods to the existing tree:

- Spec 020 phase 2 extends `MambaCache` with `TapeReplayCache` conformance — `MambaCache` is outside this spec's storage/eviction axes (it's SSM state, not K/V), so it's clean either way. But its **iterator wiring touches `canTrimPromptCache` → `canRollbackPromptCache`**, which has to remain consistent with the new `StandardKVCache` (consolidated `KVCacheSimple` + `RotatingKVCache`) `isTrimmable` semantics.
- Spec 017 phase 1B adds `serialise()` / `hydrate(from:)` to **every** concrete cache type — bolting these onto `KVCacheSimple` + `RotatingKVCache` + `QuantizedKVCache` + `TurboQuantKVCache` then ripping out three of those four in spec 006 PR 2 is wasted churn. **Land spec 006 PR 1 first; spec 017 phase 1B's per-class methods target `StandardKVCache` / `AffineQuantizedKVCache` / `TurboQuantizedKVCache`.**

## Design

### Pushback on a strict inheritance hierarchy

The original proposal was `KVCache → RotatingKVCache → CompressedKVCache`. Three reasons we don't go that way:

1. **Inheritance forces rotation onto compression even when it doesn't make sense.** TurboQuant is two-phase: prefill = raw FP16, decode = compressed; rotating turbo has no good definition.
2. **Rotation and compression aren't the only axes.** `MambaCache` / `ArraysCache` / `CacheList` / `ChunkedKVCache` aren't on a "K/V storage + eviction" continuum — they need to conform to `KVCache` for the generation loop, but their internals are unrelated.
3. **Swift doesn't have mixins.** Protocol default implementations can't access stored properties they don't declare; structural composition via inheritance forces class explosions (`RawRotatingCache`, `AffineRotatingCache`, `TurboRotatingCache`, …).

**Recommendation: composition for storage + eviction, inheritance only for the abstract-base ↔ concrete-class relationship (`BaseKVCache` already does this).**

### Axes

Two orthogonal axes, one factory:

1. **Storage** — how K/V tensors are held: raw FP16/BF16, affine-quantized, turbo-compressed.
2. **Eviction** — when old tokens get dropped: never, sliding window with optional sink tokens.

`MambaCache` (SSM state) and `CacheList` (composite) stay as they are — outside the factory.

### Types

```swift
public enum KVStorage: Sendable {
    case raw
    case affine(bits: Int, groupSize: Int = 64, startOffset: Int = 0)
    case turbo(keyBits: Int, valueBits: Int, seed: UInt64 = 42)
}

public enum KVEviction: Sendable {
    case unbounded
    case window(size: Int, keep: Int = 0)
}

extension KVCache {
    /// User-facing compression-scheme enum, parsed from CLI / `GenerateParameters.kvScheme`.
    /// Nested under `KVCache` because it's scoped to the cache layer (decision: 2026-05-04).
    public enum CompressionAlgorithm: Sendable, CustomStringConvertible {
        case none
        case affine(bits: Int, groupSize: Int = 64)
        case turbo(keyBits: Int, valueBits: Int)
        public var description: String { ... }
        public init?(_ string: String) { ... }     // "turbo4v2" → .turbo(4, 2); single source of truth
    }
}

public enum KVStorageKind: Sendable {
    case raw
    case affineQuantized(bits: Int, groupSize: Int)
    case turboCompressed(keyBits: Int, valueBits: Int)
    case ssm
    case composite
}
```

### Factory

```swift
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded
) -> any KVCache
```

`makeKVCache(scheme: .turbo(...), eviction: .window(...))` triggers a precondition trap. Rationale: TurboQuant is a two-phase design with no clean definition of "evict a token from the compressed store". For sliding-window-quantized, use `.affine(bits:)`.

### Concrete classes

| Class | Replaces | Storage | Eviction |
|---|---|---|---|
| `StandardKVCache` | `KVCacheSimple` (already aliased) + `RotatingKVCache` (new typealias) | Raw | `.unbounded` \| `.window` |
| `AffineQuantizedKVCache` | `QuantizedKVCache` (new typealias) | Affine | `.unbounded` \| `.window` |
| `TurboQuantizedKVCache` | `TurboQuantKVCache` (new typealias — **user-added rename for symmetry with `AffineQuantizedKVCache`**) | Turbo | `.unbounded` only — precondition enforced |
| `SSMStateCache` | `MambaCache` (renamed in PR 1; typealias removed in PR 2) | SSM state | N/A |
| `CacheList` | (unchanged) | Composite | N/A |
| `ChunkedKVCache` | **PR 1: audit to confirm scope of usage**; PR 2 either deletes (if unused) or folds into `StandardKVCache` as a `.chunked` eviction variant. |

### Self-transitioning storage (kills `maybeQuantizeKVCache`)

Currently the generation loop calls `maybeQuantizeKVCache(&cache, kvBits: …)` once per step to swap a raw cache for a quantized one at `startOffset`. Move this transition **inside** `AffineQuantizedKVCache`:

```swift
public final class AffineQuantizedKVCache: BaseKVCache {
    private var raw: RawStorage
    private var quantized: QuantizedStorage?
    private let startOffset: Int

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if let q = quantized { return q.update(keys: keys, values: values) }
        let result = raw.update(keys: keys, values: values)
        if offset >= startOffset {
            quantized = raw.quantize(...)
            raw.clear()
        }
        return result
    }
}
```

Same pattern for `TurboQuantizedKVCache` (it already does this internally; just remove the external swap path). Result: **the generation loop has no cache-type dispatch**. It calls `cache.update(…)`. Every cache knows its own lifecycle.

### `storageKind` property

Replaces ad-hoc `cache as? QuantizedKVCacheProtocol` downcasts in `AttentionUtils.attentionWithCacheUpdate`:

```swift
extension KVCache {
    public var storageKind: KVStorageKind { .raw }   // default
}
```

Each concrete class overrides. `AttentionUtils` switches on `storageKind` instead of downcasting. Cleaner, type-safer, model code stays blind to concrete cache classes.

## Migration strategy (4 PRs — expanded from 2 on 2026-05-04 as work scaled)

The original draft split this into 3 PRs (additive-only types → migration → cleanup). User compressed to 2 on 2026-05-05 to ship more aggressively. As implementation progressed the work was further split into 4 small bisect-friendly PRs to keep each PR's review surface tight and let specs 020 + 017 phase 1B unblock after PR 1 lands. PRs 1-3 have shipped; PR 4 is the final swap-elimination pass.

### PR 1 — Typed surface lands end-to-end (medium-large) — **PRECONDITION FOR SPECS 020 + 017 PHASE 1B** ✅ shipped

- Add `KVStorage`, `KVEviction`, `KVCache.CompressionAlgorithm`, `KVStorageKind` enums.
- Add `StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`, `SSMStateCache` classes; `makeKVCache(...)` factory.
- Add typealiases for old names: `typealias KVCacheSimple = StandardKVCache`, `typealias RotatingKVCache = StandardKVCache` (with a `convenience init(maxSize:keep:step:)` so old call sites compile), `typealias QuantizedKVCache = AffineQuantizedKVCache`, `typealias TurboQuantKVCache = TurboQuantizedKVCache`, `typealias MambaCache = SSMStateCache`.
- Add `storageKind` to the `KVCache` protocol with default `.raw`; concrete overrides on every cache class.
- **Switch `AttentionUtils.attentionWithCacheUpdate`** to `storageKind`-based dispatch (replaces `cache as? TurboQuantKVCache` + `cache as? QuantizedKVCacheProtocol` downcasts).
- 12 byte-equivalence tests; smoke matrix on 7 model families.

### PR 2 — Cleanup (typealiases + ChunkedKVCache + protocol surface) ✅ shipped

- Remove all 5 typealiases (`KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, `TurboQuantKVCache`, `MambaCache`). Forces remaining callers to migrate.
- Delete `ChunkedKVCache` (audit confirmed unused outside one parametric test).
- Drop `QuantizedKVCacheProtocol` as a public type — `AttentionUtils` uses `as? AffineQuantizedKVCache` directly.
- Update README + benchmarks/README + skill docs to drop legacy class names.

### PR 3 — Typed `KVCache.CompressionAlgorithm` API (medium) ✅ shipped

- Replace 4 legacy `GenerateParameters` fields (`kvBits`, `kvGroupSize`, `quantizedKVStart`, `kvScheme: String?`) with a single typed `compressionAlgorithm: KVCache.CompressionAlgorithm?`.
- Refactor `maybeQuantizeKVCache` to take the typed enum; switch on `case let .turbo` / `case let .affine`.
- Migrate `Qwen35` + `NemotronH` factories to read `parameters.compressionAlgorithm` and construct `TurboQuantizedKVCache` up-front when `.turbo` is set (closes the turbo JIT-warmup hot path).
- Refactor `WiredMemoryUtils.kvBytesPerTokenPerHead(...)` + `estimateKVBytes(...)` to take typed `algorithm`.
- `parseTurboScheme(_:)` retained for legacy benchmark callsites; new code uses `KVCache.CompressionAlgorithm.init?(_:)`.

### PR 4 — Eliminate `maybeQuantizeKVCache` swap (medium)

- Add `makeAttentionCache(parameters:maxSize:keep:)` helper that returns `AffineQuantizedKVCache` when `parameters.compressionAlgorithm == .affine`, else `StandardKVCache`. One canonical entry point for every model factory's attention-layer cache slot.
- Migrate **every** model's `newCache(parameters:)` to `makeAttentionCache(...)`: dense LLMs (Qwen35, NemotronH, AfMoE, Exaone4, GPT-OSS, MiMoV2Flash, Mistral3Text, Olmo3, Gemma3Text, Gemma3nText, Gemma4 LLM + VLM, Pixtral, Mistral3 VLM, Qwen35 VLM) and hybrids (BaichuanM1, FalconH1, GraniteMoeHybrid, Jamba, LFM2, LFM2MoE, LFM2VL, Qwen3Next). Plus the `LanguageModel where Self: KVCacheDimensionProvider` default. Hybrids use `makeAttentionCache` only on the attention slot of the `CacheList`; SSM slot stays `SSMStateCache()`.
- **Delete `maybeQuantizeKVCache`** entirely. Every cache is constructed correctly up-front; the swap is now dead code.
  - Removes the closure indirection (`quantizeKVCache: (inout [KVCache]) -> Void`) from `TokenIterator`, `SpeculativeTokenIterator`, and `NGramSpeculativeTokenIterator`.
  - Removes the 4 step-time + reset-time call sites (`Evaluate.swift` step, `NgramSpeculativeDecoding.swift` × 4, `WiredMemoryUtils.swift` × 2, `InferenceBenchmark.swift` wikitext PPL).
- **Per-cache compression scheme** in `SpeculativeTokenIterator`: add `GenerateParameters.draftCompressionAlgorithm: KVCache.CompressionAlgorithm?` (defaults to `nil`, falls back to `compressionAlgorithm`). When set differently from the main algorithm, the iterator constructs a draft-specific `GenerateParameters` and passes it to `draftModel.newCache(...)`. Unblocks main-turbo + draft-affine combinations.
- **Behaviour deltas vs PR 3 swap**:
  - `turbo` boundary-skip (skip first/last 2 layers) is dropped — Qwen35/NemotronH factories construct turbo for *every* attention layer. PR 3 already shipped without boundary-skip on the up-front turbo path; this just removes the dead `maybeQuantizeKVCache.turbo` codepath that would never have fired (early-return on existing `TurboQuantizedKVCache`). Restoring boundary-skip is a separate follow-up if PPL/quality regressions appear.
  - In SpeculativeTokenIterator, the prior gate "if either main or draft doesn't support turbo, disable on both" goes away. Each model's factory makes its own decision based on `supportsTurboQuantization`; `draftCompressionAlgorithm` lets the caller force a different draft algorithm.
- **Risk:** medium. Hot-path code; affects every generate() call. Mitigation: full smoke matrix (7 models × 6 cells) + bench parity gate at `kv=none`.

## Acceptance criteria

1. `scripts/benchmark.sh` with every `--kv` value runs end-to-end on `qwen35-{0.8,2,4,9,27}b`, `qwen35-35b-a3b`, `gemma4-{e2b,e4b,31b,26b-a4b}`, `gpt-oss-20b`, `nemotron-30b-a3b` (and `nemotron-cascade-2-30b-a3b-4bit` once spec 020 lands).
2. **`kv=turbo4v2` reduces `KV Cache` bytes ≥ 3× vs `kv=none`** at ctx=1024 on every windowed model.
3. **`kv=affine4` works on sliding-window layers** (currently a no-op there).
4. Decode tok/s at `kv=none` **unchanged within ±2%** on all benchmarked models. This refactor is correctness, not perf.
5. KL divergence on Qwen3.5-4B at `kv=turbo4v2` and `kv=affine4` within the same band as on `ek/tom-eric-moe-tuning`.
6. `maybeQuantizeKVCache` deleted; no internal call sites remain (PR 4).
7. `newCache(parameters:)` implementations route through `makeAttentionCache(parameters:maxSize:keep:)` for attention-layer caches (PR 4).
9. `SpeculativeTokenIterator` supports per-cache compression schemes via `GenerateParameters.draftCompressionAlgorithm` (PR 4).
8. Serialization round-trip preserved — existing `.safetensors` checkpoints with old class names continue to load (back-compat loader at the `metaState`-keyed lookup table at [KVCache.swift:1482-1483](../Libraries/MLXLMCommon/KVCache.swift)).

## Files touched (PR 1)

| File | What |
|---|---|
| `Libraries/MLXLMCommon/KVCacheTypes.swift` (new) | `KVStorage`, `KVEviction`, `KVCache.CompressionAlgorithm`, `KVStorageKind` enums + `makeKVCache(scheme:eviction:)` factory + default `storageKind` extension. |
| `Libraries/MLXLMCommon/KVCache.swift` | `StandardKVCache` (consolidates `KVCacheSimple` + `RotatingKVCache`, including `convenience init(maxSize:keep:step:)` for back-compat); `AffineQuantizedKVCache` (renames + extends `QuantizedKVCache`); `SSMStateCache` (renames `MambaCache`); typealiases for back-compat; `storageKind` overrides on each. Port `RotatingKVCache.reserve(_:)` to `StandardKVCache` (only meaningful when `eviction == .window`). Self-transition logic for `AffineQuantizedKVCache`. |
| `Libraries/MLXLMCommon/TurboQuantKVCache.swift` | Class rename `TurboQuantKVCache` → `TurboQuantizedKVCache`; `precondition(eviction == .unbounded)` guard at construction; typealias for back-compat. Self-transition logic moves into the class itself. |
| `Libraries/MLXLMCommon/AttentionUtils.swift` | `attentionWithCacheUpdate` switches from `as?` downcasts to `cache.storageKind` dispatch. Same dispatch shape; cleaner type hygiene. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Delete `maybeQuantizeKVCache` (5 call sites updated to rely on cache self-transition); update `GenerateParameters.kvScheme: String?` to optionally accept `KVCache.CompressionAlgorithm` directly (`String?` stays for back-compat at the public API). |
| `Libraries/MLXLLM/Models/*.swift` (~13 files) | Each model's `newCache(parameters:)` migrates to `makeKVCache(scheme:eviction:)`. Existing per-layer dispatch (e.g., `MambaCache` for linear layers, `RotatingKVCache` for sliding windows) preserved via the typed enum. |
| `Libraries/MLXVLM/Models/*.swift` (~3 files) | Same migration for VLM models. |
| `Tests/MLXLMTests/KVCacheTests.swift` | 12 new tests: byte-equivalence (`testStandardKVCacheUnboundedMatchesKVCacheSimple`, `testStandardKVCacheWindowMatchesRotatingKVCache`, `testAffineQuantizedKVCacheMatchesQuantizedKVCache`, `testTurboQuantizedKVCacheRenameRoundTrip`, `testSSMStateCacheRenameRoundTrip`); factory + parser (`testMakeKVCacheFactoryAllSchemes`, `testMakeKVCacheTurboWithWindowPreconditionFails`, `testCompressionAlgorithmStringParseRoundTrip`); dispatch (`testKVStorageKindDispatchOnEveryCacheType`); self-transition (`testAffineQuantizedKVCacheSelfTransitionsAtStartOffset`, `testTurboQuantizedKVCacheSelfTransitionsAtFirstDecodeStep`); back-compat (`testGenerateLoopWorksWithoutMaybeQuantizeKVCache`). |
| `README.md` + `benchmarks/README.md` | ~5 class-name mentions update to new primary names. |

## Risks

| Risk | Mitigation |
|---|---|
| `StandardKVCache` handling both unbounded + windowed paths makes `update` more branchy than `RotatingKVCache`'s today | Internal `updateUnbounded(...)` / `updateWindowed(...)` keep the legacy bodies byte-identical; public `update(...)` dispatches via `eviction`. Bench at `kv=none` ctx=1024 on sentinel models; require ≤2% regression. |
| Out-of-tree subclasses of `KVCacheSimple` / `RotatingKVCache` break | In-tree audit: zero subclasses today. Out-of-tree paths get one release of typealias deprecation warning. |
| Self-transitioning caches have race between update + transition | Already the case for `TurboQuantKVCache`; we're formalizing, not introducing. Fuzz-test `update` at the `startOffset` boundary. |
| `kv=turbo*` users expect sliding-window and get unbounded | Precondition trap at `makeKVCache`; not silent. Error message points to `kv=affine*` for windowed-quantized. |
| GPT-OSS's inlined attention bypasses `attentionWithCacheUpdate` (won't auto-pick up quantized routing) | Already covered in spec 005; that spec migrates GPT-OSS to `attentionWithCacheUpdate` as part of its scope. |
| Combinatorial growth if new storage types appear later | Composition handles it: add new case to `KVStorage`; one new class or reuse one. No hierarchy reshape. |
| `RotatingKVCache.reserve(_:)` from PR #152 | Port `reserve(_:)` to `StandardKVCache` (only meaningful when `eviction == .window`) during PR 1 — small extra hunk; cleaner from day 1. |
| `isTrimmable` semantics on consolidated `StandardKVCache` | Today `KVCacheSimple.isTrimmable = true`, `RotatingKVCache.isTrimmable = (offset < maxCacheSize)`. Consolidated impl returns the eviction-strategy-appropriate value: `true` for `.unbounded`, conditional for `.window`. Spec 020's `canRollbackPromptCache` predicate handles both. |

## Decisions (resolved 2026-05-04)

1. **`ChunkedKVCache`**: PR 1 audits every call site (grep + manual trace) to confirm scope of usage. PR 2 either deletes (if unused) or folds into `StandardKVCache` as a `.chunked` eviction variant. Decision in PR 2 based on audit.
2. **`MambaCache` → `SSMStateCache`**: rename **in scope for PR 1**. `MambaCache` is misleading — Qwen3.5 calls it a "linear attention" cache (`isLinear` → `MambaCache`) even though the model is GatedDeltaNet. The cache holds SSM state, not Mamba-specific state. Typealias kept for one release.
3. **Speculative decoding**: `SpeculativeTokenIterator` has separate main / draft `[KVCache]` and they **likely will use different schemes** (e.g., main = turbo, draft = none). Persistence path must round-trip two different schemes per snapshot. Add a regression test in PR 1 that round-trips a main+draft cache pair through `savePromptCache` / `loadPromptCache` with different schemes.
4. **Deprecation timeline**: **one release with typealiases**, then remove in PR 3. Refactor moves fast; downstream callers get one release window to migrate.
5. **Compression-algorithm enum location**: **nested under `KVCache`**. Renamed from `KVScheme` → `KVCache.CompressionAlgorithm`. Reasoning: scoped to the cache layer; clearer at the call site (`KVCache.CompressionAlgorithm.turbo(keyBits: 4, valueBits: 2)` reads as a configuration of the cache, not a separate concept). `KVStorage` + `KVEviction` stay top-level for now (internal axes; can be nested later if it improves call-site clarity).

## What we keep

- `BaseKVCache` abstract class — useful for shared `offset`, default `memoryBytes`.
- `SSMStateCache` (was `MambaCache`) / `ArraysCache` — specialized for SSM state.
- `CacheList` — composite pattern is right for heterogeneous layer types.
- `KVCache` protocol surface — already the right shape.
- All TurboQuant compression math — hard-won correctness; only construction path changes.
- Persistence formats (`state`, `metaState`) — ship-compat.

## What we delete

- `maybeQuantizeKVCache` (external swap function) — **PR 4** (every model factory constructs the right cache class up-front via `makeAttentionCache`; the swap becomes dead code).
- `RotatingKVCache` as a distinct class — **PR 1** (consolidated into `StandardKVCache`; typealias removed in PR 2).
- `QuantizedKVCacheProtocol` as a public type (use `storageKind`) — **PR 2**.
- `ChunkedKVCache` — **PR 2** (audit confirmed zero model usage, only one parametric test).
- All typealiases (`KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, `TurboQuantKVCache`, `MambaCache`) — **PR 2** (forces all callers to migrate to new names).
- 4 legacy `GenerateParameters` fields (`kvBits`, `kvGroupSize`, `quantizedKVStart`, `kvScheme: String?`) — **PR 3** (folded into typed `compressionAlgorithm: KVCache.CompressionAlgorithm?`).
- `quantizeKVCache` closure indirection in `TokenIterator` / `SpeculativeTokenIterator` / `NGramSpeculativeTokenIterator` — **PR 4** (no longer needed once the swap is gone).

## What we rename (in PR 1; typealiases bridge old → new names; PR 2 removes typealiases)

- `KVCacheSimple` → `StandardKVCache`. Existing `public typealias StandardKVCache = KVCacheSimple` at [KVCache.swift:1670](../Libraries/MLXLMCommon/KVCache.swift); flip which is the primary name.
- `RotatingKVCache` → `StandardKVCache` (consolidated, eviction strategy as a stored property).
- `QuantizedKVCache` → `AffineQuantizedKVCache`. Symmetric with `TurboQuantizedKVCache`.
- `TurboQuantKVCache` → `TurboQuantizedKVCache`. Symmetric with `AffineQuantizedKVCache`.
- `MambaCache` → `SSMStateCache`. The cache holds SSM state generally; "Mamba" is misleading since Qwen3.5 uses GatedDeltaNet, not Mamba. (Decision 2026-05-04, in scope for PR 1.)
- `KVScheme` (top-level) → `KVCache.CompressionAlgorithm` (nested). User-facing scheme stays as-is at the call site (`generateParams.kvScheme`); only the type spelling changes.

## References

- [GitHub issue #73](https://github.com/ekryski/mlx-swift-lm/issues/73) — original proposal + discussion.
- Spec 005 (TurboQuant wiring) — pieces of which this spec supersedes; PR 1 of spec 005 also migrates GPT-OSS to `attentionWithCacheUpdate`.
- Spec 020 — depends on this spec's PR 1 surface for `MambaCache: TapeReplayCache` extension + iterator wiring.
- Spec 017 — depends on this spec's PR 1 surface for per-class `serialise()` / `hydrate(from:)` extensions.
