# 006 — KVCache architecture refactor

**Status:** spec, ready to issue (architectural; lands before specs 020 + 017 phase 1B as the clean surface for cross-cutting concerns)
**Branch:** new branches per PR — see Migration strategy
**Tracked in:** [GitHub issue #73](https://github.com/ekryski/mlx-swift-lm/issues/73)
**Owner:** Eric

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
| `SSMStateCache` | `MambaCache` (renamed in PR 1; typealias kept) | SSM state | N/A |
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

## Migration strategy (3 PRs)

### PR 1 — Introduce types (low risk, additive only) — **PRECONDITION FOR SPECS 020 + 017 PHASE 1B**

- Add `KVStorage`, `KVEviction`, `KVCache.CompressionAlgorithm`, `KVStorageKind` enums.
- Add `StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`, `SSMStateCache` classes; `makeKVCache(...)` factory.
- Keep all existing classes; `typealias KVCacheSimple = StandardKVCache`, `typealias RotatingKVCache = StandardKVCache`, `typealias QuantizedKVCache = AffineQuantizedKVCache`, `typealias TurboQuantKVCache = TurboQuantizedKVCache`, `typealias MambaCache = SSMStateCache`.
- Add `storageKind` with default implementations on every cache.
- `AttentionUtils.attentionWithCacheUpdate` keeps current dispatch (downcasting); `storageKind`-based dispatch is PR 2.
- `maybeQuantizeKVCache` stays but gains a soft deprecation warning.
- **Audit `ChunkedKVCache`** — trace every call site + check whether any third-party checkpoint depends on it. If unused, mark for PR 2 deletion. If load-bearing, keep as a separate class for now (PR 2 folds into `StandardKVCache` as a `.chunked` eviction variant only if usage warrants it).
- **Risk:** low. Nothing changes behaviourally; new code paths fire only for callers that opt in.
- **Test guarantee:** every consolidated class is byte-identical to the legacy class over a write workload (`testStandardKVCacheUnboundedMatchesKVCacheSimple` etc.).

### PR 2 — Self-transition + migrate `newCache` call sites (medium risk)

- Move `KVCacheSimple → AffineQuantizedKVCache` transition into `AffineQuantizedKVCache` itself.
- Same for Turbo.
- Update every model's `newCache(parameters:)` to the `makeKVCache(scheme:eviction:)` pattern.
- Delete `maybeQuantizeKVCache` and call sites.
- Switch `AttentionUtils` dispatch to `storageKind`.
- Retire `RotatingKVCache` as a class; keep `typealias RotatingKVCache = StandardKVCache` for one release.
- **Risk:** medium. Touches every model. Caught by the acceptance test matrix below.

### PR 3 — Cleanup (low risk)

- Drop deprecated typealiases and `QuantizedKVCacheProtocol` (replaced by `storageKind`).
- Drop `ChunkedKVCache` if confirmed unused, or fold in.
- Remove `kvBits` from `GenerateParameters`; `kvScheme` is now the typed source of truth.
- Update `skills/` and docs.

## Acceptance criteria

1. `scripts/benchmark.sh` with every `--kv` value runs end-to-end on `qwen35-{0.8,2,4,9,27}b`, `qwen35-35b-a3b`, `gemma4-{e2b,e4b,31b,26b-a4b}`, `gpt-oss-20b`, `nemotron-30b-a3b` (and `nemotron-cascade-2-30b-a3b-4bit` once spec 020 lands).
2. **`kv=turbo4v2` reduces `KV Cache` bytes ≥ 3× vs `kv=none`** at ctx=1024 on every windowed model.
3. **`kv=affine4` works on sliding-window layers** (currently a no-op there).
4. Decode tok/s at `kv=none` **unchanged within ±2%** on all benchmarked models. This refactor is correctness, not perf.
5. KL divergence on Qwen3.5-4B at `kv=turbo4v2` and `kv=affine4` within the same band as on `ek/tom-eric-moe-tuning`.
6. `maybeQuantizeKVCache` deleted; no internal call sites remain (PR 2).
7. `newCache(parameters:)` implementations lose their `if/else` cache-type ladders (PR 2).
8. Serialization round-trip preserved — existing `.safetensors` checkpoints with old class names continue to load (back-compat loader at the `metaState`-keyed lookup table at [KVCache.swift:1482-1483](../Libraries/MLXLMCommon/KVCache.swift)).

## Files touched (PR 1)

| File | What |
|---|---|
| `Libraries/MLXLMCommon/KVCacheTypes.swift` (new) | `KVStorage`, `KVEviction`, `KVCache.CompressionAlgorithm`, `KVStorageKind` enums + `makeKVCache(scheme:eviction:)` factory. |
| `Libraries/MLXLMCommon/KVCache.swift` | `StandardKVCache` (consolidates `KVCacheSimple` + `RotatingKVCache`); `AffineQuantizedKVCache` (renames + extends `QuantizedKVCache`); `SSMStateCache` (renames `MambaCache`); typealiases for back-compat; `storageKind` defaults + overrides. Port `RotatingKVCache.reserve(_:)` to `StandardKVCache` (only meaningful when `eviction == .window`). |
| `Libraries/MLXLMCommon/TurboQuantKVCache.swift` | Class rename `TurboQuantKVCache` → `TurboQuantizedKVCache`; `precondition(eviction == .unbounded)` guard at construction; typealias for back-compat. ~30 reference touchpoints across the tree (per project memory). |
| `Libraries/MLXLLM/Models/Qwen3Next.swift` + others using `MambaCache` | Update to `SSMStateCache` (compiler-driven via typealias deprecation warnings). |
| `Tests/MLXLMTests/KVCacheTests.swift` | 9 new tests: `testStandardKVCacheUnboundedMatchesKVCacheSimple`, `testStandardKVCacheWindowMatchesRotatingKVCache`, `testAffineQuantizedKVCacheMatchesQuantizedKVCache`, `testTurboQuantizedKVCacheRenameRoundTrip`, `testSSMStateCacheRenameRoundTrip`, `testMakeKVCacheFactoryAllSchemes`, `testMakeKVCacheTurboWithWindowPreconditionFails`, `testKVStorageKindDispatchOnEveryCacheType`, `testCompressionAlgorithmStringParseRoundTrip`. |

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

- `maybeQuantizeKVCache` (external swap function) — PR 2.
- `QuantizedKVCacheProtocol` as a public type (use `storageKind`) — PR 3.
- `RotatingKVCache` as a distinct class — PR 2 (typealias kept for one release).
- `ChunkedKVCache` — PR 1 audits scope; PR 2 deletes (if unused) or folds in (if load-bearing).
- `kvBits` parameter on `GenerateParameters` — folded into typed `KVCache.CompressionAlgorithm` — PR 3.

## What we rename (in PR 1; one-release typealias deprecation, removed in PR 3)

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
