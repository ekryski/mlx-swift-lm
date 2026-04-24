# Spec 006 — KVCache refactor

**Status:** draft (architectural; needs sign-off before implementation)
**Supersedes pieces of:** spec 005 (turbo wiring — cleaner once this lands)
**Owner:** TBD
**Goals:** kill `maybeQuantizeKVCache`; make each cache responsible for its own lifecycle; make `kv=turbo*` / `kv=affine*` / sliding-window / sink-token combinations declarative and safe.

## Summary of proposal

Pushback on strict inheritance, agreement on the architectural direction. **Compose along two axes (storage, eviction), not three levels of inheritance.** Expose a single scheme-driven factory for 90% of callers; keep the concrete classes public for edge cases.

One-liner of the new API:

```swift
let cache = makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2),
                        eviction: .window(size: 1024))
```

Rest of this doc argues for it and enumerates the migration.

---

## Honest pushback on the proposed hierarchy

The proposal was:

```
KVCache
└── RotatingKVCache
    └── CompressedKVCache (or alongside)
```

Three issues to address before we commit to that shape:

### 1. Inheritance forces rotation onto compression even when it doesn't make sense

TurboQuant's design is explicitly **two-phase**:
- **Prefill phase:** store raw FP16 K/V (like `KVCacheSimple`)
- **Transition:** on first decode call, compress the whole buffer once in a batch
- **Decode phase:** incremental compressed updates, decompress on demand or compute in compressed domain

A rotating turbo cache has no good definition. Two options:
- **(a)** rotate raw, then compress — but then the first decode after a window eviction re-compresses everything
- **(b)** compress and refuse to rotate — semantically a different cache from what the class name implies

`ek/tom-eric-moe-tuning` and alpha both ducked this by not supporting the combination. Spec 003 called it out and documented "turbo on a rotating layer grows past the window". An inheritance chain `RotatingKVCache → CompressedKVCache` would hard-code answer (a), which **changes the memory profile of turbo in the wrong direction** (you pay rotation's window eviction AND compression overhead).

### 2. Rotation and compression aren't the only axes

Looking at the current tree: `KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, `ChunkedKVCache`, `TurboQuantKVCache`, `ArraysCache`, `MambaCache`, `CacheList`. Of these, only the first four are really "KV storage + eviction" variants. `MambaCache` / `ArraysCache` hold SSM state, not K/V — they need to conform to `KVCache` so the generation loop can iterate them, but their internals are unrelated. `CacheList` is a composite. `ChunkedKVCache` exists for a specific quirk ([comment](Libraries/MLXLMCommon/KVCache.swift:1102)).

A strict inheritance chain puts these on a continuum they don't actually belong on.

### 3. Swift doesn't have mixins, and protocol default implementations are a worse fit than composition here

Swift's closest thing to mixins is **protocol + default implementations** (via extensions). That works for cross-cutting behavior (e.g. "every cache has `memoryBytes` — default implementation sums `state`"). It does **not** work for structural composition — you can't have "rotation behavior" as a mixin that changes how `update()` writes into `keys` / `values` because you'd need to override stored-property access from the mixin, and Swift protocols can't touch stored properties across types.

The two practical options in Swift:
- **Class inheritance** — rigid, one axis at a time, produces class explosions (`RawRotatingCache`, `AffineRotatingCache`, `TurboRotatingCache`, …).
- **Composition** — one class, stored strategy objects for the axes. Cleanest, most testable, no combinatorial class blowup.

I recommend **composition** for storage + eviction, **keep inheritance** only for the abstract-base ↔ concrete-class relationship (`BaseKVCache` already does this — don't fight it).

---

## Proposed architecture

### Axes

Two orthogonal axes, one factory:

1. **Storage** — how K/V tensors are held: raw FP16, affine-quantized, turbo-compressed.
2. **Eviction** — when old tokens get dropped: never, sliding window with optional sink tokens.

Plus one dimension that's **not** in the factory because it's model-shape, not user choice: SSM state (`MambaCache`), composite (`CacheList`). Those stay as they are.

### Types

```swift
/// What the cache stores and (for quantized variants) how it encodes.
public enum KVStorage: Sendable {
    /// Raw FP16/BF16 K/V tensors.
    case raw

    /// Group-quantized K/V via MLX's affine quantization.
    /// `startOffset` — wait until the cache has accumulated this many
    /// tokens before switching from raw to quantized; keeps prefill fast.
    case affine(bits: Int, groupSize: Int = 64, startOffset: Int = 0)

    /// TurboQuant compression (MSE codec). `keyBits = 0` enables raw-key
    /// mode (only values compressed).
    case turbo(keyBits: Int, valueBits: Int, seed: UInt64 = 42)
}

/// How the cache discards old tokens.
public enum KVEviction: Sendable {
    /// Never evict; cache grows with every token.
    case unbounded

    /// Sliding window of `size` tokens. First `keep` tokens are preserved
    /// across rotations (attention-sink pattern; GPT-OSS uses keep = 4).
    case window(size: Int, keep: Int = 0)
}

/// Scheme strings used by benchmarks / CLI / parameters.
/// Hand-mapped in one place to `KVStorage` + `KVEviction` so we never
/// parse "turbo4v2" in more than one file.
public enum KVScheme: Sendable, CustomStringConvertible {
    case none
    case affine(bits: Int, groupSize: Int = 64)
    case turbo(keyBits: Int, valueBits: Int)

    public var description: String { ... }
    public init?(_ string: String) { ... }  // "turbo4v2" → .turbo(4, 2)
}
```

### Factory (the ergonomic 90% case)

```swift
/// Construct a `KVCache` appropriate for one attention layer.
///
/// `scheme` + `eviction` compose orthogonally with the following
/// caveat: `.turbo(...)` + `.window(...)` is **not supported** and
/// throws a precondition at construction time. Rationale: TurboQuant
/// is a two-phase design (raw prefill, compressed decode) with no
/// clean definition of "evict a token from the compressed store".
/// For sliding-window layers that want memory compression, use
/// `.affine(bits:)` — it supports windowed eviction natively.
public func makeKVCache(
    scheme: KVScheme = .none,
    eviction: KVEviction = .unbounded
) -> any KVCache
```

### Concrete classes (advanced path, also used by the factory)

One class per storage strategy; eviction is a stored property on the class where it makes sense:

| Class | Replaces | Storage | Eviction |
|---|---|---|---|
| `StandardKVCache` | `KVCacheSimple`, `RotatingKVCache` (consolidated) | Raw | `.unbounded` \| `.window` |
| `AffineQuantizedKVCache` | `QuantizedKVCache` (extended to support window) | Affine | `.unbounded` \| `.window` |
| `TurboQuantKVCache` | (same name, unchanged storage, API trimmed) | Turbo | `.unbounded` only — precondition enforced |
| `MambaCache` | (unchanged) | SSM state | N/A |
| `CacheList` | (unchanged) | Composite | N/A |
| `ChunkedKVCache` | **removed** — fold its specific behavior into `StandardKVCache` as a `.chunked` eviction variant or a `chunkSize` parameter, once we confirm it's still used (grep suggests only one model relies on it) |

`RotatingKVCache` as a class **goes away**. Its call sites become `StandardKVCache(eviction: .window(size: N, keep: M))`. This is where the proposal's "RotatingKVCache inherits from KVCache" collapses into composition — rotation is a behavior on the raw storage class, not a separate type.

`QuantizedKVCache` becomes `AffineQuantizedKVCache` (renamed for symmetry with `TurboQuantKVCache`). Supports windowed eviction natively, so `kv=affine4` + sliding-window layer works correctly — **closing the other TODO at [KVCache.swift:1803](Libraries/MLXLMCommon/KVCache.swift:1803)**.

### Self-transitioning storage (kills `maybeQuantizeKVCache`)

Current code has `maybeQuantizeKVCache(&cache, kvBits: …)` called from the generation loop once per step. That external swap is the "crutch" — fix by pushing the transition inside the cache:

```swift
public final class AffineQuantizedKVCache: BaseKVCache {
    private var raw: RawStorage         // until offset >= startOffset
    private var quantized: QuantizedStorage?
    private let startOffset: Int

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if let q = quantized {
            return q.update(keys: keys, values: values)
        }
        let result = raw.update(keys: keys, values: values)
        if offset >= startOffset {
            quantized = raw.quantize(...)
            raw.clear()
        }
        return result
    }
}
```

Same for `TurboQuantKVCache` — it already does this internally; just remove the external swap path.

Result: **the generation loop has no cache-type dispatch**. It calls `cache.update(…)`. Every cache knows its own lifecycle. `maybeQuantizeKVCache` is deleted.

### Protocol surface stays small

Keep `KVCache` as-is (it's already the right set: `offset`, `update`, `peek`, `state`, `memoryBytes`, `makeMask`, `copy`, `trim`). The protocol doesn't need new methods.

Drop `QuantizedKVCacheProtocol` as a public type — callers that want to check for quantized dispatch should use `cache.storageKind` (a new read-only property on `KVCache`) rather than type-checking. Concrete examples:

```swift
public extension KVCache {
    /// What this cache holds *right now*. May differ from what it was
    /// created with (self-transitioning caches report their current state).
    var storageKind: KVStorageKind { .raw }  // default override-me
}

public enum KVStorageKind: Sendable {
    case raw
    case affineQuantized(bits: Int, groupSize: Int)
    case turboCompressed(keyBits: Int, valueBits: Int)
    case ssm       // MambaCache / ArraysCache — not KV
    case composite // CacheList
}
```

`AttentionUtils.attentionWithCacheUpdate` switches on `cache.storageKind` instead of downcasting to `QuantizedKVCacheProtocol` / `TurboQuantKVCache`. Cleaner, type-safer, and it keeps model code blind to concrete cache classes.

---

## What we keep

- `BaseKVCache` abstract class — still useful for shared `offset`, default `memoryBytes`, etc.
- `MambaCache` / `ArraysCache` — specialized for SSM state, no need to touch.
- `CacheList` — composite pattern is the right shape for heterogeneous layer types.
- The `KVCache` protocol surface.
- All the TurboQuant compression math — that's hard-won correctness; only its construction path changes.
- All persistence formats (`state`, `metaState`) — ship-compat.

## What we delete

- `maybeQuantizeKVCache` (external swap function).
- `QuantizedKVCacheProtocol` as a public type (use `storageKind` instead).
- `RotatingKVCache` as a distinct class (behavior moves into `StandardKVCache`).
- `ChunkedKVCache` if we confirm only legacy usage (one grep hit).
- The `kvBits` parameter on `GenerateParameters` — folded into `kvScheme` (which becomes typed, not a string).

## What we rename

- `KVCacheSimple` → `StandardKVCache`. Current code already has `public typealias StandardKVCache = KVCacheSimple` at [KVCache.swift:1670](Libraries/MLXLMCommon/KVCache.swift:1670); flip which is the primary name.
- `QuantizedKVCache` → `AffineQuantizedKVCache`. Preserve a deprecated `typealias` for a release.

---

## Model integration

Every `newCache(parameters:)` becomes one pattern:

```swift
public func newCache(parameters: GenerateParameters?) -> [KVCache] {
    let scheme = parameters?.kvScheme ?? .none
    return config.layerTypes.map { layerType in
        let eviction: KVEviction = (layerType == "sliding_attention")
            ? .window(size: config.slidingWindow, keep: 0)
            : .unbounded
        return makeKVCache(scheme: scheme, eviction: eviction)
    }
}
```

GPT-OSS variant (`keep: 4` for attention sinks):

```swift
let eviction: KVEviction = (lt == "full_attention")
    ? parameters?.maxKVSize.map { .window(size: $0, keep: 4) } ?? .unbounded
    : .window(size: slidingWindow, keep: 0)
```

Qwen3.5 hybrid (attention layers get the rolling cache; GDN layers get `MambaCache` directly):

```swift
return model.layers.map { layer in
    if layer.isLinear { return MambaCache() }
    let eviction: KVEviction = parameters?.maxKVSize.map {
        .window(size: $0, keep: 0)
    } ?? .unbounded
    return makeKVCache(scheme: parameters?.kvScheme ?? .none,
                       eviction: eviction)
}
```

Every model's `newCache` collapses to 5–10 lines with no cache-type branching. Today they have per-type `if/else` ladders.

## Generation-loop integration

```swift
// before (pseudo):
maybeQuantizeKVCache(cache: &cache, kvBits: kvBits, ...)

// after: delete the call entirely; the cache handles it.
```

`AttentionUtils.attentionWithCacheUpdate` changes its switch to `cache.storageKind` and keeps the existing dispatch bodies. No model code changes.

---

## Acceptance criteria

1. `scripts/benchmark.sh` with every shipped `--kv` value runs end-to-end and produces coherent output on:
   - `qwen35-0.8b`, `qwen35-2b`, `qwen35-4b`, `qwen35-9b`, `qwen35-27b`, `qwen35-35b-a3b`
   - `gemma4-e2b`, `gemma4-e4b`, `gemma4-31b`, `gemma4-26b-a4b`
   - `gpt-oss-20b`
   - `nemotron-30b-a3b`
2. **`kv=turbo4v2` actually reduces `KV Cache` bytes on every windowed model.** Target: ≥ 3× smaller than `kv=none` at ctx=1024 on GPT-OSS-20B and Gemma 4 26B A4B.
3. **`kv=affine4` works on sliding-window layers** (currently a no-op there). Same target.
4. Decode tok/s at `kv=none` **unchanged within ±2%** on all benchmarked models (this refactor is correctness, not perf).
5. KL divergence on Qwen3.5-4B at `kv=turbo4v2` and `kv=affine4` within the same band as on `ek/tom-eric-moe-tuning` (which exercised both via manual paths).
6. `maybeQuantizeKVCache` deleted from the tree; no internal call sites remain.
7. `newCache(parameters:)` implementations in `Libraries/MLXLLM/Models/` lose their `if/else` cache-type ladders.
8. Serialization round-trip preserved — existing `.safetensors` checkpoints with saved cache state continue to load (see Migration below).

## Measurement plan

```bash
# Functional matrix — every scheme on every model:
for model in qwen35-4b qwen35-35b-a3b gemma4-e2b gemma4-26b-a4b \
             gpt-oss-20b nemotron-30b-a3b; do
  scripts/benchmark.sh --model $model --method summarization \
    --quant 4bit --kv none,affine4,turbo4v2 --context 1024 --kld
done

# No-regression sentinels:
scripts/benchmark.sh --model qwen35-4b,gemma4-e2b --method summarization \
    --quant 4bit --kv none --context 128,1024,4096
```

Report before / after KLD for affine + turbo, before / after KV cache bytes, before / after decode tok/s at `kv=none`.

---

## Migration strategy (staged, low-risk)

This is a big refactor. Land it over **three PRs**, each shippable on its own:

### PR 1 — Introduce the new types without removing old ones

- Add `KVStorage`, `KVEviction`, `KVScheme` enums.
- Add `StandardKVCache`, `AffineQuantizedKVCache`, `makeKVCache` factory.
- Keep all existing classes; `typealias KVCacheSimple = StandardKVCache` for source compat.
- Add `storageKind` with default implementations on every existing cache class.
- `AttentionUtils.attentionWithCacheUpdate` switches on `storageKind` but accepts the old types too.
- `maybeQuantizeKVCache` stays but gains a soft deprecation warning.

**Risk:** low. Nothing changes behaviorally; new code paths only fire for callers that opt in.

### PR 2 — Self-transition + migrate `newCache` call sites

- Move the `KVCacheSimple → AffineQuantizedKVCache` transition into `AffineQuantizedKVCache` itself.
- Same for Turbo.
- Update every model's `newCache(parameters:)` to the `makeKVCache(scheme:eviction:)` pattern.
- Delete `maybeQuantizeKVCache` and its call sites.
- Retire `RotatingKVCache` as a class; keep a `typealias RotatingKVCache = StandardKVCache` for one release and emit deprecation.

**Risk:** medium. Touches every model. Caught by the acceptance test matrix.

### PR 3 — Cleanup

- Drop deprecated typealiases and `QuantizedKVCacheProtocol` (replaced by `storageKind` switches).
- Drop `ChunkedKVCache` if confirmed unused, or fold in.
- Remove `kvBits` from `GenerateParameters`; `kvScheme` is now the typed source of truth.
- Update `skills/` and docs.

**Risk:** low. Pure cleanup.

### Persistence / serialization

Saved cache state uses `metaState: [String]` for identification ([KVCache.swift:1482-1483](Libraries/MLXLMCommon/KVCache.swift:1482) has a lookup table). Add a loader that maps **old names** (`"KVCache"`, `"KVCacheSimple"`, `"RotatingKVCache"`, `"QuantizedKVCache"`) **to the new classes** on read. Write new saves with new names. Preserves compatibility with persisted caches from previous releases.

---

## Risks

| Risk | Mitigation |
|---|---|
| `StandardKVCache` handling both unbounded and windowed paths makes its `update` more branchy than `RotatingKVCache`'s today | Benchmark at `kv=none` ctx=1024 on the sentinel models; require ≤2% regression. If the branching matters, split into two internal paths with a shared public surface, not two classes. |
| Model authors who subclassed `KVCacheSimple` or `RotatingKVCache` for custom behavior get broken | Audit: `grep -r "class.*KVCacheSimple\|class.*RotatingKVCache" Libraries/` currently shows **zero** subclasses in-tree. Out-of-tree subclasses get a deprecation window via typealiases. |
| Self-transitioning caches have subtle bugs (race between update-and-transition) | Already the case for `TurboQuantKVCache` — we're not introducing the pattern, just formalizing it. Fuzz-test `update` at the `startOffset` boundary. |
| `storageKind` returning stale info right after a transition | `storageKind` is a computed property on the transitioned cache — it reflects the current state. If we hold a reference to a transitioned cache, it reports the new kind. |
| `kv=turbo*` users expect sliding behavior and get unbounded | Precondition at `makeKVCache` throws; not silently. Error message points to `kv=affine*` for windowed-quantized. |
| GPT-OSS's inlined attention routing bypasses `attentionWithCacheUpdate` and won't auto-pick up quantized routing | Covered in spec 005 already — GPT-OSS migrates to `attentionWithCacheUpdate`. Do that step as part of PR 1 of this refactor. |
| Combinatorial growth if new storage types appear later (e.g. int8 MSE, asymmetric schemes) | The composition model handles them: add an `.int8(...)` case to `KVStorage`, construct a new concrete class or reuse one. No class hierarchy to reshape. |

---

## Alternatives considered

### Alternative A — protocol composition via default implementations (closer to Ruby/Scala mixins)

```swift
protocol Rotatable: KVCache { var evictionPolicy: KVEviction { get } }
extension Rotatable { /* default trim logic */ }

protocol Quantizable: KVCache { var storage: KVStorage { get } }
extension Quantizable { /* default quantize-on-transition logic */ }
```

**Reject.** Protocol extensions can call requirements but can't access stored properties they don't declare — which means the mixin-style behavior can't store the rotation buffer or quantized tensors. It ends up either (a) requiring every conforming class to declare identical stored properties (boilerplate) or (b) using associated objects via Objective-C runtime (not portable, not Sendable-friendly). We'd be fighting Swift's type system.

### Alternative B — strict inheritance (original proposal, `RotatingKVCache → CompressedKVCache`)

**Reject** for the three reasons in the Pushback section above. Summary: conflates semantics that don't compose cleanly, hard-codes the wrong answer for turbo + window, and blows up the class count when we add new storage schemes.

### Alternative C — single class with all behavior inside and a big switch

**Reject.** Loses the type-driven compile-time checking we want (turbo + window throwing at construction) and makes the single class a god-object. The whole point of the factory + multiple classes is that you can reason about one storage strategy at a time.

### Alternative D — keep `maybeQuantizeKVCache`, just add turbo support

Tempting but short-sighted. It concentrates the cache-lifecycle logic outside the cache, meaning every new scheme needs another `if` branch in that function and every cache type that wants to self-quantize needs to either (a) be special-cased in `maybeQuantize*` or (b) run its own parallel transition logic. We'd be doubling down on the crutch.

---

## Open questions

1. **`ChunkedKVCache`**: what's it for? One grep hit, no tests I can see. Trace its usage before PR 3 deletes it; if it's load-bearing for some third-party checkpoint, preserve as a `StandardKVCache` + chunked-eviction variant.
2. **`MambaCache` naming**: Qwen3.5 calls it a "linear attention" cache in code (`isLinear` → `MambaCache`) even though the model is GatedDeltaNet. Rename to `SSMStateCache` for clarity? Out of scope; file as follow-up.
3. **Speculative decoding (`SpeculativeTokenIterator`)** — has separate main / draft caches. Each is an independent `[KVCache]` so the refactor applies unchanged. Double-check the persistence path works with two different schemes (main + draft could plausibly differ).
4. **Deprecation timeline**: one release with typealiases, then remove? Two releases? Depends on who depends on us downstream.
5. **Should `KVScheme` be a top-level enum or nest under `KVCache`?** Leaning top-level — it's referenced from `GenerateParameters` and CLI layers too. If we nest, it becomes `KVCache.Scheme` and that reads fine; decide at PR 1 review.
