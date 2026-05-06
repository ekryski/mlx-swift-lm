# Migrating from v3 to v4

How to update an existing v3.x integration to v4.x.

## Overview

Version 4 of MLX Swift LM rewrites the KV-cache architecture under spec 006. The library-level concepts didn't change — every model still produces logits, the iterator still yields tokens, persistence still round-trips through `safetensors` — but the *types* you reach for and the way you configure compression both look different.

The headline changes:

- KV-cache classes consolidated along two orthogonal axes (storage × eviction).
- Class renames: `KVCacheSimple` / `RotatingKVCache` / `QuantizedKVCache` / `TurboQuantKVCache` / `MambaCache` are gone.
- A typed `KVCache.CompressionAlgorithm` replaces the four legacy `GenerateParameters` fields (`kvBits`, `kvGroupSize`, `quantizedKVStart`, `kvScheme`).
- `maybeQuantizeKVCache(...)` deleted — caches now construct the right class up-front in each model's `newCache(parameters:)`.
- `ChunkedKVCache` deleted (audit confirmed zero in-tree usage).
- `QuantizedKVCacheProtocol` dropped — `as? AffineQuantizedKVCache` directly is the public-API replacement.

The persistence on-disk format is unchanged — `savePromptCache` still writes the legacy class-name strings (`"KVCache"`, `"RotatingKVCache"`, `"QuantizedKVCache"`, `"MambaCache"`) so checkpoints saved by mlx-lm Python and earlier mlx-swift-lm versions continue to load.

## At a glance — type renames

| v3 | v4 |
|---|---|
| `KVCacheSimple` | `StandardKVCache` (default `eviction: .unbounded`) |
| `RotatingKVCache(maxSize: N)` | `StandardKVCache(maxSize: N)` (windowed eviction; convenience init still exists) |
| `QuantizedKVCache` | `AffineQuantizedKVCache` |
| `TurboQuantKVCache` | `TurboQuantizedKVCache` |
| `MambaCache` | `SSMStateCache` |
| `ChunkedKVCache` | (removed — was unused) |
| `QuantizedKVCacheProtocol` | (removed — use `as? AffineQuantizedKVCache`) |

The legacy names were available as `typealias` shims during the spec 006 PR 1 transition and have been deleted in PR 2. Anything that used them will fail to compile against v4 until renamed.

## Cache architecture: two axes + a factory

v3 had a flat type hierarchy; v4 makes the two real axes explicit:

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
```

Plus a `storageKind: KVStorageKind { get }` requirement on the `KVCache` protocol so dispatch code can switch on the kind without downcasting:

```swift
extension KVCache {
    public var storageKind: KVStorageKind { .raw }   // default
}
```

A factory ties them together:

```swift
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded
) -> any KVCache
```

`makeKVCache(scheme: .turbo(...), eviction: .window(...))` traps with a precondition. TurboQuant is a two-phase design and there's no clean definition of "evict from the compressed store"; for sliding-window-quantized caches, use `.affine`.

For models that don't need full storage/eviction control (the common case), v4 ships a thinner helper used by every model factory:

```swift
public func makeAttentionCache(
    parameters: GenerateParameters?,
    maxSize: Int? = nil,
    keep: Int = 0
) -> KVCache
```

It returns an `AffineQuantizedKVCache` when `parameters?.compressionAlgorithm == .affine`, and a `StandardKVCache` otherwise. This is what every model's `newCache(parameters:)` calls — see "model factories" below.

## `GenerateParameters` cleanup

v3 carried four legacy fields for KV compression:

```swift
// v3
public var kvBits: Int?              // nil = no quantization
public var kvGroupSize: Int          // 64
public var quantizedKVStart: Int     // 0
public var kvScheme: String?         // "turbo4v2", "affine4", etc.
```

v4 collapses all of them into one typed enum:

```swift
// v4
public var compressionAlgorithm: KVCache.CompressionAlgorithm?

// New in v4: per-cache compression for speculative decoding (#PR 4).
// nil ⇒ draft model uses `compressionAlgorithm` (matches main).
public var draftCompressionAlgorithm: KVCache.CompressionAlgorithm?
```

```swift
extension KVCache {
    public enum CompressionAlgorithm: Sendable, CustomStringConvertible {
        case none
        case affine(bits: Int, groupSize: Int = 64)
        case turbo(keyBits: Int, valueBits: Int)

        public var description: String { ... }
        public init?(_ string: String) { ... }   // "turbo4v2" → .turbo(4, 2)
    }
}
```

The string parser is the single source of truth for CLI strings — code that used to parse `"turbo4v2"` etc. by hand should now go through `KVCache.CompressionAlgorithm.init?(_:)`.

## `maybeQuantizeKVCache` is gone

In v3 the generation loop called `maybeQuantizeKVCache(&cache, kvBits: ...)` once per step to swap a raw cache for a quantized one. v4 removes that helper entirely. Every model's `newCache(parameters:)` now constructs the right cache class up-front:

```swift
// v4 — typical model factory pattern
public func newCache(parameters: GenerateParameters?) -> [KVCache] {
    (0 ..< numLayers).map { _ in
        makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
    }
}
```

If you wrote your own model factory that called `maybeQuantizeKVCache` after the iterator started, that code needs to:

1. Delete the `maybeQuantizeKVCache` call.
2. Make sure your `newCache(parameters:)` returns the quantized cache class up-front when `parameters?.compressionAlgorithm == .affine`. The `makeAttentionCache(...)` helper does this for the common case; if your model has special cache requirements (e.g. layer-typed dispatch in hybrid models like Qwen3-Next or NemotronH), construct the cache type explicitly:

   ```swift
   if case .affine(let bits, let groupSize) = parameters?.compressionAlgorithm {
       return AffineQuantizedKVCache(groupSize: groupSize, bits: bits)
   }
   return StandardKVCache(maxSize: parameters?.maxKVSize, keep: keep)
   ```

3. If you opted into TurboQuant and need boundary-skip behaviour: TurboQuant cache construction lives in the model factory itself (see `Qwen35.newCache(parameters:)` and `NemotronH.newCache(parameters:)` in the source for the pattern). The v3 boundary-skip-after-the-fact path is gone.

## Loading cached prompts is unchanged

`savePromptCache` and `loadPromptCache` still work with both v3-saved and v4-saved checkpoints. The on-disk class-name strings stay legacy:

| Disk string | v4 class produced by loader |
|---|---|
| `"KVCache"` / `"KVCacheSimple"` | `StandardKVCache()` (unbounded) |
| `"RotatingKVCache"` | `StandardKVCache(maxSize: …)` (windowed) |
| `"QuantizedKVCache"` | `AffineQuantizedKVCache()` |
| `"MambaCache"` | `SSMStateCache()` |
| `"ChunkedKVCache"` | (throws — deleted in v4) |

## Migration examples

### Setting KV compression at call site

```swift
// v3
var params = GenerateParameters()
params.kvBits = 4
params.kvGroupSize = 64
params.kvScheme = "affine4"

// v4
var params = GenerateParameters()
params.compressionAlgorithm = .affine(bits: 4, groupSize: 64)
```

### Parsing a CLI flag

```swift
// v3
if let scheme = process.arguments["--kv"] as? String {
    if scheme.hasPrefix("turbo") {
        // hand-parse "turbo4v2" → bits 4, valueBits 2
        params.kvScheme = scheme
    } else if scheme.hasPrefix("affine") {
        params.kvBits = 4 // hand-parse "affine4" → 4
    }
}

// v4 — single source of truth for parsing
if let scheme = process.arguments["--kv"] as? String,
   let algo = KVCache.CompressionAlgorithm(scheme) {
    params.compressionAlgorithm = algo
}
```

### Type-checking a cache

```swift
// v3
if let q = cache as? QuantizedKVCacheProtocol {
    // …use q.bits, q.groupSize…
}

// v4
if let q = cache as? AffineQuantizedKVCache {
    // …use q.bits, q.groupSize…
}

// or — preferred where the kind matters more than the concrete type:
switch cache.storageKind {
case .raw: ...
case .affineQuantized(let bits, let groupSize): ...
case .turboCompressed(let keyBits, let valueBits): ...
case .ssm: ...
case .composite: ...
}
```

### Constructing a sliding-window cache directly

```swift
// v3
let cache = RotatingKVCache(maxSize: 1024)

// v4 — the convenience init produces a windowed `StandardKVCache`
let cache = StandardKVCache(maxSize: 1024)

// or — explicit
let cache = StandardKVCache(eviction: .window(size: 1024))
```

### Custom model factory

```swift
// v3
public func newCache(parameters: GenerateParameters?) -> [KVCache] {
    (0 ..< layerCount).map { _ in
        if let maxSize = parameters?.maxKVSize {
            return RotatingKVCache(maxSize: maxSize, keep: 4)
        } else {
            return KVCacheSimple()
        }
    }
}

// v4
public func newCache(parameters: GenerateParameters?) -> [KVCache] {
    (0 ..< layerCount).map { _ in
        makeAttentionCache(
            parameters: parameters,
            maxSize: parameters?.maxKVSize,
            keep: 4)
    }
}
```

### Speculative decoding with different main / draft compression

New in v4:

```swift
var params = GenerateParameters()
params.compressionAlgorithm = .turbo(keyBits: 4, valueBits: 2)
params.draftCompressionAlgorithm = .affine(bits: 4, groupSize: 64)

let iterator = try SpeculativeTokenIterator(
    input: input,
    mainModel: mainModel,
    draftModel: draftModel,
    parameters: params,
    numDraftTokens: 4
)
```

When `draftCompressionAlgorithm` is `nil`, both caches use `compressionAlgorithm` (v3 behaviour).

## Removed APIs reference

For grep-ability when porting, here's the full list of v3 symbols that no longer exist in v4:

**Types**
- `KVCacheSimple`
- `RotatingKVCache`
- `QuantizedKVCache`
- `TurboQuantKVCache`
- `MambaCache`
- `ChunkedKVCache`
- `QuantizedKVCacheProtocol`

**Functions**
- `maybeQuantizeKVCache(cache:kvBits:kvGroupSize:quantizedKVStart:kvScheme:turboBoundarySkip:)`
- `parseTurboScheme(_:)` is **kept** for the bench harness's legacy callers but new code should use `KVCache.CompressionAlgorithm.init?(_:)`.

**`GenerateParameters` fields**
- `kvBits: Int?`
- `kvGroupSize: Int`
- `quantizedKVStart: Int`
- `kvScheme: String?`

## See also

- <doc:v2-to-v3-migration> for the v2→v3 upgrade (decoupled tokenizer + downloader, new imports, loading API changes).
- <doc:publishing-a-release> for the manual-trigger release pipeline used to ship v4.
