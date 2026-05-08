# KV Cache System

Reference for the post-spec-006 KV cache surface (v4.x). For the older
`KVCacheSimple` / `RotatingKVCache` / `QuantizedKVCache` / `MambaCache`
names + the `maybeQuantizeKVCache(...)` helper, see
[`documentation/migrations/v3-to-v4.md`](../../../documentation/migrations/v3-to-v4.md).

## Quick reference

| Class | Storage | Eviction | Use case |
|---|---|---|---|
| `StandardKVCache` | raw fp16 / bf16 | `.unbounded` or `.window(size:keep:)` | Default. Legacy `KVCacheSimple` (unbounded) and `RotatingKVCache` (windowed) collapsed into this single class. |
| `AffineQuantizedKVCache` | 4 / 6 / 8-bit affine group-quant | unbounded only | Memory-efficient. Self-transitions from raw → quantized at `startOffset` so prefill stays fast. Windowed-eviction requests fall back to raw `StandardKVCache(maxSize:)` per legacy `maybeQuantizeKVCache` swap behaviour. |
| `TurboQuantizedKVCache` | TurboQuant MSE codec, asymmetric K/V bits | unbounded **or** `.window(size:)` via `makeKVCache` (direct construction); `unbounded` only via `makeAttentionCache` (model-factory path, until [#185](https://github.com/ekryski/mlx-swift-lm/issues/185) is fixed) | Best memory ratio. `keyBits`=0 enables raw-key mode. Sliding-window via `rotatingMaxSize` / `rotatingIdx` machinery — verified working on Mistral 3 / Ministral 3 / Gemma 3. Gemma 4 produces incoherent output under windowed turbo (KV-shared + mixed sliding/full-attention layers); the model-factory path falls through to `StandardKVCache(maxSize:)` until that's investigated. `.keep` (attention-sink prefix) is not surfaced through the codec — windowed turbo treats the buffer as a flat rotating window. |
| `ArraysCache` | generic indexed `[MLXArray?]` slots | n/a | Building block for non-K/V caches. |
| `SSMStateCache` | conv state + recurrent state (subclass of `ArraysCache`) | n/a (SSM state is cumulative) | Mamba / GatedDeltaNet / hybrid linear-attention layers. Replaces legacy `MambaCache`. |
| `BatchedKVCache` | raw fp16 / bf16 across N streams | unbounded | Speculative decoding + multi-request servers. |
| `BatchedMambaCache` / `BatchedHybridCache` | batched SSM / hybrid | n/a | Batched analogues of `SSMStateCache` for hybrid models. |
| `PagedKVCache` | page-table-allocated raw K/V | unbounded | Paged attention; experimental. |
| `CacheList` | heterogeneous sub-caches | composite | Hybrid models (e.g. Qwen 3.5 GDN+Attention) that need multiple cache shapes per layer set. |

**Files:**
- `Libraries/MLXLMCommon/KVCache.swift` — protocol, `StandardKVCache`, `AffineQuantizedKVCache`, `ArraysCache`, `SSMStateCache`, `CacheList`, factories, save / load / trim helpers.
- `Libraries/MLXLMCommon/KVCacheTypes.swift` — typed surface (`KVStorage`, `KVEviction`, `KVStorageKind`, `KVCacheCompressionAlgorithm`) + `makeKVCache` / `makeAttentionCache` / `turboBoundarySkipSet`.
- `Libraries/MLXLMCommon/TurboQuantKVCache.swift` — `TurboQuantizedKVCache` and the MSE codec.
- `Libraries/MLXLMCommon/BatchedKVCache.swift` — `BatchedKVCache`.
- `Libraries/MLXLMCommon/BatchedHybridCache.swift` — `BatchedMambaCache`, `BatchedHybridCache`, `BatchedHybridLLM`.
- `Libraries/MLXLMCommon/PagedKVCache.swift` — `PagedKVCache`.

## The typed surface

The cache classes are constructed from two orthogonal axes plus a parsed
user-facing string format.

### Storage axis (`KVStorage`)

```swift
public enum KVStorage: Sendable, Equatable {
    case raw                                                    // fp16/bf16
    case affine(bits: Int, groupSize: Int = 64, startOffset: Int = 0)
    case turbo(keyBits: Int, valueBits: Int, seed: UInt64 = 42)
}
```

### Eviction axis (`KVEviction`)

```swift
public enum KVEviction: Sendable, Equatable {
    case unbounded
    case window(size: Int, keep: Int = 0)   // `keep` = attention-sink prefix
}
```

### Runtime dispatch tag (`KVStorageKind`)

What the cache currently holds. Self-transitioning caches
(`AffineQuantizedKVCache`, `TurboQuantizedKVCache`) report their
*post-transition* state, so attention dispatch doesn't need `as?`
downcasts on concrete types.

```swift
public enum KVStorageKind: Sendable, Equatable {
    case raw
    case affineQuantized(bits: Int, groupSize: Int)
    case turboCompressed(keyBits: Int, valueBits: Int)
    case ssm
    case composite
}

cache.storageKind  // available on every KVCache
```

### User-facing string format (`KVCache.CompressionAlgorithm`)

The `GenerateParameters.compressionAlgorithm` parameter takes a
`KVCache.CompressionAlgorithm` (typealias for the top-level
`KVCacheCompressionAlgorithm`). It also has a string parser used by the
bench harness's `--kv` flag.

```swift
public enum KVCacheCompressionAlgorithm: Sendable, Equatable, CustomStringConvertible {
    case none
    case affine(bits: Int, groupSize: Int = 64)
    case turbo(
        keyBits: Int,
        valueBits: Int,
        skipBoundaryLayerCompression: Bool = true,
        boundaryLayersToSkip: Int = 2
    )
}

// Programmatic
let algo: KVCache.CompressionAlgorithm = .turbo(keyBits: 4, valueBits: 2)

// Or from a string (CLI / scheme):
let algo = KVCache.CompressionAlgorithm("turbo4v2")        // .turbo(4, 2)
let algo = KVCache.CompressionAlgorithm("turbo4")          // .turbo(4, 4)
let algo = KVCache.CompressionAlgorithm("turbo0v4")        // raw-key mode
let algo = KVCache.CompressionAlgorithm("affine4")         // .affine(4, 64)
let algo = KVCache.CompressionAlgorithm("affine4g32")      // .affine(4, 32)
let algo = KVCache.CompressionAlgorithm("affine8g32")      // .affine(8, 32)
let algo = KVCache.CompressionAlgorithm("none")            // .none
```

`description` round-trips: `algo.description == "turbo4v2"` etc.

## Factories

These are the call sites that ~14 model `newCache(parameters:)` factories
use. Don't hand-instantiate cache classes from outside the model
factories unless you're writing a custom cache strategy.

### `makeAttentionCache(parameters:maxSize:keep:)`

The 90% case for `newCache(parameters:)` — picks the right class based on
the parameters' `compressionAlgorithm`.

```swift
public func makeAttentionCache(
    parameters: GenerateParameters?,
    maxSize: Int? = nil,
    keep: Int = 0
) -> KVCache
```

Decision tree:
- `.affine(bits:groupSize:)` → `AffineQuantizedKVCache`. Window eviction is
  ignored (matches the legacy `maybeQuantizeKVCache` swap behaviour).
- `.turbo(...)` → caller's responsibility. Turbo construction needs
  per-model `headDim` for kernel JIT pre-warm + boundary-skip logic, so
  models that opt into turbo construct `TurboQuantizedKVCache` directly.
- `.none` / `nil` → `StandardKVCache(maxSize: maxSize, keep: keep)` if
  `maxSize` is set; else unbounded `StandardKVCache()`.

### `makeKVCache(scheme:eviction:)`

Single-cache factory composing the storage + eviction axes orthogonally.

```swift
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded
) -> any KVCache
```

`.turbo(...)` + `.window(size:)` is supported — the codec's
`rotatingMaxSize` / `rotatingIdx` machinery wraps writes at `maxSize`
once the raw → compressed transition completes, and the SDPA path
honours windowed semantics for the mask. The `.keep` (attention-sink
prefix) parameter on `.window(...)` is not currently surfaced through
the TurboQuant codec; windowed turbo treats the buffer as a flat
rotating window. Use `.affine(bits:)` instead if you need the
attention-sink prefix.

### `turboBoundarySkipSet(attentionLayerIndices:algorithm:)`

For models that opt into `TurboQuant`, returns the set of attention-layer
indices that should stay uncompressed (first N / last N — most
PPL-sensitive).

```swift
public func turboBoundarySkipSet(
    attentionLayerIndices: [Int],
    algorithm: KVCache.CompressionAlgorithm?
) -> Set<Int>
```

Returns an empty set when the algorithm is `nil` / not turbo /
`skipBoundaryLayerCompression == false` / fewer than
`4 * boundaryLayersToSkip` attention layers (the floor exists so small
models like Qwen 3.5 0.8B don't end up with half their layers skipped).
Hybrid models like NemotronH thread Mamba / MLP / MoE layers around the
attention ones, so the caller computes `attentionLayerIndices` from its
own layer-type discovery.

Example pattern from `Qwen35TextModel.newCache`:

```swift
let layerIndices = (0..<args.hiddenLayers).filter {
    !linearLayerSet.contains($0)
}
let skipSet = turboBoundarySkipSet(
    attentionLayerIndices: layerIndices,
    algorithm: parameters?.compressionAlgorithm
)

return (0..<args.hiddenLayers).map { layerIdx in
    if linearLayerSet.contains(layerIdx) {
        return SSMStateCache()
    }
    if skipSet.contains(layerIdx) {
        return makeAttentionCache(parameters: nil, maxSize: maxSize, keep: keep)
    }
    return TurboQuantizedKVCache(keyBits: keyBits, valueBits: valueBits, ...)
}
```

## `GenerateParameters` knobs

```swift
let params = GenerateParameters(
    maxKVSize: 4096,                                         // Window size (StandardKVCache `.window`)
    compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2),
                                                             // Single source of truth for KV compression.
                                                             // Replaces v3's `kvBits` / `kvGroupSize` /
                                                             // `quantizedKVStart`.
    turboBoundarySkip: 2                                     // Codebook boundary skip — lower raises PPL
                                                             // slightly, faster encode. Default 2.
)
```

Bench-side equivalent: `--kv turbo4v2`, `--kv affine4`, `--kv none`.

## The `KVCache` protocol

```swift
public protocol KVCache: Evaluatable {
    var offset: Int { get }
    var maxSize: Int? { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func peek() -> (MLXArray, MLXArray)?         // KV sharing (Gemma 4)
    var state: [MLXArray] { get set }            // serialization
    var metaState: [String] { get set }
    var isTrimmable: Bool { get }
    @discardableResult func trim(_ n: Int) -> Int
    var memoryBytes: Int { get }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode
    func copy() -> any KVCache
    var isDonor: Bool { get set }
    var storageKind: KVStorageKind { get }
}
```

Default behaviours come from the `BaseKVCache` open class. The base
`makeMask(n:windowSize:returnArray:)` returns:
- `.none` for `n == 1` (single-token decode — no mask needed)
- `.array(...)` if `returnArray` or `n > windowSize`
- `.causal` (symbolic) otherwise

`QuantizedKVCacheProtocol` was removed in spec 006 PR 2. The only
quantized cache type today is `AffineQuantizedKVCache`; external dispatch
is via `cache.storageKind == .affineQuantized(...)` or a direct
`as? AffineQuantizedKVCache` downcast when concrete-class methods are
needed (`groupSize`, `bits`, `mode`, `updateQuantized`, `getQuantizedState`).

## Common patterns

### From a model — preferred

```swift
// Models implement newCache(parameters:); it picks the right cache class
// based on the architecture (hybrid? sliding window? standard?) and the
// passed-in compression algorithm.
let cache: [KVCache] = model.newCache(parameters: generateParameters)
```

### Standalone factories

```swift
// Single attention layer, with turbo's behaviour
let cache = makeKVCache(
    scheme: .turbo(keyBits: 4, valueBits: 2),
    eviction: .unbounded
)

// Convenience for `newCache` factories
let cache = makeAttentionCache(
    parameters: parameters,
    maxSize: 4096,           // optional — sets up windowed eviction
    keep: 4                  // attention-sink prefix
)

// Build a list with known layer count
let cache = makePromptCacheWithLayerCount(
    numLayers: 32,
    maxKVSize: 4096          // nil = unbounded
)

// Full per-model build
let cache = makePromptCache(model: model, parameters: params)
```

### Direct construction

```swift
// Standard, unbounded
let standard = StandardKVCache()

// Standard, windowed (legacy "RotatingKVCache" pattern)
let windowed = StandardKVCache(maxSize: 4096, keep: 4, step: 256)
windowed.reserve(promptLen + maxTokens)   // optional workload-size hint

// Affine 4-bit, group size 64
let affine = AffineQuantizedKVCache(groupSize: 64, bits: 4)

// TurboQuant 4-bit K, 2-bit V
let turbo = TurboQuantizedKVCache(keyBits: 4, valueBits: 2)

// Hybrid SSM state (Mamba / GDN)
let ssm = SSMStateCache()

// Batched cache for speculative decoding
let batched = BatchedKVCache(numLayers: 32, batchSize: 4)
```

### Trimming

Removes tokens from the end of the cache. `StandardKVCache.unbounded` and
`AffineQuantizedKVCache` are trimmable; rotating `StandardKVCache` and
TurboQuant are not (they don't preserve a complete tail).

```swift
if canTrimPromptCache(cache) {
    let trimmed = trimPromptCache(cache, numTokens: 10)
}

cache.first?.trim(10)            // direct trim on a single cache
```

### Serialization

```swift
try savePromptCache(
    url: fileURL,
    cache: cache,
    metadata: ["prompt": "My cached prompt"]
)

let (loaded, metadata) = try loadPromptCache(url: fileURL)
```

`.safetensors` format with metadata.

### Mask creation

```swift
// Cache-driven (preferred — encapsulates eviction + window logic)
let mask = cache.makeMask(n: seqLen, windowSize: nil, returnArray: false)

// Public helper used by model decoders
let mask = makeAttentionMask(
    n: n,
    cache: cache,
    windowSize: nil,
    returnArray: false
)

// Returns MLXFast.ScaledDotProductAttentionMaskMode:
//   .none        → no mask needed (single token)
//   .causal      → symbolic causal mask
//   .array(...)  → explicit MLXArray
```

## Hybrid models — `CacheList` and SSM state

Qwen 3.5, Nemotron-H, and other hybrid GDN + Attention models alternate
linear-attention (SSM) layers with standard attention via a `layer_types`
config. Their `newCache(parameters:)` returns one cache per layer where
attention layers get a K/V cache and linear layers get an `SSMStateCache`.

```swift
let cache: [any KVCache] = (0..<args.hiddenLayers).map { layerIdx in
    if layerIsLinear(layerIdx) {
        return SSMStateCache()
    }
    return makeAttentionCache(parameters: parameters, ...)
}
```

`SSMStateCache: ArraysCache` exposes its internal arrays via
`innerState()`, so the standard prefill-sync barrier in VLM `prepare(...)`
covers the SSM tensors without infrastructure changes:

```swift
var cacheArrays: [MLXArray] = []
for c in cache {
    cacheArrays.append(contentsOf: c.innerState())
}
eval(cacheArrays + [output.logits])
```

Composite caches with multiple sub-shapes per layer use `CacheList`:

```swift
let cache = CacheList(kvCache, ssmCache)
let kv  = cache[0] as! StandardKVCache
let ssm = cache[1] as! SSMStateCache

cache.storageKind   // .composite
```

## Quantized attention dispatch

When a quantized cache is in use, attention typically runs against the
quantized representation rather than dequantizing first. The dispatch is
handled by `AttentionUtils.attentionWithCacheUpdate` based on
`cache.storageKind`; you usually don't call it directly. For the rare
case of writing a custom attention kernel:

```swift
if let qCache = cache as? AffineQuantizedKVCache {
    let (qKeys, qValues) = qCache.updateQuantized(keys: keys, values: values)
    let output = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: qKeys,
        quantizedValues: qValues,
        scale: scale,
        mask: .none,
        groupSize: qCache.groupSize,
        bits: qCache.bits
    )
}
```

`TurboQuantizedKVCache` exposes its own decode path
(`compressedAttention` / fused dequant kernel / TurboFlash) gated by
`useCompressedAttention` and the `TURBO_*` env vars listed in the
[`documentation/kv-cache.md` § Environment-variable overrides](../../../documentation/kv-cache.md#environment-variable-overrides).

## Constraints and footguns

- **`.turbo(...)` + `.window(size:keep:)` ignores `keep`.** The codec's
  rotating buffer treats the window as flat — the attention-sink prefix
  parameter is not currently surfaced. Use `.affine(bits:)` on a
  separate `StandardKVCache` if you need the sink prefix.
- **`AffineQuantizedKVCache` ignores windowed eviction** even when passed
  through `makeAttentionCache(maxSize:)` — matches legacy
  `maybeQuantizeKVCache` behaviour. If you need both, manage context
  length manually.
- **TurboQuantizedKVCache values are not directly trimmable** —
  `isTrimmable` is `false`. The decode-side compressed store doesn't
  preserve a clean tail.
- **KV-sharing donors must not be quantized** — `isDonor` flagged caches
  return raw fp16 / bf16 K / V to shared layers. Self-transitioning
  caches respect `isDonor` and stay raw. If you set `isDonor` on a
  pre-constructed quantized cache, behaviour is undefined.
- **Self-transition timing** — `AffineQuantizedKVCache` stays raw until
  `startOffset` (default `0` for spec-006 callers), then transitions in
  place. Inspect `cache.storageKind` to see the *current* state.

## Removed in spec 006

| v3 name | v4 replacement |
|---|---|
| `KVCacheSimple` | `StandardKVCache()` |
| `RotatingKVCache(maxSize:keep:step:)` | `StandardKVCache(maxSize: keep: step:)` (same shape, single class) |
| `QuantizedKVCache` | `AffineQuantizedKVCache` |
| `MambaCache` | `SSMStateCache` |
| `QuantizedKVCacheProtocol` | `cache.storageKind == .affineQuantized(...)` or `as? AffineQuantizedKVCache` |
| `maybeQuantizeKVCache(cache:kvBits:kvGroupSize:quantizedKVStart:)` | `compressionAlgorithm` on `GenerateParameters` + `makeAttentionCache(...)` factory at `newCache(parameters:)` time |
| `cache.toQuantized(groupSize:bits:)` | Construct `AffineQuantizedKVCache` directly; or rely on `AffineQuantizedKVCache`'s own `startOffset` self-transition |
| `RotatingKVCache.toQuantized()` (always trapped) | Use unbounded `AffineQuantizedKVCache`; manage context length manually |
| `kvBits` / `kvGroupSize` / `quantizedKVStart` on `GenerateParameters` | `compressionAlgorithm: .affine(bits:groupSize:)` / `.turbo(keyBits:valueBits:)` |

Full upgrade guide: [`documentation/migrations/v3-to-v4.md`](../../../documentation/migrations/v3-to-v4.md).
