# KV Cache System

Reference for the post-spec-006 KV cache surface (v4.x) with spec
041 (flash quantized SDPA), spec 020 (GDN state-replay), and spec 040
(Mamba state-replay) layered on top. For the older `KVCacheSimple` /
`RotatingKVCache` / `QuantizedKVCache` / `MambaCache` names + the
`maybeQuantizeKVCache(...)` helper, see
[`documentation/migrations/v3-to-v4.md`](../../../documentation/migrations/v3-to-v4.md).
For the full user-facing surface (env vars, behaviour knobs, cross-doc
context) see [`documentation/kv-cache.md`](../../../documentation/kv-cache.md).

## Quick reference

| Class | Storage | Eviction | Use case |
|---|---|---|---|
| `StandardKVCache` | raw fp16 / bf16 | `.unbounded` or `.window(size:keep:)` | Default. Legacy `KVCacheSimple` (unbounded) and `RotatingKVCache` (windowed) collapsed into this single class. |
| `AffineQuantizedKVCache` | 4 / 6 / 8-bit affine group-quant | unbounded **or** rotating window via `init(... maxSize:)` (spec 041 phase 1.2) | Memory-efficient. Self-transitions from raw → quantized at `startOffset` so prefill stays fast. Rotating eviction engaged automatically by `makeAttentionCache(...)` when an architectural `slidingWindow` or user `parameters.maxKVSize` cap is set. Gemma 4 / Gemma 3n / Gemma 4 VLM KV-shared donors retain affine compression end-to-end via the Phase 5 reader path (closes [#202](https://github.com/ekryski/mlx-swift-lm/issues/202)). |
| `TurboQuantizedKVCache` | TurboQuant MSE codec, asymmetric K/V bits | unbounded **or** `.window(size:)` (`rotatingMaxSize` / `rotatingIdx` machinery) | Best memory ratio. `keyBits=0` enables raw-key mode. Sliding-window works via both `makeKVCache` and `makeAttentionCache` ([#185](https://github.com/ekryski/mlx-swift-lm/issues/185) closed 2026-05-12). DC-bias correction (`useBias: true`) recovers the structured offset that K/V from `RMSNorm → Linear(bias=True)` carry — default-on for GPT-OSS-20B under `--kv turbo*`. `.keep` (attention-sink prefix) is not surfaced through the codec; windowed turbo treats the buffer as a flat rotating window. |
| `ArraysCache` | generic indexed `[MLXArray?]` slots | n/a | Building block for non-K/V caches. |
| `SSMStateCache` | conv state + recurrent state (subclass of `ArraysCache`) | n/a (SSM state is cumulative) | Mamba / GatedDeltaNet / hybrid linear-attention layers. Replaces legacy `MambaCache`. Conforms to `StateReplayCache` (spec 020 GDN + spec 040 Mamba 2) for speculative-decode rollback. |
| `BatchedKVCache` | raw fp16 / bf16 across N streams | unbounded | Speculative decoding + multi-request servers. |
| `BatchedMambaCache` / `BatchedHybridCache` | batched SSM / hybrid | n/a | Batched analogues of `SSMStateCache` for hybrid models. |
| `PagedKVCache` | page-table-allocated raw K/V | unbounded | Paged attention; experimental ([#127](https://github.com/ekryski/mlx-swift-lm/issues/127) + [#128](https://github.com/ekryski/mlx-swift-lm/issues/128) + [#129](https://github.com/ekryski/mlx-swift-lm/issues/129)). |
| `CacheList` | heterogeneous sub-caches | composite | Hybrid models (e.g. Qwen 3.5 GDN+Attention) that need multiple cache shapes per layer set. |

**Files:**
- `Libraries/MLXLMCommon/KVCache.swift` — protocol, `StandardKVCache`, `AffineQuantizedKVCache`, `ArraysCache`, `SSMStateCache`, `CacheList`, factories, save / load / trim helpers, `AffineSDPAStrategy` + `quantizedScaledDotProductAttention(...)`.
- `Libraries/MLXLMCommon/KVCacheTypes.swift` — typed surface (`KVStorage`, `KVEviction`, `KVStorageKind`, `KVCacheCompressionAlgorithm`) + `makeKVCache` / `makeAttentionCache` / `turboBoundarySkipSet`.
- `Libraries/MLXLMCommon/TurboQuantKVCache.swift` — `TurboQuantizedKVCache` and the MSE codec (`useBias` lives here).
- `Libraries/MLXLMCommon/StateReplayCache.swift` — `StateReplayCache` protocol + `canRollbackPromptCache(_:)` helper.
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

These are the call sites that the ~14 model `newCache(parameters:)`
factories use. Don't hand-instantiate cache classes from outside the
model factories unless you're writing a custom cache strategy.

### `makeAttentionCache(parameters:slidingWindow:keep:prefillStep:forceRawKV:useBias:)`

The 90% case for `newCache(parameters:)` — picks the right class based on
the parameters' `compressionAlgorithm` and the layer's architectural shape.

```swift
public func makeAttentionCache(
    parameters: GenerateParameters?,
    slidingWindow: Int? = nil,
    keep: Int = 0,
    prefillStep: Int? = nil,
    forceRawKV: Bool = false,
    useBias: Bool = false
) -> KVCache
```

**Two distinct caps drive rotating eviction:**

1. `slidingWindow: Int?` — **architectural** cap (per-layer flag set by the
   model factory when the layer must attend over only the last N tokens —
   Gemma 4 local layers, GPT-OSS sliding layers, Mistral 3 sliding layers,
   etc.). Non-negotiable.
2. `parameters?.maxKVSize` — **user budget** cap (read internally from the
   parameters tuple). Honored uniformly across `.none`, `.affine`, and `.turbo`
   when the layer is **not** architecturally sliding-window.

Precedence when both are present: `slidingWindow` wins.

Decision tree (after resolving `effectiveMaxSize = slidingWindow ?? parameters?.maxKVSize`):
- `.affine(bits:groupSize:)` + cap set → `AffineQuantizedKVCache(maxSize: cap, ...)` (spec 041 phase 1.2 rotating-window affine).
- `.affine(bits:groupSize:)` + no cap → `AffineQuantizedKVCache(...)` (unbounded).
- `.affine(...)` + `forceRawKV: true` → `StandardKVCache(...)` (KV-sharing donor whose reader can't consume the quantised tuple — Gemma 3n + Gemma 4 VLM readers; Gemma 4 LLM Phase 5 reader handles the quantised tuple directly so it doesn't set `forceRawKV`).
- `.turbo(...)` + cap set → `TurboQuantizedKVCache(maxSize: cap, useBias:)`.
- `.turbo(...)` + no cap → caller's responsibility (turbo construction needs per-model `headDim` for kernel JIT pre-warm + boundary-skip logic; models that opt into turbo construct `TurboQuantizedKVCache` directly).
- `.none` / `nil` + cap → `StandardKVCache(maxSize: cap, keep: keep, step: resolvedStep)`.
- `.none` / `nil` + no cap → `StandardKVCache(step: resolvedStep)` (unbounded).

`prefillStep` resolves the per-cache growth-chunk size for `concatenated(...)`:
`parameters?.prefillStepSize ?? prefillStep ?? 256`. The factory threads
this uniformly into all three cache classes' `step:` constructor parameter
so growth events match the model's prefill chunk 1:1.

Diagnostic env knob: `MLX_TURBO_WINDOWED=0` forces the legacy `StandardKVCache`
fallback for the windowed-turbo case. Useful for A/B testing; not needed in
normal operation.

### `makeKVCache(scheme:eviction:prefillStep:)`

Single-cache factory composing the storage + eviction axes orthogonally.
Used by infrastructure / tests / custom paths that don't go through
`newCache(parameters:)`.

```swift
public func makeKVCache(
    scheme: KVCache.CompressionAlgorithm = .none,
    eviction: KVEviction = .unbounded,
    prefillStep: Int? = nil
) -> any KVCache
```

`.turbo(...)` + `.window(size:)` is supported — the codec's `rotatingMaxSize` /
`rotatingIdx` machinery wraps writes at `maxSize` once the raw → compressed
transition completes, and the SDPA path honours windowed semantics for the
mask. The `.keep` (attention-sink prefix) parameter on `.window(...)` is not
currently surfaced through the TurboQuant codec; windowed turbo treats the
buffer as a flat rotating window.

`.affine(...)` + `.window(size:)` falls back to non-rotating affine (this
factory predates spec 041 phase 1.2). Use `makeAttentionCache(...)` (the
model-factory entry point) if you need rotating affine — it reads
`parameters.maxKVSize` and constructs `AffineQuantizedKVCache(maxSize:)`
directly.

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
        return makeAttentionCache(parameters: nil)
    }
    return TurboQuantizedKVCache(keyBits: keyBits, valueBits: valueBits, ...)
}
```

## `GenerateParameters` knobs

```swift
let params = GenerateParameters(
    maxKVSize: 4096,                                         // User-budget cap.
                                                             // Engages rotating
                                                             // eviction uniformly
                                                             // across `.none` /
                                                             // `.affine` / `.turbo`.
    compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2),  // KV compression.
                                                             // Single source of truth
                                                             // (replaces v3's `kvBits` /
                                                             // `kvGroupSize` /
                                                             // `quantizedKVStart`).
    turboBoundarySkip: 2,                                    // Codebook boundary skip
                                                             // — lower raises PPL
                                                             // slightly, faster encode.
                                                             // Default 2.
    prefillStepSize: 1024                                    // Per-cache growth-chunk
                                                             // size. Defaults to the
                                                             // model's
                                                             // `defaultPrefillStepSize`.
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
// based on the architecture (hybrid? sliding window? KV-shared?) and the
// passed-in compression algorithm.
let cache: [KVCache] = model.newCache(parameters: generateParameters)
```

### Standalone factories

```swift
// Single attention layer via the typed-axes factory
let cache = makeKVCache(
    scheme: .turbo(keyBits: 4, valueBits: 2),
    eviction: .unbounded
)

// Model-factory entry point — reads parameters.maxKVSize + applies precedence
let cache = makeAttentionCache(
    parameters: parameters,
    slidingWindow: 4096,    // architectural cap (per-layer)
    keep: 4,                // attention-sink prefix (StandardKVCache only)
    useBias: false          // DC-bias correction (TurboQuant only)
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

// Affine 4-bit, group size 64 (unbounded)
let affine = AffineQuantizedKVCache(groupSize: 64, bits: 4)

// Affine 4-bit, rotating window at 4096 (spec 041 phase 1.2)
let affineWindowed = AffineQuantizedKVCache(
    groupSize: 64, bits: 4, step: 256, maxSize: 4096
)

// Affine with explicit SDPA strategy (spec 041 phases 1+1.1)
let affineKernel = AffineQuantizedKVCache(
    groupSize: 64, bits: 4, sdpaStrategy: .kernel
)

// TurboQuant 4-bit K, 2-bit V
let turbo = TurboQuantizedKVCache(keyBits: 4, valueBits: 2)

// TurboQuant with DC-bias correction (GPT-OSS-20B default)
let turboBias = TurboQuantizedKVCache(
    keyBits: 4, valueBits: 2, useBias: true
)

// Hybrid SSM state (Mamba / GDN)
let ssm = SSMStateCache()

// Batched cache for speculative decoding
let batched = BatchedKVCache(numLayers: 32, batchSize: 4)
```

### Trimming

Removes tokens from the end of the cache. Trimmability:

- `StandardKVCache` — trimmable in `.unbounded` mode; in `.window(maxSize)`
  mode, trimmable only while `offset < maxSize` (i.e., before the rotating
  buffer has wrapped).
- `AffineQuantizedKVCache` — trimmable in both unbounded and rotating
  variants (the spec 041 phase 1.2 rotating-window cache).
- `TurboQuantizedKVCache` — trimmable; same wrap-time caveat as windowed
  `StandardKVCache` for the rotating variant. Trim adjusts the absolute
  `offset` but does not re-slice the packed K/V buffer.

```swift
if canTrimPromptCache(cache) {
    let trimmed = trimPromptCache(cache, numTokens: 10)
}

cache.first?.trim(10)            // direct trim on a single cache
```

### State-replay rollback (hybrid models, speculative decoding)

`SSMStateCache` conforms to `StateReplayCache` for partial-accept rollback
during speculative decoding. Both **GatedDeltaNet** (Qwen 3.5 / 3.6 — spec
020) and **Mamba 2** (NemotronH / GraniteMoeHybrid / FalconH1 — spec 040)
are wired end-to-end. Jamba's cache conforms but its mamba-mixer recording
branch is a pending follow-up.

```swift
if canRollbackPromptCache(cache) {
    cache.first?.beginRecord()
    // run verify forward — kernels (`gated_delta_step_record` / `ssm_step_record`)
    // capture per-step deltas onto the cache's delta log
    cache.first?.rollback(acceptedPrefix: k)    // re-fold first k deltas
    cache.first?.commitFull()                    // accept everything
}
```

See `Libraries/MLXLMCommon/StateReplayCache.swift` and
[`documentation/speculative-decoding.md`](../../../documentation/speculative-decoding.md#hybrid-model-coverage-spec-020).

### Cross-request prefix cache (spec 017)

Opt in via `GenerateParameters.prefixCacheEnabled = true` (or env
`MLX_PREFIX_CACHE=1`). The runtime snapshots the cache at request end
keyed on a stable token prefix and hydrates at the next request's start.
Per-class `serialise()` / `hydrate(from:)` covers `StandardKVCache`
(unbounded + windowed, refuses wrapped buffers), `AffineQuantizedKVCache`,
`TurboQuantizedKVCache` (raw mode only — compressed-mode round-trip is
[#197](https://github.com/ekryski/mlx-swift-lm/issues/197) /
[spec 039](../../../specs/039-compressed-prefix-kv-cache.md)),
`SSMStateCache` (via spec 020 / 040 state-replay path).

Default policy auto-resolves per model family via
`AssistantOpener.detect(forModelID:)` — Qwen / Gemma / GPT-OSS families
get matching openers; unknown families fall back to `IdentityPolicy`.
Phase 4 disk persistence at `~/.cache/mlx-swift-lm/prefix/` is opt-in
(`prefixCacheDiskEnabled` / `MLX_PREFIX_CACHE_DISK=1`).

See
[`documentation/generate-parameters.md`](../../../documentation/generate-parameters.md#cross-request-prefix-kv-cache-spec-017).

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

Qwen 3.5, Nemotron-H, Granite-MoE-Hybrid, FalconH1, Jamba alternate
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
        bits: qCache.bits,
        strategy: qCache.sdpaStrategy
    )
}
```

### `AffineQuantizedKVCache.sdpaStrategy` (spec 041 phases 1+1.1+2+4)

Picks the SDPA path for affine caches.

```swift
public enum AffineSDPAStrategy: Sendable {
    case auto       // flash for L>1 (prefill), discrete for L=1 (decode)
    case kernel     // fused Metal kernel (spec 041 phase 1.1) — correct, not yet matrix-engine-optimised
    case flash      // dequant-then-MLXFastSDPA stop-gap — wins on small-model long-context shapes
    case discrete   // legacy quantizedMM → softmax → quantizedMM (regression check)
}

let cache = AffineQuantizedKVCache(
    groupSize: 64, bits: 4, sdpaStrategy: .flash
)
```

Precedence: env var `MLX_AFFINE_SDPA={auto,kernel,flash,discrete}` when
set ▶ per-cache `sdpaStrategy` ▶ `.auto`. Empirical: auto-strategy saves
-42% peak GPU on GPT-OSS-20B 8k prefill and -60% on Gemma 4 31B 8k prefill
vs discrete-only, with no decode tok/s regression. Kernel-path perf parity
is tracked under [spec 042](../../../specs/042-metal-kernel-simd-audit.md)
Phase 1b.

### TurboQuant decode paths

`TurboQuantizedKVCache` exposes its own decode path (`compressedAttention` /
fused dequant kernel / TurboFlash) gated by `useCompressedAttention` and
the `TURBO_*` env vars listed in
[`documentation/kv-cache.md` § Environment-variable overrides](../../../documentation/kv-cache.md#environment-variable-overrides).
See also
[`documentation/turbo-kernels.md`](../../../documentation/turbo-kernels.md)
for the per-kernel reference + A/B path map.

| Path | Default | Notes |
|---|---|---|
| **A path — TurboFlash** | yes | Compressed-domain Metal kernel; scores directly against packed K/V; no FP16 working buffer. Unconditional A path on TurboQuant caches as of `85afa9b`. |
| **A path — sinks variant** (`turbo_flash_sdpa_v`) | sinks-using models | Single-pass kernel folds attention sinks inline (GPT-OSS-20B). |
| **B path — dequant-then-SDPA** | opt-in via `TURBO_DEQUANT_SDPA=1` | `bulkDequantRotated` → FP16 K/V → `MLXFast.scaledDotProductAttention`. Trades a per-layer FP16 working buffer for matmul-engine throughput. Forced on when `useBias: true` (TurboFlash kernels don't yet consume the stored bias term). |

Performance uplift work for the A path is tracked under
[spec 043](../../../specs/043-turboflash-kernel-uplift.md) (TurboFlash
decode-time kernel uplift — bit-unpack reuse, bf16 V accumulator,
headDim-aware tile autotune, bias-aware kernel).

### DC-bias correction (`useBias`)

The MSE codec is a zero-mean Lloyd-Max quantiser; per-vector DC offsets
that K/V from `RMSNorm → Linear(bias=True)` carry land in the codec's
error term. `useBias: true` subtracts the per-vector mean before WHT
encode and adds it back at decode, recovering 9-22% MSE on this
distribution.

Storage: one fp32 per token-head alongside `norms` (negligible vs packed
indices). When enabled, the cache forces the B path because TurboFlash
kernels don't yet consume the stored bias term — fusable into the kernel
as [spec 043](../../../specs/043-turboflash-kernel-uplift.md) Phase 4.

**Default off** for general use; **default on** for GPT-OSS-20B under
`--kv turbo*` (see `Libraries/MLXLLM/Models/GPTOSS.swift`'s
`newCache(...)`). Env override: `TURBO_BIAS=1` / `=0`.

## Constraints and footguns

- **`.turbo(...)` + `.window(size:keep:)` ignores `keep`.** The codec's
  rotating buffer treats the window as flat — the attention-sink prefix
  parameter is not currently surfaced. Use `.affine(bits:)` or `.none`
  on a separate `StandardKVCache` if you need the sink prefix.
- **`makeKVCache(scheme:.affine, eviction:.window(...))` falls back to
  unbounded affine.** That factory predates spec 041 phase 1.2's
  rotating-window affine cache. Go through `makeAttentionCache(...)` (the
  model-factory path) if you need rotating affine — it reads
  `parameters.maxKVSize` and constructs `AffineQuantizedKVCache(maxSize:)`
  directly.
- **TurboQuantizedKVCache values are not directly trimmable** —
  `isTrimmable` is `false` in unbounded mode. The decode-side compressed
  store doesn't preserve a clean tail. Rotating-window turbo trims by
  adjusting `offset` only; the packed K/V buffer is not re-sliced.
- **KV-sharing donors and the affine reader path** — Gemma 4 LLM /
  Gemma 4 VLM / Gemma 3n KV-shared donor layers retain affine compression
  end-to-end via the Phase 5 reader (`Gemma4TextAttention.callAsFunction`
  routes the donor's `(wq, scales, biases)` tuple through
  `quantizedScaledDotProductAttention`). Older readers (Gemma 3n /
  Gemma 4 VLM raw path) still set `forceRawKV: true` and fall back to
  `StandardKVCache` — closes [#202](https://github.com/ekryski/mlx-swift-lm/issues/202)
  for the LLM path; readers that still need raw K/V continue working via
  the `forceRawKV` flag.
- **Self-transition timing** — `AffineQuantizedKVCache` stays raw until
  `startOffset` (default `0` for spec-006 callers), then transitions in
  place. Inspect `cache.storageKind` to see the *current* state.
- **`MLX_AFFINE_SDPA` env var wins over per-cache `sdpaStrategy`** when
  set. Useful for whole-process A/B tests, easy to forget in benchmark
  scripts.

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
| `RotatingKVCache.toQuantized()` (always trapped) | Use `AffineQuantizedKVCache(maxSize:)` (rotating-window affine, spec 041 phase 1.2) |
| `kvBits` / `kvGroupSize` / `quantizedKVStart` on `GenerateParameters` | `compressionAlgorithm: .affine(bits:groupSize:)` / `.turbo(keyBits:valueBits:)` |

Full upgrade guide: [`documentation/migrations/v3-to-v4.md`](../../../documentation/migrations/v3-to-v4.md).
