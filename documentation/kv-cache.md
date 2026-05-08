# KV Cache

The KV cache holds the per-layer K and V tensors so subsequent decode steps
don't re-compute attention over the entire prefix. `mlx-swift-lm` ships several
implementations under `Libraries/MLXLMCommon/` plus a typed compression-algorithm
parser. This page covers the public API and how to choose between them.

## Concrete classes

All implementations conform to the `KVCache` protocol. The factory
`makeAttentionCache(parameters:maxSize:keep:)` is the preferred way to
instantiate one — it parses `GenerateParameters.compressionAlgorithm` and
returns the right concrete type.

| Class | When to use it |
|---|---|
| `StandardKVCache` | Default. Uncompressed fp16 K / V. Supports unbounded growth or windowed eviction (rotating buffer). |
| `AffineQuantizedKVCache` | Affine 4 / 6 / 8-bit K and V (per-row scale + zero point). Reduces KV memory ~3.5× at 4-bit; modest decode-tok/s tax. |
| `TurboQuantizedKVCache` | Block-wise quantization with separate key / value bit budgets (e.g. 4-bit K, 2-bit V — `turbo4v2`). Best memory ratio of the shipped algorithms. Two attention paths inside (compressed-attention "B" path is the default; raw fp16 working buffer "A" path opt-in). See the **TurboQuant note** in the [README](../README.md#choosing-a-deployment-shape-apple-silicon). |
| `BatchedKVCache` | One cache shared across N concurrent decode streams. Used by speculative decoding and multi-request servers. |
| `SSMStateCache` | Hybrid models (Qwen 3.5, Nemotron-H, Jamba). Stores conv + recurrent state for the SSM / GatedDeltaNet blocks. Inherits `ArraysCache.innerState()` so the same prefill-sync barrier covers it. |

## Choosing a configuration

```swift
let parameters = GenerateParameters(
    temperature: 0.6,
    maxTokens: 1024,
    compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2)   // "turbo4v2"
)
```

The `compressionAlgorithm` field accepts:

- `.none` — `StandardKVCache` (default).
- `.affine(bits: Int, groupSize: Int = 64)` — `AffineQuantizedKVCache`.
- `.turbo(keyBits: Int, valueBits: Int)` — `TurboQuantizedKVCache`. Allowed
  pairs: `(8,8)`, `(6,6)`, `(4,4)`, `(4,3)`, `(4,2)`. The `(keyBits,
  valueBits)` shape captures the "turbo" trade-off — keys are typically more
  compressible than values.

You can also pass a string (`"turbo4v2"`, `"affine4"`, `"none"`) via the
parser:

```swift
let parameters = GenerateParameters(compressionAlgorithm: .init("turbo4v2"))
```

## Compatibility per architecture

Not every model produces coherent output under every algorithm. Models that
ship with `supportsTurboQuantization: false` silently fall back to no-quant
when you request a turbo variant — see `Libraries/MLXLMCommon/WiredMemoryUtils.swift`
for the gate.

| Model | Notes |
|---|---|
| GPT-OSS-20B + `turbo4v2` | Falls back to no-quant. Sinks-bearing attention loses Harmony channel-marker fidelity at value-bits=2. Tracked in [#171](https://github.com/ekryski/mlx-swift-lm/issues/171). |
| All others | Affine + turbo work. See [models.md](models.md) for per-model gaps. |

## Bench-time vs library-time

The KV cache strategy you pick at the library level (`GenerateParameters`)
is the same one the bench harness exposes via `--kv`:

```bash
./scripts/benchmark.sh --model qwen35-9b --kv none,affine4,turbo4v2 --quick
```

Internally the bench harness sets `GenerateParameters.compressionAlgorithm`
from the `--kv` string. If you're tuning a deployment, sweep the library API
and the bench together.

## Wired memory and the cache

The wired-memory estimator computes `weights + kv(maxTokens × batchSize,
compressionAlgorithm) + workspace`. Pick a compression algorithm and a
`maxKVSize` together — the estimator sizes the wired ticket from both. See
[wired-memory.md](wired-memory.md).

## See also

- The `GenerateParameters` field reference in the [README](../README.md#generateparameters-programmatic-api)
  covers every sampling and KV-cache knob.
- `TurboQuantizedKVCache` env-var overrides (A path vs B path, dequant
  kernel selection, sparse-V threshold) live in the [README's _Cache /
  attention path_ section](../README.md#cache--attention-path).
- [Migrating from 3.x](migrations/v3-to-v4.md) walks through the spec-006
  KV-cache rewrite (the `kvScheme: "turbo4v2"` string → typed
  `compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2)` transition).
