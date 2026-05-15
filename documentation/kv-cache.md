# KV Cache

The KV cache holds the per-layer K and V tensors so subsequent decode steps
don't re-compute attention over the entire prefix. `mlx-swift-lm` ships several
implementations under `Libraries/MLXLMCommon/` plus a typed compression-algorithm
parser. This page covers the public API and how to choose between them.

## What's supported today

| Algorithm | When to use | Memory ratio | Notes |
|---|---|---|---|
| **No compression** (`StandardKVCache`, default) | Baseline. Speed-critical short / mid context. | 1× (raw fp16/bf16) | Supports unbounded growth or sliding-window eviction (`maxKVSize`). |
| **Affine quantization** (`AffineQuantizedKVCache`) | Memory-constrained; attention quality matters. | ~3.5× at 4-bit | 4 / 6 / 8-bit affine group-quant. Self-transitions raw → quantized at `startOffset` so prefill stays fast. |
| **TurboQuant compression** (`TurboQuantizedKVCache`) | Best memory ratio at minimal quality loss. Sliding-window models (Gemma 4, GPT-OSS, Mistral 3, etc.) supported via the codec's built-in rotating buffer. | ~6–8× at `turbo4v2` | Block-wise MSE codec with **asymmetric K/V bits** (e.g. 4-bit K, 2-bit V). Decode runs through one of two attention paths: **A path (default)** — TurboFlash compressed-domain Metal kernel (scores directly against packed K/V); **B path (opt-in via `TURBO_DEQUANT_SDPA=1`)** — `bulkDequantRotated` to FP16 + Apple's `MLXFast.scaledDotProductAttention`. |
| **Hybrid (`SSMStateCache`)** | Mamba / GatedDeltaNet (Qwen 3.5 / Nemotron-H / Jamba) | n/a | Stores conv + recurrent state instead of K/V. Composes with attention layers via `CacheList`. |
| **Batched (`BatchedKVCache` / `BatchedHybridCache`)** | Multi-stream decode (speculative, B>1 serving) | linear in B | Slot-based admission for fixed-size batches. |

For configuration shapes (Qwen 3.5 / Nemotron-H boundary-layer skip, GPT-OSS sinks gate, etc.) see [models.md](models.md).

## What's shipped recently

- **Spec 020 — `SSMStateCache: StateReplayCache`** (shipped 2026-05-11). GatedDeltaNet families (Qwen 3.5 / 3.6 GDN+attention) gained speculative-decode rollback. `SSMStateCache` conforms to `StateReplayCache`, exposing `beginRecord` / `recordStep` / `commitFull` / `rollback(acceptedPrefix:)` / `cancel`. Native Metal kernels (`gated_delta_step_record` / `state_replay`) replace the legacy snapshot/restore pair on the speculative path. See [speculative-decoding.md](speculative-decoding.md#hybrid-model-coverage-spec-020).
- **Spec 040 — Mamba / Mamba 2 state-replay** (shipped 2026-05-14). Extended the spec 020 infrastructure to the Mamba recurrence via new `ssm_step_record` / `ssm_replay` Metal kernels. `NemotronH` (Mamba 2), `GraniteMoeHybrid` (Mamba 2), and `FalconH1` (Mamba 2) are integrated end-to-end — `cache.isRecording` routes through `ssmStepRecord`; partial-accept rollback restores both the recurrent state and the depthwise-conv buffer (per-step `mambaConvPadded` capture). Jamba's cache exposes `canStateReplay = true` but its mamba-mixer forward path does not yet wire the recording branch — a mechanical follow-up tracked in [spec 040 known limitations](../specs/040-mamba-state-replay.md).
- **Spec 017 — Cross-request prefix KV cache** (shipped 2026-05-12 as **opt-in for v1**). Snapshot the cache at request end keyed on a stable token prefix; hydrate at the next request's start. Per-class `serialise()` / `hydrate(from:)` for `StandardKVCache` (unbounded + windowed, refuses wrapped windowed buffers), `AffineQuantizedKVCache`, `TurboQuantizedKVCache` (**raw mode only — compressed-mode round-trip tracked in [#197](https://github.com/ekryski/mlx-swift-lm/issues/197)**), `SSMStateCache` (via spec 020 state-replay path). Defence-in-depth `quantisationKindMismatch` guard prevents hydrating across quantisation boundaries. **Opt in** via `GenerateParameters.prefixCacheEnabled = true` or env `MLX_PREFIX_CACHE=1`. When enabled, the default policy is **auto-resolved per model family** via `AssistantOpener.detect(forModelID:)`: Qwen (1.x–3.6, QwQ) → ChatML opener; Gemma (1/2/3/4) → `<start_of_turn>model\n`; GPT-OSS → harmony; unknown families fall back to `IdentityPolicy`. `prefixCacheModelID` auto-resolves from `ModelContext.configuration.name`. Phase 4 disk persistence at `~/.cache/mlx-swift-lm/prefix/` is strictly opt-in (`prefixCacheDiskEnabled` / `MLX_PREFIX_CACHE_DISK=1`). Measured **2-10× TTFT** on multi-turn chat. Default-on flip is gated on [#197](https://github.com/ekryski/mlx-swift-lm/issues/197) (TurboQuant compressed-mode round-trip) and [#196](https://github.com/ekryski/mlx-swift-lm/issues/196) (Gemma 4 26B/31B lookup miss) — both surfaced under `--kv turbo4v2`. See [generate-parameters.md](generate-parameters.md#cross-request-prefix-kv-cache-spec-017) for full opt-in + limitations.

## What's coming

Tracked in [`specs/IMPLEMENTATION-PLAN.md`](../specs/IMPLEMENTATION-PLAN.md):

- **Spec 024 — KV-cache write fusion.** Eliminates ~60 `copy_bfloat16` dispatches per decode token on Gemma 4 E2B by fusing the per-layer K/V append into one Metal kernel. Tier 4 follow-up.
- **Spec 027 — adaptive per-layer mixed-precision KV.** Per-layer bit budget driven by an offline calibration pass (most PPL-sensitive layers at higher precision).
- **Spec 038 — Active KV cache SSD offload.** InfiniGen-style mid-generation page-out of cold pages to SSD. Disjoint from spec 017's *cross-request* disk cache; this is *single-request* memory overflow for long-context workloads on memory-constrained Macs. Tier 4, blocked on paged KV + DuoAttention.
- **Issue [#117](https://github.com/ekryski/mlx-swift-lm/issues/117) — RMSNorm + GEMV fusion.** Folds the q/k norm into the attention GEMV when the storage is quantized. Precondition for a unified `Attention` consolidation across families.

## How models use the KV cache

Every `LLMModel` / `VLMModel` provides `newCache(parameters:)`. The
factory looks at `GenerateParameters.compressionAlgorithm` and the
model's per-layer requirements (hybrid vs standard, sliding window or
not, KV-shared or not) and returns one cache per layer:

```swift
let parameters = GenerateParameters(
    compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2),
    maxKVSize: 4096       // optional: sliding-window eviction
)
let cache: [KVCache] = model.newCache(parameters: parameters)
```

Then pass to `generate(...)`:

```swift
let stream = try generate(
    input: lmInput,
    cache: cache,
    parameters: parameters,
    context: context
)
```

Or omit `cache:` and let `generate(...)` build one from the parameters
internally — the same `model.newCache(parameters:)` call site.

For multi-turn / prefix-cache reuse, keep the cache around across calls.
All three single-stream cache classes are trimmable via
`trimPromptCache(cache, numTokens:)`:

- `StandardKVCache` — trimmable in `.unbounded` mode; in `.window(maxSize)`
  mode, trimmable only while `offset < maxSize` (i.e., before the rotating
  buffer has wrapped — after wrap, the physical tail no longer corresponds
  to the logical tail).
- `AffineQuantizedKVCache` — trimmable. Both unbounded and rotating
  variants (the spec 041 phase 1.2 rotating-window cache shipped in this
  PR).
- `TurboQuantizedKVCache` — trimmable. Same wrap-time caveat as
  `StandardKVCache` applies to the rotating variant; trim adjusts the
  absolute `offset` but does not re-slice the packed K/V buffer.

For **cross-request** prefix-cache snapshot/hydrate the picture is
narrower — `PrefixKVCache.snapshot()` currently round-trips
`TurboQuantizedKVCache` in raw mode only ([#197](https://github.com/ekryski/mlx-swift-lm/issues/197);
compressed-domain round-trip is spec 039 follow-up). That's separate
from in-cache `trim(_:)`, which works on the compressed buffer directly.

For hybrid models — Qwen 3.5 / 3.6 (GDN+attention) and Mamba / Mamba 2
families (NemotronH, GraniteMoeHybrid, FalconH1; Jamba partial) —
`SSMStateCache` is not trimmable in the positional sense but supports
**state-replay rollback** for speculative decoding via specs 020 (GDN)
and 040 (Mamba). See `canRollbackPromptCache(_:)` in
`Libraries/MLXLMCommon/StateReplayCache.swift` and
[speculative-decoding.md](speculative-decoding.md#hybrid-model-coverage-spec-020).

For **cross-request prefix caching** (different `generate(...)` calls
sharing a long prompt prefix), don't manually thread the cache between
calls — opt into `GenerateParameters.prefixCacheEnabled` instead and
let the runtime snapshot / hydrate transparently. See
[generate-parameters.md](generate-parameters.md#cross-request-prefix-kv-cache-spec-017).

## Concrete classes

All implementations conform to the `KVCache` protocol. The factory
`makeAttentionCache(parameters:slidingWindow:keep:prefillStep:forceRawKV:useBias:)`
is the preferred way to instantiate one — it parses
`GenerateParameters.compressionAlgorithm` and returns the right concrete type.

**Two distinct caps drive rotating eviction**, both honored uniformly across
every compression scheme:

| Source | Where it comes from | Behaviour |
|---|---|---|
| **Architectural** (`slidingWindow: Int?`) | Per-layer flag from the model factory — set when the layer's attention path **must** attend over only the last N tokens (Gemma 4 local layers, GPT-OSS sliding layers, Mistral 3 sliding layers, etc.). | When non-nil: rotating eviction at this cap. Non-negotiable — the model architecture demands it. |
| **User budget** (`parameters?.maxKVSize`) | Read internally from the parameters tuple — the `GenerateParameters.maxKVSize` field. | When the layer is **not** architecturally sliding and `maxKVSize` is set: rotating eviction at the user's cap. Honored uniformly across `.none`, `.affine`, and `.turbo`. |

Precedence when both are set: `slidingWindow` wins. The architectural cap is
non-negotiable — the user cannot make a sliding-window layer attend over more
tokens than the architecture allows.

### What you get per scheme when the user sets `maxKVSize`

End users pass `maxKVSize` via `GenerateParameters` — they do not call
`makeAttentionCache(...)` directly. The factory is invoked internally by each
model's `newCache(parameters:)`.

```swift
let params = GenerateParameters(
    maxKVSize: 8192,
    compressionAlgorithm: .affine(bits: 4, groupSize: 64)
)

let stream = try generate(
    input: lmInput,
    parameters: params,
    context: context
)
```

| `compressionAlgorithm` | What `makeAttentionCache` returns | Behaviour |
|---|---|---|
| `.none` (default) | `StandardKVCache(maxSize: 8192, keep: 4)` | Rotating eviction at 8K, fp16 K/V. |
| `.affine(bits: 4, groupSize: 64)` | `AffineQuantizedKVCache(maxSize: 8192)` | Rotating eviction at 8K + ~3.5× affine compression. |
| `.turbo(keyBits: 4, valueBits: 2)` | `TurboQuantizedKVCache(maxSize: 8192)` | Rotating eviction at 8K + ~6-8× TurboQuant compression. |

When the user omits `maxKVSize` and the model has no architectural sliding-
window layers, all three schemes return their **unbounded** variants.

| Class | When to use it |
|---|---|
| `StandardKVCache` | Default. Uncompressed fp16 K / V. Supports unbounded growth or windowed eviction (rotating buffer). |
| `AffineQuantizedKVCache` | Affine 4 / 6 / 8-bit K and V (per-row scale + zero point). Reduces KV memory ~3.5× at 4-bit; modest decode-tok/s tax. Supports rotating eviction via the optional `maxSize` constructor parameter — engaged automatically by `makeAttentionCache(...)` when either an architectural `slidingWindow` or the user's `parameters.maxKVSize` is set. Sliding-window layers (Gemma 4 local, GPT-OSS sliding, Mistral 3 sliding, etc.) and KV-shared donor layers (Gemma 4 LLM + VLM, Gemma 3n) now retain affine compression end-to-end (closed [#202](https://github.com/ekryski/mlx-swift-lm/issues/202) via spec 041 phase 1.2 + phase 5). |
| `TurboQuantizedKVCache` | Block-wise MSE codec with separate key / value bit budgets (e.g. 4-bit K, 2-bit V — `turbo4v2`). Best memory ratio of the shipped algorithms. Decode runs the TurboFlash compressed-domain Metal kernel by default (A path) — scores directly against packed K/V with no FP16 working buffer; opt into the dequant-then-SDPA path (B) via `TURBO_DEQUANT_SDPA=1` for matmul-engine performance at the cost of a per-layer FP16 K/V working buffer. See the [batched-decoding](batched-decoding.md) deployment-shape table for memory / quality tradeoffs. |
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
| GPT-OSS-20B + `turbo*` | Coherent — `GPTOSSModel.newCache(...)` enables DC-bias correction on full-attention turbo caches and routes sliding-window layers to raw-FP16 `StandardKVCache`. Bias correction currently requires the B (dequant-SDPA) path, so GPT-OSS with `--kv turbo*` runs the B path regardless of `TURBO_DEQUANT_SDPA`. Closes [#171](https://github.com/ekryski/mlx-swift-lm/issues/171) / [#130](https://github.com/ekryski/mlx-swift-lm/issues/130). |
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
[memory-management.md](memory-management.md).

## Constructor toggles

Behaviour knobs on the cache classes themselves:

| Flag | Default | Notes |
|---|---|---|
| `TurboQuantizedKVCache.useDequantSDPA` | `false` | Decode-time attention path selector. **Default (A path)**: TurboFlash. The compressed-domain Metal kernel scores directly against packed K/V; no per-layer FP16 K/V materialisation. **Opt-in (B path, `true`)**: `bulkDequantRotated` builds a transient FP16 K/V tensor per decode step → `MLXFast.scaledDotProductAttention`. Trades a per-layer FP16 working buffer (`B*nKV*T*D*2` bytes per step) for matmul-engine throughput. Env override: `TURBO_DEQUANT_SDPA=1` forces on globally, `=0` forces off. |
| `AffineQuantizedKVCache.sdpaStrategy` | `.auto` | SDPA path selector (spec 041 phases 1+2+4). **`.auto`**: flash for L>1 (prefill) — no `[B, H, L, T]` score-tensor materialisation; discrete for L=1 (decode) — small score matrix, avoids per-step K/V dequant; sliding-window mask always picks flash (MLXFast's native sliding kernel is faster than the discrete augmented-softmax fold). **`.kernel`**: the spec 041 phase 1.1 fused Metal kernel (`fusedFlashQuantizedSDPA`) — correct but unoptimised, opt-in for kernel validation; perf parity tracked in spec 042 Phase 1b. **`.flash`**: force flash everywhere; wins on small-model long-context shapes where score materialisation dominates. **`.discrete`**: force discrete everywhere; regression-check / legacy materialise-scores path. Set via the `sdpaStrategy:` constructor parameter: `AffineQuantizedKVCache(bits: 4, sdpaStrategy: .flash)`. Env var `MLX_AFFINE_SDPA={auto,kernel,flash,discrete}` overrides this globally when set. |
| `TurboQuantizedKVCache.useBias` | `false` (per-model override) | **Per-vector DC-bias correction.** Captures the mean offset the zero-mean Lloyd-Max codebook can't represent — the dominant codec-error source on K/V from `RMSNorm → Linear(bias=True)`. Storage: one fp32 per token-head alongside `norms` (negligible vs packed indices). When enabled, the cache forces the B path because TurboFlash kernels don't yet consume the stored bias term. **Default off** for general use; **default on** for GPT-OSS-20B under `--kv turbo*` (see `Libraries/MLXLLM/Models/GPTOSS.swift`'s `newCache(...)`). Env override: `TURBO_BIAS=1`/`=0`. |
| `BatchedKVCache.maxBatch` | constructor arg | Max simultaneous decode streams sharing one cache. Must match the request shape. |
| `StandardKVCache.reserve(_:)` | not called | **Opt-in workload-size hint** for windowed eviction. Pre-allocates the rotating buffer to a known size up front (typically `prompt_length + maxTokens`) instead of growing in `step`-sized chunks. Idempotent: only takes effect before the first write. Clamped to `maxCacheSize`, floored at `step`. No-op when eviction is `.unbounded`. <br/>`let cache = StandardKVCache(maxSize: 4096); cache.reserve(promptLen + maxTokens)`. |
| Cache `step` (growth chunk) | model's `defaultPrefillStepSize` (1024 for most LLMs); falls back to `256` | The buffer grows in `step`-sized chunks per `concatenated(...)`. `makeAttentionCache(...)` threads `parameters.prefillStepSize ?? prefillStep ?? 256` uniformly into the `step:` constructor parameter of **all three** cache classes (`StandardKVCache`, `AffineQuantizedKVCache`, `TurboQuantizedKVCache`), so the per-class grow-chunk matches the model's prefill chunk 1:1 — collapsing growth events instead of paying `chunkSize/256`× the realloc cost. Override globally with `GenerateParameters.prefillStepSize`. |

Programmatic single-cache construction:
`makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2), eviction: .unbounded)` —
single source of truth for `compressionAlgorithm` string parsing.

## Environment-variable overrides

These take precedence over the constructor / `GenerateParameters`
defaults. They exist for **diagnostics, A/B testing, and tuning** —
not as the primary user-facing API. Set in the shell before launching
an inference process; read once at first use and cached.

| Variable | Effect |
|---|---|
| `TURBO_DEQUANT_SDPA=1` / `=0` | Decode-time attention path selector for `TurboQuantizedKVCache` (`useDequantSDPA` row above). `=1` forces the B path (dequant K/V to FP16 → MLXFast SDPA — adds a per-layer FP16 working buffer per decode step) globally; `=0` forces the A path (TurboFlash compressed-domain — no FP16 materialisation); unset honours the constructor. |
| `TURBO_BIAS=1` / `=0` | Per-vector DC-bias correction in the MSE codec (`useBias` row above). When enabled, the cache forces the B path because TurboFlash kernels don't yet consume the stored bias term. Default-on for GPT-OSS-20B under `--kv turbo*`; default-off elsewhere. |
| `TURBO_DEQUANT_JIT=1` | Force the JIT'd `MLXFast.metalKernel` bulk-dequant path instead of the precompiled `MLXFast.turboBulkDequantRotated`. Use for A/B comparison when iterating on the dequant kernel itself. |
| `TURBO_FLASH_BLOCK_SIZE=N` | Pin TurboFlash pass1's kernel block size (override the adaptive `tokenCount/32` heuristic). Powers of two only. |
| `TURBO_FLASH_NR0=N` | Number of query rows handled per SIMD group in the first pass of TurboFlash decode. Default `2`; `1` falls back to single-row first pass. |
| `TURBO_SPARSE_V_THRESHOLD=N` | Skip-V threshold for the separated `mseWeightedSum` kernel. Default `1e-6`. `0.0` disables; `1e-4` is too aggressive and clips long-context attention. |
| `TURBO_DEBUG=1` | Verbose logging from `compressedAttention` (offsets, shapes, key-norm sanity). Only enable for short debugging — impacts speed. |
| `MLX_AFFINE_SDPA={auto,flash,discrete}` | `AffineQuantizedKVCache` SDPA strategy (spec 041 phases 1+2+4). **`auto` (default)**: route `L > 1` (prefill) through the **flash** path (`dequantized(K) + dequantized(V) + MLXFast.scaledDotProductAttention` — no `[B, H, L, T]` score-tensor materialisation) and `L = 1` (decode) through the **discrete** path (`quantizedMM → softmax → quantizedMM` — score matrix is small at L=1 and dequanting K/V per step costs ~15% decode tok/s at long context). **`flash`**: force the flash path everywhere — wins on small-model long-context shapes where score materialisation dominates. **`discrete`**: force the discrete path everywhere — useful for A/B regression checks. Empirical: auto-strategy saves -42% peak GPU on GPT-OSS-20B 8k prefill and -60% on Gemma 4 31B 8k prefill vs discrete-only, with no decode tok/s regression. When set, this env var overrides the per-cache `AffineQuantizedKVCache.sdpaStrategy` constructor parameter (see behaviour-knob table above) for all calls in the process. |

For wired-memory env vars (`MLX_MEMORY_LIMIT`, `MLX_SMART_MEMORY`) see
[memory-management.md](memory-management.md). For model-specific perf
env vars (`GEMMA4_FUSED_NORM_ROPE`, `MLX_COMPILE_SHARED_MLP`,
`GDN_EVAL_INTERVAL`) see [generate-parameters.md](generate-parameters.md).

## See also

- [`GenerateParameters` reference](generate-parameters.md) — the full
  field table including every KV-cache parameter.
- [Memory management](memory-management.md) — wired-memory tickets that
  size their KV term from `compressionAlgorithm`.
- [Batched decoding](batched-decoding.md) — `BatchedKVCache` and
  per-stream lifecycle.
- [Migrating from 3.x](migrations/v3-to-v4.md) — the spec-006 rewrite
  (`kvScheme: "turbo4v2"` string → typed
  `compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2)`).
