# Spec 005 — Wire TurboQuant KV through `generate()` and support `RotatingKVCache`

**Status:** draft
**Target:** Unblock `kv=turbo*` for every model that uses `RotatingKVCache` for sliding-window or bounded-context layers. Primary beneficiaries: GPT-OSS-20B, Gemma 4 (E2B / E4B / 31B / 26B A4B), any model with a sliding-attention config.
**Owner:** TBD
**Expected gain:** GPT-OSS-20B decode ~51 → ~57 tok/s at ctx=1024 turbo4v2 (match `ek/tom-eric-moe-tuning` 2026-04-15 baseline). Bigger memory wins at long contexts.

## Motivation — what's broken today

Running any benchmark with `--kv turbo4v2` on alpha reports the **same KV cache footprint as `kv=none`**. Example:

```
Gemma 4 26B A4B, ctx=1024:
  kv=none      → KV Cache: 432 MB
  kv=turbo4v2  → KV Cache: 432 MB  (expected ~110 MB at 4-bit)

GPT-OSS-20B, ctx=1024:
  kv=none      → KV Cache: 54 MB
  kv=turbo4v2  → KV Cache: 54 MB  (same)
```

Root cause has three layers:

1. **`maybeQuantizeKVCache` only handles `kvBits`, not `kvScheme`.** `.turbo(bits)` / `.turboAsym(kb, vb)` return `kvBits = nil` (see [InferenceBenchmark.swift:432-435](Tests/Benchmarks/InferenceBenchmark.swift:432)) and set `kvScheme = "turbo4v2"` instead. The guard at [KVCache.swift:1790](Libraries/MLXLMCommon/KVCache.swift:1790) is `guard let kvBits = kvBits, ... else { return }` — so turbo never triggers quantization.

2. **`maybeQuantizeKVCache` has no branch for `RotatingKVCache`.** Even for `kvBits=4` (affine quant), the function only handles `KVCacheSimple`:
    ```swift
    if let simpleCache = cache[i] as? KVCacheSimple {
        cache[i] = simpleCache.toQuantized(groupSize: kvGroupSize, bits: kvBits)
    }
    // TODO: RotatingKVCache.toQuantized() is not implemented yet
    ```
    Every model with sliding attention (Gemma 4, GPT-OSS, …) builds caches via `RotatingKVCache(maxSize: slidingWindow)` in its `newCache()`. That's why affine KV quant doesn't reduce memory on those models either.

3. **`TurboQuantKVCache` exists but no one instantiates it from `generate()`.** `KVCacheSimple.toTurboQuantized(bits:keyBits:valueBits:)` exists and returns a working `TurboQuantKVCache`. There is no analog on `RotatingKVCache`, and nothing in `Evaluate.swift` calls either.

Net effect: `kv=turbo*` currently stores the scheme string on `TokenIterator` but never acts on it.

## Goals

- Make `kv=turbo4`, `kv=turbo4v2`, `kv=turbo8v4`, etc. actually engage TurboQuant compression during decode.
- Work for every cache topology we ship today: flat `[KVCache]`, mixed `[KVCacheSimple | RotatingKVCache]` (Qwen3.5-hybrid), mixed `[RotatingKVCache | StandardKVCache]` (GPT-OSS / Gemma 4 sliding+full).
- Match `ek/tom-eric-moe-tuning` turbo4v2 decode perf on GPT-OSS-20B and exceed it on Gemma 4.
- Memory: observable drop in `GenerateCompletionInfo.kvCacheBytes` (via spec 004's new accessor or the current API) for turbo runs — roughly N/4 of no-quant size at 4-bit, N/8 at ~2-bit asymmetric values.

## Non-goals

- Change the benchmark harness semantics. `kv=turbo4v2` already parses correctly; this is strictly framework plumbing.
- Ship a new `MLXFast` primitive. `TurboQuantKVCache` already has `compressedAttention` / `updateAndDequant` Metal kernels.
- Support `turbo*` on custom caches that don't use FP16 K/V tensors (MambaCache, GatedDeltaNet state) — those skip compression silently, as they do today for affine.
- Redesign `kvBits` / `kvScheme` API surface. Keep both; `kvScheme` takes precedence when set.

## Design

### 1. Add `toTurboQuantized` to `RotatingKVCache`

File: `Libraries/MLXLMCommon/KVCache.swift`.

Mirror `KVCacheSimple.toTurboQuantized` ([KVCache.swift:472](Libraries/MLXLMCommon/KVCache.swift:472)) but read keys/values from the rotating-cache state buffer, sized at `offset` — not the full step-allocated buffer:

```swift
extension RotatingKVCache {
    public func toTurboQuantized(
        bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil
    ) -> TurboQuantKVCache {
        let turbo = TurboQuantKVCache(
            bits: bits, keyBits: keyBits, valueBits: valueBits)
        turbo.offset = self.offset
        // Read in temporal order (rotating caches can be out-of-phase):
        if let (orderedKeys, orderedValues) = self.peek() {
            // TurboQuantKVCache uses the same lazy-compress strategy as
            // from-`KVCacheSimple`: set raw K/V state, compress on first
            // decode call.
            turbo.state = [orderedKeys, orderedValues]
        }
        return turbo
    }
}
```

Gotchas:
- `TurboQuantKVCache` loses the rotating behavior — once compressed, the cache doesn't evict old tokens. **For sliding-window layers this is semantically different** (no eviction). Acceptable if `turbo.offset` ≤ window size at transition; otherwise we must trim first.
- Solution: trim in temporal order during `toTurboQuantized`. `peek()` already returns trimmed `[B,H,offset,D]` tensors.
- The rotating cache's post-transition growth semantics change too — `TurboQuantKVCache` grows linearly. Document: **turbo4* on sliding-attention layers trades memory compression for loss of eviction after the compression point.** At ctx=32K on GPT-OSS this means the compressed layers hold 32K tokens of compressed data even though the original rotating cache would have kept only 128. Users hitting this tradeoff probably want `--kv affine*` (which does trim) instead. Flag in docs.

### 2. Add `maybeTurboQuantizeKVCache` in the KVCache module

Near the existing `maybeQuantizeKVCache`, mirror its structure:

```swift
/// Convert cache entries to TurboQuant compressed form once the iterator
/// has passed the start-offset. Mirrors `maybeQuantizeKVCache` but drives
/// off `kvScheme` instead of `kvBits`.
///
/// Parsing: `"turbo<k>"` → symmetric, `"turbo<k>v<v>"` → asymmetric.
/// "turbo0v4" (raw keys + 4-bit values) supported via keyBits=0.
public func maybeTurboQuantizeKVCache(
    cache: inout [KVCache],
    kvScheme: String?,
    quantizedKVStart: Int = 0
) {
    guard let scheme = kvScheme, scheme.hasPrefix("turbo"),
        !cache.isEmpty,
        !(cache[0] is TurboQuantKVCache),
        cache[0].offset > quantizedKVStart
    else { return }

    guard let (keyBits, valueBits) = parseTurboScheme(scheme) else {
        return  // unrecognized string; leave cache alone
    }

    for i in 0 ..< cache.count {
        let entry = cache[i]
        if let simple = entry as? KVCacheSimple {
            cache[i] = simple.toTurboQuantized(
                bits: valueBits, keyBits: keyBits, valueBits: valueBits)
        } else if let rotating = entry as? RotatingKVCache {
            cache[i] = rotating.toTurboQuantized(
                bits: valueBits, keyBits: keyBits, valueBits: valueBits)
        }
        // Other cache types (MambaCache, custom SSM state) skip silently —
        // they don't have quantizable FP16 K/V to compress.
    }
}

private func parseTurboScheme(_ s: String) -> (keyBits: Int, valueBits: Int)? {
    // "turbo4" → (4, 4)
    // "turbo4v2" → (4, 2)
    // "turbo0v4" → (0, 4) (rawKeyMode)
    // "turbo8v2" etc.
    let stripped = String(s.dropFirst("turbo".count))
    let parts = stripped.split(separator: "v", maxSplits: 1)
    guard let k = Int(parts[0]) else { return nil }
    let v = parts.count == 2 ? Int(parts[1]) : k
    guard let vv = v else { return nil }
    return (keyBits: k, valueBits: vv)
}
```

### 3. Call from the generation loops

File: `Libraries/MLXLMCommon/Evaluate.swift`.

The existing affine call sites ([Evaluate.swift:1064, 1194](Libraries/MLXLMCommon/Evaluate.swift:1064)) look like:
```swift
maybeQuantizeKVCache(
    cache: &cache,
    kvBits: kvBits,
    kvGroupSize: kvGroupSize,
    quantizedKVStart: quantizedKVStart
)
```

Add a turbo call **right after** each:
```swift
maybeTurboQuantizeKVCache(
    cache: &cache,
    kvScheme: kvScheme,
    quantizedKVStart: quantizedKVStart
)
```

Same pattern in `SpeculativeTokenIterator`'s main + draft update sites.

### 4. Decode-path routing (already in place)

`AttentionUtils.attentionWithCacheUpdate` ([AttentionUtils.swift:54](Libraries/MLXLMCommon/AttentionUtils.swift:54)) already has the routing for TurboQuantKVCache:

```swift
if let turboCache = cache as? TurboQuantKVCache {
    if turboCache.useCompressedAttention { return turboCache.compressedAttention(...) }
    // else: updateAndDequant + MLXFast.scaledDotProductAttention + inverseRotateOutput
}
```

No changes needed here once the cache is an actual `TurboQuantKVCache` instance.

**Exception: GPT-OSS.** Its attention block ([GPTOSS.swift:164 onwards](Libraries/MLXLLM/Models/GPTOSS.swift:164)) does **not** go through `attentionWithCacheUpdate`. It has its own inlined attention routing with only two branches: `cache as? QuantizedKVCacheProtocol` and the generic path. Once we start producing `TurboQuantKVCache` instances, GPT-OSS will fall into the generic path — which calls `cache.update(keys:values:)`. That returns wrong outputs for TurboQuant (which expects `updateAndDequant` or `compressedAttention`).

Two options for GPT-OSS:

**A.** Refactor GPT-OSS to use `attentionWithCacheUpdate`. Simplest, aligns with other models. Tom-branch had a similar inlined TurboQuant branch — mirror it at the shared call site so every model benefits.

**B.** Add an inlined `if let turbo = cache as? TurboQuantKVCache` branch inside GPT-OSS's attention, matching the tom-branch code that was removed.

Prefer A — it's cheaper to keep one attention routing path. B if A turns out to regress GPT-OSS prefill (unlikely, but `attentionWithCacheUpdate` has extra dispatch glue).

### 5. `ModelContainer.generate()` signature

No changes — `kvScheme` already flows from `GenerateParameters` into `TokenIterator`.

### 6. TurboQuantKVCache first-transition cost

`TurboQuantKVCache.toTurboQuantized(from: KVCacheSimple)` currently does the heavy encode-all lazily on the first `compressedAttention` or `updateAndDequant` call. First decode step after transition is **slow** (compresses up to `offset` tokens at once); subsequent steps are fast. For the benchmark harness this is fine — TTFT includes prefill but not the transition. For real-time streaming we should print a one-time `[TURBO] compressing {N} tokens at offset {M}` diagnostic on the first decode call so operators can spot the transition cost.

## Correctness plan

TurboQuant is **lossy**. Ship a KLD check:

```bash
# Qwen3.5-4B is small enough to baseline with bf16
scripts/benchmark.sh --model qwen35-4b --method summarization \
    --quant 4bit --kv turbo4v2 --context 1024 --kld
```

Acceptance: generation KLD within the same band as `--kv affine4` at the same bit-width. `ek/tom-eric-moe-tuning` reported `turbo4v2` KLDs on Qwen3.5-35B-A3B between 0.2 and 1.5 — stay in that range.

## Acceptance criteria

1. `scripts/benchmark.sh --model gpt-oss-20b --kv turbo4v2 --context 1024` reports `KV Cache` < 20 MB (baseline: 54 MB). Decode tok/s ≥ 55 (baseline: 51, tom 04-15: 57).
2. Same for `gemma4-26b-a4b --kv turbo4v2 --context 1024`: `KV Cache` ≤ 130 MB (baseline 432 MB at 4-bit, target ~110 MB).
3. All currently passing tests still pass. The existing `turbo*` codec unit tests in `Tests/MLXLMTests/` exercise `TurboQuantKVCache` internals directly and should not regress.
4. KL divergence on Qwen3.5-4B turbo4v2 within 1.5× the `affine4` KLD.
5. Gemma 4 E2B prefill + decode unchanged within ±2% — it uses `RotatingKVCache` for sliding layers so the new `toTurboQuantized` path gets exercised but the math has to remain equivalent.
6. Speculative decoding path (`SpeculativeTokenIterator`) continues to work when `kvScheme` is set — main + draft caches both transition.

## Measurement plan

```bash
# Primary wins we're claiming:
scripts/benchmark.sh --model gpt-oss-20b --method summarization \
    --quant 4bit --kv none,turbo4v2 --context 128,1024,4096,32768
scripts/benchmark.sh --model gemma4-26b-a4b --method summarization \
    --quant 4bit --kv none,turbo4v2 --context 1024,4096

# KLD correctness:
scripts/benchmark.sh --model qwen35-4b --method summarization \
    --quant 4bit --kv affine4,turbo4v2 --context 1024 --kld

# Regression sentinels (must not drop):
scripts/benchmark.sh --model qwen35-0.8b,gemma4-e2b --method summarization \
    --quant 4bit --kv none --context 1024
```

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `RotatingKVCache.toTurboQuantized` loses eviction → memory grows past window at long context | Documented tradeoff; add a one-line `[TURBO]` warning on transition when the source cache has `maxSize < offset`. Long-term: define a `TurboQuantRotatingKVCache` variant that keeps the window. Not blocking the first ship. |
| `TurboQuantKVCache`'s first-transition compress is slow and skews decode tok/s at ctx=128 | The transition happens once between prefill and decode. Benchmark harness already bakes TTFT into prefill tok/s. For live streaming, surface it. |
| GPT-OSS attention refactor (Option A) regresses prefill because of the extra call indirection | Benchmark before/after. If measurable, fall back to Option B. |
| `parseTurboScheme` accepts malformed strings silently, giving user the wrong cache | Unit test every registered scheme name (`turbo3`, `turbo4`, `turbo8`, `turbo3v2`, `turbo4v2`, `turbo4v3`, `turbo8v2`, `turbo8v4`) + a handful of invalid ones that must return nil. |
| `CacheList` / nested caches are missed by the `for i in 0..<cache.count` replacement loop | The current `maybeQuantizeKVCache` has the same limitation. Match behavior; file a separate follow-up to handle nested caches uniformly. |
| Turbo output quality is worse than users expect for their domain | The KLD check gates shipping. If KLD is bad, the numbers drive the conversation, not a silent regression. |

## Out of scope / follow-ups

- **`rmsNormQuantizedGEMV`-equivalent for Turbo** (fused pre-norm + turbo attention): new kernel, needs mlx upstream work.
- **`TurboQuantRotatingKVCache`** — preserves window eviction while compressing. Larger design effort.
- **Dynamic bit-width schedules** (e.g. `turbo4-after-2k`): deferred; the current `quantizedKVStart` is the only start-offset knob.
- **Turbo support for `MambaCache` / GDN state**: different compression math; tracked separately.

## Open questions

1. Should we deprecate `kvBits` in favor of `kvScheme`, with `kvBits` becoming sugar for `"affine<bits>"`? Simpler API, one code path. Leaning yes but out of scope for the first ship.
2. For Gemma 4 26B A4B's mixed-precision VLM checkpoint, do any layers have per-layer KV precision overrides that `parseTurboScheme` needs to respect? Skim the config, confirm no. If yes, per-layer turbo is a follow-up.
3. Do we want `useCompressedAttention: true` by default, or keep it off (dequant-first) unless the user opts in for long contexts? Default-off matches `TurboQuantKVCache`'s current setting. Keep.
