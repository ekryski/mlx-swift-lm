# 039 — Compressed-domain prefix KV cache (lossless + low-memory)

**Status:** Spec drafted 2026-05-12. Builds on top of [spec 017](017-prefix-kv-cache.md)'s plumbing (post-prefill snapshot hook, `PrefixKVCache.shared` LRU, `LastAssistantOpenerPolicy` family routing, disk persistence). **Not started.**
**Branch:** TBD (`ek/039-compressed-prefix-cache-phase1` once implementation begins)
**Depends on:** spec 017 phases 1–5 (shipped via PRs [#144](https://github.com/ekryski/mlx-swift-lm/pull/144) + [#198](https://github.com/ekryski/mlx-swift-lm/pull/198)). No hard dependency on any other Tier 1+ spec.

## Problem

Spec 017's Option A snapshot-post-prefill timing eliminates the dequant cost and SSM staleness, but the on-the-wire snapshot is still **raw FP16 K/V at full prefix size**:

- Qwen3.5-35B-A3B at 32K → ~1.6 GB per cached prefix
- Gemma 4 26B-A4B at 8K → ~440 MB per cached prefix
- Gemma 4 31B at 4K → ~210 MB per cached prefix

The default `PrefixKVCache(maxBytes: 8 GiB, maxEntries: 4)` budget holds ~4–5 long-context prefixes before LRU evicts. On warm-turn hydrate, we load the FP16 K/V into a fresh `TurboQuantizedKVCache` in raw mode; the first decode step then compresses them — paying compression cost on every warm turn, **plus** an FP16 → packed round-trip's worth of precision shift on every load.

**Goal:** snapshot the cache in its compressed-domain representation, hydrate it back into compressed buffers, and run the warm suffix prefill entirely in the compressed domain. Result: **~4× smaller snapshots** (turbo4v2 ratio), **zero compression / dequant on the warm path**, and **lossless precision** (the same compressed bytes round-trip).

## The pieces that already exist

Spec 017 + spec 020 + the TurboQuant runtime ship most of the primitives. Inventory:

### 1. Post-prefill snapshot hook

`PrefixCacheRouteState.snapshotPostPrefill(cache:)` (introduced in spec 017 phase 5) fires immediately after `model.prepare(...)` returns. At that timing:

- `TurboQuantizedKVCache.isCompressed == false` (compression triggers on the first decode step, not prefill).
- Sliding-window buffers have not yet wrapped.
- `SSMStateCache` reflects only the prompt's evolution.

Spec 039 hooks into the same call site — no new iterator-level plumbing.

### 2. Batch compressed encoder

`TurboQuantizedKVCache.fusedEncodeDispatch(input:codec:headDim:)` (called from `compressRawCache` + `encodeNewToken`) is **already batch-friendly**: it accepts `[B, H, T, D]` raw K/V and emits `[B, H, T, packedWidth] uint32` packed indices + `[B, H, T]` float norms via a single Metal kernel dispatch. We get O(prefix_len) snapshot compression for free.

### 3. Compressed-domain attention for L > 1

`TurboQuantizedKVCache.compressedAttention(queries:keys:values:scale:mask:)` already supports `L > 1` queries. From the doc comment (TurboQuantKVCache.swift:1541–1546):

> For L=1 (decode): uses TurboFlashAttention — a single Metal dispatch that fuses Q×K scoring + online softmax + Attn×V aggregation.
> For L>1 (prefill chunks): falls back to separate score → softmax → value kernels since causal masking across multiple query positions requires the full score matrix.

The L > 1 path already exists. The limiter today is `attentionWithCacheUpdate` in `AttentionUtils.swift`, which routes L > 1 to the raw-cache path:

```swift
case .turboCompressed:
    let L = queries.dim(2)
    if L > 1 {
        // Prefill (L>1): raw update + standard SDPA. Zero overhead.
        let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(...)
    }
    // L == 1: compressedAttention runs in compressed domain
    return turboCache.compressedAttention(queries: queries, ...)
```

Spec 039 rewires this dispatcher to call `compressedAttention(L>1)` when the cache is **already compressed at entry** (i.e. hydrated from a compressed snapshot).

### 4. Stored encoder state

`TurboQuantizedKVCache`'s compressed buffers (`keyPackedMSE`, `keyNorms`, `valPackedMSE`, `valNorms`) plus its `metaState` array (introduced for spec 017 issue #197) carry the codec identity (`seed`, `keyBits`, `valueBits`), the buffer geometry (`rotatingMaxSize`, `step`), and the live position counters (`offset`, `isCompressed`, `rotatingIdx`, `compressedWriteOffset`). The deterministic-from-seed `MSECodec` rebuilds on the hydrate side with byte-identical rotation matrices.

## Design

### A. Compress-on-snapshot in `serialiseTurbo`

Currently `serialiseTurbo` (`KVCacheSerialisation.swift`) emits a 2-array raw payload — either directly off `cache.state` (Option A path) or via `dequantToRaw()` (fallback path). Spec 039 changes the Option A path:

```swift
private func serialiseTurbo(
    _ cache: TurboQuantizedKVCache, upTo: Int? = nil
) throws -> LayerCacheState {
    // ...wrap checks unchanged...
    if cache.isCompressed {
        // Fallback path: dequant → raw (lossy, retained for non-Option-A callers).
        return try serialiseTurboRawFallback(cache, upTo: upTo)
    }
    // Spec 039 path: encode the prefix slice into compressed-domain form
    // before storing. Hydrate then loads compressed buffers directly.
    let stableLen = upTo ?? cache.offset
    let (kPacked, kNorms, vPacked, vNorms) = try cache.compressPrefixSlice(
        upTo: stableLen)
    return LayerCacheState(
        kind: .turboCompressed(keyBits: cache.keyBits, valueBits: cache.valueBits),
        tokenCount: stableLen,
        arrays: cache.rawKeyMode
            ? [kPacked, vPacked, vNorms]   // rawKeyMode: K stays raw
            : [kPacked, kNorms, vPacked, vNorms],
        metaState: cache.metaState)  // round-trips codec params + geometry
}
```

The new helper `TurboQuantizedKVCache.compressPrefixSlice(upTo:)` is a one-shot batch encode of `rawKeys[0..stableLen]` + `rawValues[0..stableLen]` through `fusedEncodeDispatch`. Single Metal dispatch per (K, V) per layer; runs once per cold turn at snapshot time. Output: the four packed/norms arrays sliced to `stableLen`.

For rawKeyMode (Qwen 3.5 / NemotronH default), K stays raw FP16 — only V gets the compress-on-snapshot path. The snapshot array shape stays the existing 3-array form `[rawKeys, valPacked, valNorms]`.

### B. Compressed-mode hydrate in `hydrateTurbo`

`hydrateTurbo` already knows how to load 3-array (rawKeyMode compressed) and 4-array (standard compressed) state shapes via the `state` setter dispatch (TurboQuantKVCache.swift:1932–1952). What's missing today is the **metaState restoration** path — currently the post-prefill snapshot emits raw state with `metaState = []`, so the cache lands as raw mode regardless of the original shape.

Spec 039 just stops emitting `metaState = []` for the Option A path. The existing setter restores `isCompressed = true` + `rotatingIdx` + `compressedWriteOffset` from the new (compressed) snapshot's metaState. No code change in `hydrateTurbo`; the state-setter dispatch and metaState round-trip already handle it.

### C. L > 1 dispatch on hydrated compressed cache

Rewire `attentionWithCacheUpdate` in `AttentionUtils.swift` to route L > 1 through `compressedAttention(...)` when `cache.isCompressed == true`:

```swift
case .turboCompressed:
    let L = queries.dim(2)
    if turboCache.useCompressedAttention && sinks == nil
        && (L == 1 || turboCache.isCompressed)
    {
        // Compressed-domain attention. Handles both:
        //   - L=1 decode (existing path, TurboFlashAttention single dispatch)
        //   - L>1 warm suffix prefill on a hydrated compressed cache (NEW —
        //     uses the score→softmax→value fallback already implemented)
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask)
    }
    // Cold turn raw prefill, sinks-using model, or compressed-attention
    // explicitly disabled — falls through to the raw update + SDPA path.
    if L > 1 {
        let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(...)
    }
    // ...rest unchanged...
```

`compressedAttention(...)` already calls `encodeNewToken(...)` (which handles any `numSteps` value) to write the new tokens into the packed buffers, then runs the L > 1 score → softmax → value path. The new tokens land in the compressed buffer alongside the hydrated prefix; no raw allocation, no dequant.

### D. `encodeNewToken` batch path

The existing `encodeNewToken(keys:values:)` in `TurboQuantKVCache.swift:1316+` already supports `numSteps = newKeys.dim(2)` as an integer parameter — it slices `valPackedMSE[..., writeIdx..(writeIdx+numSteps), ...]` and writes the batch. No code change needed. The only requirement: the compressed buffers must be **allocated to ≥ writeIdx + numSteps**. For a fresh cache hydrated from a `stableLen`-sized snapshot, the buffers are allocated to `stableLen`; `encodeNewToken` already has the grow path for this case (TurboQuantKVCache.swift:1339–1356).

### E. Disk-persistence wire compatibility

`PrefixKVCacheDisk` (spec 017 phase 4) serialises `PrefixSnapshot` via `MLX.save(arrays:url:)`. Compressed buffers are `uint32` (packed indices) + `float32` (norms) — both supported by `safetensors`. The on-disk fingerprint includes the format version; spec 039 bumps `PrefixKey.currentFormatVersion` to `3` so v2 (Option A raw) snapshots on disk are rejected as mismatched and re-created in compressed form on next write.

## Phasing

1. **Phase 1 — `compressPrefixSlice(upTo:)` + serialiseTurbo wiring.** Compress on snapshot, hydrate from compressed via existing state-setter path. Bench: snapshot bytes ≥ 70% reduction vs current Option A on Qwen3.5-35B-A3B / Gemma 4 26B-A4B / Gemma 4 31B at 1K / 8K / 32K contexts (TurboQuant ratio is 4× for V at 2-bit, varies for K).
2. **Phase 2 — `attentionWithCacheUpdate` L > 1 routing.** Rewire dispatcher to call `compressedAttention(L>1)` on hydrated compressed caches. Bench: warm-turn TTFT and decode tok/s vs spec 017 Option A baseline. Expectation: equal-or-better warm TTFT (no FP16 → compressed re-conversion), same decode tok/s (compressed path is identical post-prefill).
3. **Phase 3 — `PrefixKey.formatVersion = 3` bump + disk migration.** L2 disk format change. Old v2 snapshots ignored on read. New writes are v3 compressed.
4. **Phase 4 — Mamba / Mamba 2 SSM compression** (deferred). The TurboQuant compressed-mode pipeline does not apply to SSM caches; the SSM state is small (constant per layer regardless of context) so the win is marginal. Out-of-scope for spec 039. Mamba prefix-cache coverage proper comes from [spec 040](040-mamba-state-replay.md) (the state-replay kernel pair); spec 039 then covers Mamba's attention layers automatically (Jamba / Granite-MoE-Hybrid have attention layers that use TurboQuant).

## Expected impact

| Model              | ctx  | Spec 017 (Option A) snapshot bytes | Spec 039 snapshot bytes | Ratio |
|--------------------|------|------------------------------------|--------------------------|-------|
| Qwen3.5-0.8B       | 32K  | 848 MB                             | ~250 MB                  | 3.4×  |
| Qwen3.5-35B-A3B    | 4K   | ~210 MB                            | ~62 MB                   | 3.4×  |
| Qwen3.5-35B-A3B    | 32K  | ~1.65 GB                           | ~485 MB                  | 3.4×  |
| Gemma 4 26B-A4B    | 4K   | ~440 MB                            | ~130 MB                  | 3.4×  |
| Gemma 4 31B        | 4K   | ~210 MB                            | ~62 MB                   | 3.4×  |

Ratios assume turbo4v2 (K=4 / V=2 bit packed; norms add ~3% overhead). At the same `maxBytes` budget, spec 039 holds ~3.4× more cached prefixes — the difference between an LRU that always evicts on the 5th turn (current) versus one that holds a dozen multi-turn chat sessions resident.

**Warm-turn TTFT:** equal or marginally better. The current Option A path loads raw FP16 K/V into a fresh cache; first decode step then compresses them via `compressRawCache`. Spec 039 skips both the raw-load + compress steps — hydrate writes compressed buffers directly into the cache, no first-decode transition. Estimated savings: ~50–200 ms per warm turn at long context, mostly hidden behind decode.

**Decode tok/s:** unchanged. After the first decode step, both paths run identical compressed-domain decode.

## Test plan

- `Tests/MLXLMTests/PrefixKVCacheSerialisationTests.swift`: extend with a `compressed-mode round-trip preserves logits` test. Prefill 1K tokens, snapshot via spec 039 path, hydrate into fresh cache, run one decode step in compressed-domain attention, compare logits against the original cache's first decode step (max abs diff < tolerance — lossless, so should be exact).
- New `Tests/MLXLMTests/PrefixKVCacheCompressedAttentionTests.swift`: L > 1 dispatch on hydrated compressed cache produces same logits as the equivalent cold-path turn (modulo TurboQuant's inherent compression-vs-FP16 delta, which is the same for both paths).
- Fleet bench: re-run the spec 017 fleet matrix (Qwen 3.5 0.8B / 35B-A3B, Gemma 4 E2B / E4B / 26B-A4B / 31B, GPT-OSS-20B fallback path) with `--method summarization --kv turbo4v2 --cache-prefix --context 1K,8K,32K` and confirm:
  1. snapshot bytes ≥ 70% reduction vs spec 017,
  2. warm-turn TTFT equal-or-better,
  3. decode tok/s unchanged,
  4. output stays coherent (no precision regression on the lossless path).

## Files touched (estimated)

| File | Phase | Change |
|---|---|---|
| `Libraries/MLXLMCommon/TurboQuantKVCache.swift` | 1 | New `compressPrefixSlice(upTo:)` public method (~40 LOC). |
| `Libraries/MLXLMCommon/KVCacheSerialisation.swift` | 1 | `serialiseTurbo` Option A branch routes through `compressPrefixSlice` instead of raw state. |
| `Libraries/MLXLMCommon/AttentionUtils.swift` | 2 | Add `(L > 1 && cache.isCompressed)` branch in `attentionWithCacheUpdate` to dispatch `compressedAttention`. |
| `Libraries/MLXLMCommon/PrefixKVCache.swift` | 3 | Bump `PrefixKey.currentFormatVersion` 2 → 3. |
| `Libraries/MLXLMCommon/PrefixKVCacheDisk.swift` | 3 | Reject v2 entries on read; existing format-version check already handles this. |
| `Tests/MLXLMTests/PrefixKVCacheSerialisationTests.swift` | 1 | Compressed round-trip test. |
| `Tests/MLXLMTests/PrefixKVCacheCompressedAttentionTests.swift` (new) | 2 | L > 1 hydrate + dispatch test. |
| `Tests/Benchmarks/InferenceBenchmark.swift` | bench | Already supports `--cache-prefix` on the `summarization` method (PR [#198](https://github.com/ekryski/mlx-swift-lm/pull/198)). No change. |

## Out of scope (for this spec)

- **Mamba / Mamba 2 compressed snapshot.** The Mamba family already opts out of prefix-cache via `canStateReplay == false`. Once [spec 040](040-mamba-state-replay.md) lands the Mamba state-replay kernel, Mamba snapshots will be **raw SSM state** — TurboQuant compression isn't applicable.
- **GPT-OSS-20B attention sinks on TurboQuant Path B.** Compressed-mode attention on sinks-using models is gated on the cross-repo sinks PR chain (mlx#16 + mlx-c#8 + mlx-swift#18 + mlx-swift-lm#99). Spec 039's L > 1 dispatcher branch checks `sinks == nil` and falls through to raw on sinks-using models. Independent track.
- **Spec 017's Option A path itself.** Spec 039 builds *on* the Option A snapshot timing; the post-prefill hook stays exactly as PR #198 shipped it. Option A remains the correct hook for any future cache-format variation (paged, retrieval, hybrid).

## References

- [Spec 017](017-prefix-kv-cache.md) — prefix KV cache base (phases 1–5)
- [Spec 020](020-tape-replay-rollback-generalised.md) — state-replay rollback (SSM rollback primitive)
- [PR #198](https://github.com/ekryski/mlx-swift-lm/pull/198) — spec 017 known-limitations close-out + Option A
- TurboQuant compressed attention: `Libraries/MLXLMCommon/TurboQuantKVCache.swift::compressedAttention`, `compressRawCache`, `encodeNewToken`
- Attention dispatcher: `Libraries/MLXLMCommon/AttentionUtils.swift::attentionWithCacheUpdate`
