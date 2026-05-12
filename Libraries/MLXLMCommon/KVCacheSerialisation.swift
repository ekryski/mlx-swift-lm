// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Per-class serialise / hydrate for the prefix KV cache (spec 017 phase 1B)
//
// `PrefixKVCache.insert(...)` stores a ``PrefixSnapshot`` whose
// ``LayerCacheState`` is **kind**-discriminated. The hydrate path uses
// the discriminator to reconstruct the right concrete cache class and
// validate shape + dtype.
//
// `serialise()` is the snapshot-capture entrypoint. It runs at the end of
// prefill (after the iterator has primed the cache through the stable
// prefix). For unbounded caches the snapshot is `state` + `metaState`
// straight off the cache; for windowed caches that have wrapped
// (`offset > maxSize`), `serialise()` throws ``PrefixKVCacheError/wrappedWindowedCache``
// — the cache's circular buffer no longer maps onto a faithful prefix
// snapshot.
//
// `hydrate(into:)` is the snapshot-restore entrypoint. It populates a
// freshly-constructed cache from a ``LayerCacheState``. Each layer's
// kind must match the target cache class; mismatches throw
// ``PrefixKVCacheError/snapshotInvariantViolation(_:)``.

/// One-shot per-cache snapshot capture. Returns a populated
/// ``LayerCacheState`` whose `kind` matches the concrete cache class.
///
/// - Parameter cache: the per-layer cache. Must be in a state suitable
///   for snapshotting — for windowed caches that means `offset <= maxSize`.
/// - Returns: a ``LayerCacheState`` ready for embedding in a
///   ``PrefixSnapshot``.
/// - Throws: ``PrefixKVCacheError/wrappedWindowedCache`` if the cache is
///   windowed and has wrapped; ``PrefixKVCacheError/snapshotInvariantViolation(_:)``
///   if the cache's state shape is malformed.
public func serialiseLayerCacheState(_ cache: KVCache) throws -> LayerCacheState {
    switch cache {
    case let standard as StandardKVCache:
        return try serialiseStandard(standard)
    case let quantized as AffineQuantizedKVCache:
        return try serialiseAffineQuantized(quantized)
    case let turbo as TurboQuantizedKVCache:
        return try serialiseTurbo(turbo)
    case let ssm as SSMStateCache:
        return try serialiseSSM(ssm)
    case let list as CacheList:
        // Composite caches are out of scope for phase 1B — the iterator
        // can route a CacheList through its children individually, but
        // mixed-kind nesting is not yet a supported snapshot shape.
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "CacheList with \(list.state.count) child states is not snapshot-able in phase 1B")
    default:
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "unsupported cache class for prefix snapshot: \(type(of: cache))")
    }
}

/// One-shot per-cache snapshot restore. Populates `cache` from a
/// ``LayerCacheState`` whose `kind` matches the target class.
///
/// - Parameters:
///   - state: the layer's captured state.
///   - cache: a freshly-constructed cache to hydrate. Must be empty
///     (no prior writes). Caller is responsible for ensuring the
///     cache's class matches `state.kind`.
/// - Throws: ``PrefixKVCacheError/snapshotInvariantViolation(_:)`` on
///   kind/class mismatch.
public func hydrateLayerCache(_ state: LayerCacheState, into cache: KVCache) throws {
    switch (state.kind, cache) {
    case (.standardUnbounded, let standard as StandardKVCache):
        guard case .unbounded = standard.eviction else {
            throw PrefixKVCacheError.snapshotInvariantViolation(
                "expected unbounded StandardKVCache; got \(standard.eviction)")
        }
        try hydrateStandard(state, into: standard)
    case (.standardWindowed, let standard as StandardKVCache):
        guard case .window = standard.eviction else {
            throw PrefixKVCacheError.snapshotInvariantViolation(
                "expected windowed StandardKVCache; got \(standard.eviction)")
        }
        try hydrateStandard(state, into: standard)
    case (.affineQuantized, let quantized as AffineQuantizedKVCache):
        try hydrateAffineQuantized(state, into: quantized)
    case (.turboCompressed, let turbo as TurboQuantizedKVCache):
        try hydrateTurbo(state, into: turbo)
    case (.ssm, let ssm as SSMStateCache):
        try hydrateSSM(state, into: ssm)
    default:
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "kind \(state.kind) does not match cache class \(type(of: cache))")
    }
}

// MARK: - StandardKVCache

private func serialiseStandard(_ cache: StandardKVCache) throws -> LayerCacheState {
    let s = cache.state
    let meta = cache.metaState
    let kind: LayerCacheState.Kind
    switch cache.eviction {
    case .unbounded:
        kind = .standardUnbounded
    case .window(let size, let keep):
        // dflash-mlx's `target_cache_is_serializable(...)` refuses to
        // snapshot a `RotatingKVCache` whose buffer has wrapped — the
        // circular buffer is no longer a faithful prefix.
        if cache.offset > size {
            throw PrefixKVCacheError.wrappedWindowedCache
        }
        kind = .standardWindowed(maxSize: size, keep: keep)
    }
    // `state` may be empty if the cache was never written to. That's
    // legal — `tokenCount == 0` and an empty arrays list round-trip.
    return LayerCacheState(
        kind: kind, tokenCount: cache.offset,
        arrays: s.map { $0[.ellipsis] }, metaState: meta)
}

private func hydrateStandard(_ state: LayerCacheState, into cache: StandardKVCache) throws {
    // Empty snapshot is a no-op — leaves the cache at its initial state.
    if state.arrays.isEmpty {
        return
    }
    guard state.arrays.count == 2 else {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "StandardKVCache state expects 2 arrays, got \(state.arrays.count)")
    }
    cache.state = state.arrays
    if !state.metaState.isEmpty {
        cache.metaState = state.metaState
    }
    // For unbounded eviction the offset is restored via the state setter;
    // for windowed the offset is restored via metaState. Either way, the
    // caller's tokenCount must match.
    if cache.offset != state.tokenCount {
        // Unbounded edge case: state setter assigns offset = keys.dim(2),
        // but the snapshot's tokenCount is the **prefix-stable** length,
        // not the allocation. If they differ, force-set.
        cache.offset = state.tokenCount
    }
}

// MARK: - AffineQuantizedKVCache

private func serialiseAffineQuantized(_ cache: AffineQuantizedKVCache) throws -> LayerCacheState {
    let s = cache.state
    let meta = cache.metaState
    // state has 4 elements (biases nil) or 6 elements (biases present).
    if !s.isEmpty && s.count != 4 && s.count != 6 {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "AffineQuantizedKVCache state must have 4 or 6 arrays, got \(s.count)")
    }
    return LayerCacheState(
        kind: .affineQuantized(bits: cache.bits, groupSize: cache.groupSize),
        tokenCount: cache.offset,
        arrays: s.map { $0[.ellipsis] },
        metaState: meta)
}

private func hydrateAffineQuantized(
    _ state: LayerCacheState, into cache: AffineQuantizedKVCache
) throws {
    if state.arrays.isEmpty {
        return
    }
    guard case .affineQuantized(let bits, let groupSize) = state.kind else {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "AffineQuantizedKVCache hydrate received non-affine kind")
    }
    guard bits == cache.bits, groupSize == cache.groupSize else {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "AffineQuantizedKVCache bit-width / group-size mismatch: "
            + "snapshot=(\(bits), \(groupSize)), target=(\(cache.bits), \(cache.groupSize))")
    }
    cache.state = state.arrays
    if !state.metaState.isEmpty {
        cache.metaState = state.metaState
    }
    // metaState setter restores offset directly.
    if cache.offset != state.tokenCount {
        cache.offset = state.tokenCount
    }
}

// MARK: - TurboQuantizedKVCache

private func serialiseTurbo(_ cache: TurboQuantizedKVCache) throws -> LayerCacheState {
    // Phase 1B only captures pre-compression raw state. The
    // compressed-domain snapshot is a phase-4 follow-up because the
    // codec depends on per-layer rotation matrices that aren't currently
    // serialised; restoring would require either rebuilding the codec
    // (deterministic from `seed` + `dim` + `bits`) or storing the
    // rotation matrices alongside the snapshot. Defer.
    if cache.isCompressed {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "TurboQuantizedKVCache snapshot in compressed mode is not supported in phase 1B")
    }
    let s = cache.state
    return LayerCacheState(
        kind: .turboCompressed(keyBits: cache.keyBits, valueBits: cache.valueBits),
        tokenCount: cache.offset,
        arrays: s.map { $0[.ellipsis] },
        metaState: [])
}

private func hydrateTurbo(
    _ state: LayerCacheState, into cache: TurboQuantizedKVCache
) throws {
    if state.arrays.isEmpty {
        return
    }
    guard case .turboCompressed(let keyBits, let valueBits) = state.kind else {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "TurboQuantizedKVCache hydrate received non-turbo kind")
    }
    guard keyBits == cache.keyBits, valueBits == cache.valueBits else {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "TurboQuantizedKVCache bit-width mismatch: "
            + "snapshot=(\(keyBits)v\(valueBits)), "
            + "target=(\(cache.keyBits)v\(cache.valueBits))")
    }
    // Phase 1B: assume raw-mode snapshot (2 arrays [rawKeys, rawValues]).
    cache.state = state.arrays
}

// MARK: - SSMStateCache (spec 017 phase 3 / spec 020 phase 5)

private func serialiseSSM(_ cache: SSMStateCache) throws -> LayerCacheState {
    // Spec 020 phase 5 contract: SSM state replay lets us snapshot
    // hybrid models. Cache snapshot is the current `state` (conv +
    // recurrent slots) — both slots round-trip through the existing
    // `state` accessor.
    if !cache.canStateReplay {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "SSMStateCache has canStateReplay = false (e.g. Mamba) — snapshot not supported")
    }
    let s = cache.state
    let meta = cache.metaState
    return LayerCacheState(
        kind: .ssm,
        tokenCount: cache.offset,
        arrays: s.map { $0[.ellipsis] },
        metaState: meta)
}

private func hydrateSSM(_ state: LayerCacheState, into cache: SSMStateCache) throws {
    if state.arrays.isEmpty {
        return
    }
    // SSMStateCache uses ArraysCache's restoreFromMetaState path. The
    // metaState format is `[slotCount, presentSlots, ...]`; restore via
    // that path so slot-index nils are preserved.
    if !state.metaState.isEmpty {
        cache.restoreFromMetaState(state: state.arrays, savedMetaState: state.metaState)
    } else {
        // Best-effort: assign state directly. Loses slot-index info but
        // works when the cache has both slots present.
        cache.state = state.arrays
    }
    cache.offset = state.tokenCount
}

// MARK: - Full-stack helpers

/// Serialise an array of per-layer caches into a ``PrefixSnapshot``.
/// Maps each layer through ``serialiseLayerCacheState(_:)``.
///
/// - Parameters:
///   - cache: the per-layer cache array (length == model layer count).
///   - tokens: the stable-prefix tokens this snapshot covers.
///   - key: the target identity key.
///   - lastHidden: optional final hidden state (DFlash only — leave nil
///     for baseline / n-gram iterators).
public func serialisePrefixSnapshot(
    cache: [KVCache], tokens: [Int], key: PrefixKey, lastHidden: MLXArray? = nil
) throws -> PrefixSnapshot {
    let layerStates = try cache.map { try serialiseLayerCacheState($0) }
    return PrefixSnapshot(
        key: key, tokens: tokens, layerStates: layerStates, lastHidden: lastHidden)
}

/// Hydrate an array of per-layer caches from a ``PrefixSnapshot``.
/// Caller is responsible for constructing the right cache shape (same
/// layer count, same eviction / quantization configuration); typically
/// this comes from `model.newCache(parameters:)`.
public func hydratePrefixSnapshot(
    _ snapshot: PrefixSnapshot, into cache: [KVCache]
) throws {
    guard cache.count == snapshot.layerStates.count else {
        throw PrefixKVCacheError.snapshotInvariantViolation(
            "cache layer count (\(cache.count)) != snapshot layer count "
            + "(\(snapshot.layerStates.count))")
    }
    for (layer, ls) in zip(cache, snapshot.layerStates) {
        try hydrateLayerCache(ls, into: layer)
    }
}

// MARK: - Key inference

/// Best-effort ``PrefixKey`` derivation from a per-layer cache array.
/// Used by the generate-path wiring to construct a key without forcing
/// the caller to plumb model config through every callsite.
///
/// - Parameters:
///   - cache: the per-layer cache. Must be non-empty.
///   - modelID: stable identifier (typically `ModelConfiguration.name`).
///   - kvBits: explicit override; nil = infer from the first layer's
///     `storageKind`.
public func prefixKey(forCache cache: [KVCache], modelID: String, kvBits: Int? = nil) -> PrefixKey {
    precondition(!cache.isEmpty, "prefixKey: cache must be non-empty")
    let layerCount = cache.count
    var inferredBits: Int? = kvBits
    if inferredBits == nil {
        switch cache[0].storageKind {
        case .affineQuantized(let bits, _): inferredBits = bits
        case .turboCompressed(let kb, let vb): inferredBits = max(kb, vb)
        case .raw, .ssm, .composite: inferredBits = nil
        }
    }
    // KV head dim is not directly exposed by the cache; we use a
    // best-effort placeholder of 0. The key is hashed and compared, so
    // any monotonic value would do — what matters is that the same
    // model configuration always produces the same value. Inferring
    // from the cache's `state` shape is expensive (requires a write
    // first); instead, callers that care about cross-config rejection
    // should pass an explicit key.
    return PrefixKey(
        modelID: modelID,
        layerCount: layerCount,
        kvHeadDim: 1,  // placeholder; see comment above
        kvBits: inferredBits)
}
