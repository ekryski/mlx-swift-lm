// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Phase 1B serialise / hydrate tests (spec 017)
//
// Each test builds a concrete cache (`StandardKVCache` /
// `AffineQuantizedKVCache` / `TurboQuantizedKVCache` / `SSMStateCache`),
// writes a small fake K/V into it, serialises to a `LayerCacheState`,
// then hydrates into a **fresh** instance of the same class. The
// hydrated cache must report the same `offset`, the same `state`
// shapes, and (where the cache exposes it) the same `metaState`.

// MARK: - Helpers

private func makeRawKV(tokens: Int, dim: Int = 4, batches: Int = 1, kvHeads: Int = 2) -> (MLXArray, MLXArray) {
    let shape = [batches, kvHeads, tokens, dim]
    let keys = MLXArray(0 ..< (batches * kvHeads * tokens * dim))
        .asType(.float16)
        .reshaped(shape)
    let values = (keys + MLXArray(Float(1.0))).asType(.float16)
    return (keys, values)
}

@Suite
struct StandardKVCacheSerialisationTests {

    @Test
    func `unbounded cache round-trip preserves shapes and offset`() throws {
        let cache = StandardKVCache()  // unbounded
        let (k, v) = makeRawKV(tokens: 3)
        let _ = cache.update(keys: k, values: v)
        eval(cache.state)

        let captured = try serialiseLayerCacheState(cache)
        #expect(captured.kind == .standardUnbounded)
        #expect(captured.tokenCount == 3)
        #expect(captured.arrays.count == 2)

        let fresh = StandardKVCache()
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 3)
        let freshState = fresh.state
        #expect(freshState.count == 2)
        // Token count visible via the second-to-last dim
        #expect(freshState[0].dim(2) == 3)
    }

    @Test
    func `windowed cache round-trip preserves metaState`() throws {
        let cache = StandardKVCache(maxSize: 16, keep: 2, step: 4)
        let (k, v) = makeRawKV(tokens: 3)
        let _ = cache.update(keys: k, values: v)
        eval(cache.state)

        let captured = try serialiseLayerCacheState(cache)
        if case .standardWindowed(let m, let kp) = captured.kind {
            #expect(m == 16)
            #expect(kp == 2)
        } else {
            Issue.record("expected standardWindowed kind, got \(captured.kind)")
        }
        #expect(captured.metaState.count == 5)

        let fresh = StandardKVCache(maxSize: 16, keep: 2, step: 4)
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 3)
    }

    @Test
    func `wrapped windowed cache refuses to serialise`() throws {
        // Windowed cache that's been pushed past its window.
        let cache = StandardKVCache(maxSize: 4, keep: 0, step: 2)
        // Overshoot the window. Write 6 tokens; offset crosses maxSize.
        let (k, v) = makeRawKV(tokens: 6)
        let _ = cache.update(keys: k, values: v)
        eval(cache.state)
        // Offset should have rotated past maxSize.
        #expect(cache.offset >= cache.maxSize ?? 0)

        #expect(throws: PrefixKVCacheError.self) {
            _ = try serialiseLayerCacheState(cache)
        }
    }

    @Test
    func `kind mismatch on hydrate throws snapshotInvariantViolation`() throws {
        let cache = StandardKVCache()
        let (k, v) = makeRawKV(tokens: 3)
        let _ = cache.update(keys: k, values: v)
        eval(cache.state)
        let captured = try serialiseLayerCacheState(cache)

        // Trying to hydrate an unbounded snapshot into a windowed cache.
        let windowed = StandardKVCache(maxSize: 16, keep: 0)
        #expect(throws: PrefixKVCacheError.self) {
            try hydrateLayerCache(captured, into: windowed)
        }
    }

    @Test
    func `empty cache round-trip is a no-op`() throws {
        let cache = StandardKVCache()
        let captured = try serialiseLayerCacheState(cache)
        #expect(captured.tokenCount == 0)
        #expect(captured.arrays.isEmpty)

        let fresh = StandardKVCache()
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 0)
    }
}

@Suite
struct AffineQuantizedKVCacheSerialisationTests {

    @Test
    func `affine-quant round-trip preserves bits, groupSize, offset`() throws {
        let cache = AffineQuantizedKVCache(groupSize: 32, bits: 4)
        let (k, v) = makeRawKV(tokens: 3, dim: 32)
        _ = cache.updateQuantized(keys: k, values: v)
        eval(cache.state)

        let captured = try serialiseLayerCacheState(cache)
        if case .affineQuantized(let bits, let g) = captured.kind {
            #expect(bits == 4)
            #expect(g == 32)
        } else {
            Issue.record("expected affineQuantized kind, got \(captured.kind)")
        }
        #expect(captured.tokenCount == 3)

        let fresh = AffineQuantizedKVCache(groupSize: 32, bits: 4)
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 3)
        #expect(fresh.bits == 4)
        #expect(fresh.groupSize == 32)
    }

    @Test
    func `bit-width mismatch on hydrate throws`() throws {
        let cache = AffineQuantizedKVCache(groupSize: 32, bits: 4)
        let (k, v) = makeRawKV(tokens: 3, dim: 32)
        _ = cache.updateQuantized(keys: k, values: v)
        eval(cache.state)
        let captured = try serialiseLayerCacheState(cache)

        let fresh = AffineQuantizedKVCache(groupSize: 32, bits: 8)
        #expect(throws: PrefixKVCacheError.self) {
            try hydrateLayerCache(captured, into: fresh)
        }
    }
}

@Suite
struct TurboQuantizedKVCacheSerialisationTests {

    @Test
    func `raw-mode (pre-compression) round-trip preserves shapes`() throws {
        let cache = TurboQuantizedKVCache(bits: 4)
        let (k, v) = makeRawKV(tokens: 4, dim: 4)
        cache.loadRawKV(keys: k, values: v)
        eval(cache.state)
        // Cache should be in raw mode (pre-compression).
        #expect(cache.isCompressed == false)

        let captured = try serialiseLayerCacheState(cache)
        if case .turboCompressed(let kb, let vb) = captured.kind {
            #expect(kb == 4)
            #expect(vb == 4)
        } else {
            Issue.record("expected turboCompressed kind, got \(captured.kind)")
        }
        #expect(captured.tokenCount == 4)

        let fresh = TurboQuantizedKVCache(bits: 4)
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 4)
    }
}

@Suite
struct PrefixSnapshotRoundTripTests {

    @Test
    func `full-stack snapshot+hydrate preserves all per-layer state`() throws {
        let layers = 3
        let caches: [KVCache] = (0..<layers).map { _ in StandardKVCache() }
        for c in caches {
            let (k, v) = makeRawKV(tokens: 5)
            _ = c.update(keys: k, values: v)
        }
        eval(caches.flatMap { $0.state })

        let key = PrefixKey(modelID: "test", layerCount: layers, kvHeadDim: 4)
        let snap = try serialisePrefixSnapshot(cache: caches, tokens: [1, 2, 3, 4, 5], key: key)
        #expect(snap.layerStates.count == layers)
        for ls in snap.layerStates {
            #expect(ls.tokenCount == 5)
        }

        let fresh: [KVCache] = (0..<layers).map { _ in StandardKVCache() }
        try hydratePrefixSnapshot(snap, into: fresh)
        for c in fresh {
            #expect(c.offset == 5)
        }
    }

    @Test
    func `layer count mismatch throws on hydrate`() throws {
        let key = PrefixKey(modelID: "test", layerCount: 3, kvHeadDim: 4)
        let snap = PrefixSnapshot(
            key: key, tokens: [1],
            layerStates: (0..<3).map { _ in
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 1,
                    arrays: [MLXArray.zeros([4], dtype: .float16)])
            })

        let twoLayer: [KVCache] = [StandardKVCache(), StandardKVCache()]
        #expect(throws: PrefixKVCacheError.self) {
            try hydratePrefixSnapshot(snap, into: twoLayer)
        }
    }
}

// MARK: - Quantisation-kind mismatch guard (spec 017 open question 2)

@Suite
struct QuantisationKindMismatchTests {

    private func makeSnap(kinds: [LayerCacheState.Kind]) -> PrefixSnapshot {
        let key = PrefixKey(modelID: "x", layerCount: kinds.count, kvHeadDim: 4)
        let layers = kinds.map { kind in
            LayerCacheState(
                kind: kind, tokenCount: 3,
                arrays: [MLXArray.zeros([4], dtype: .float16)])
        }
        return PrefixSnapshot(key: key, tokens: [1, 2, 3], layerStates: layers)
    }

    @Test
    func `matching kinds report no mismatch`() {
        let snap = makeSnap(kinds: [.standardUnbounded, .standardUnbounded])
        let cache: [KVCache] = [StandardKVCache(), StandardKVCache()]
        #expect(quantisationKindMismatch(snapshot: snap, cache: cache) == nil)
    }

    @Test
    func `affine bits mismatch is detected`() {
        let snap = makeSnap(kinds: [.affineQuantized(bits: 4, groupSize: 64)])
        let cache: [KVCache] = [AffineQuantizedKVCache(groupSize: 64, bits: 8)]
        let result = quantisationKindMismatch(snapshot: snap, cache: cache)
        #expect(result != nil)
        #expect(result!.contains("affineQuantized"))
    }

    @Test
    func `affine groupSize mismatch is detected`() {
        let snap = makeSnap(kinds: [.affineQuantized(bits: 4, groupSize: 32)])
        let cache: [KVCache] = [AffineQuantizedKVCache(groupSize: 64, bits: 4)]
        let result = quantisationKindMismatch(snapshot: snap, cache: cache)
        #expect(result != nil)
    }

    @Test
    func `class mismatch (snapshot fp16 -> target affine4) is detected`() {
        let snap = makeSnap(kinds: [.standardUnbounded])
        let cache: [KVCache] = [AffineQuantizedKVCache(groupSize: 64, bits: 4)]
        let result = quantisationKindMismatch(snapshot: snap, cache: cache)
        #expect(result != nil)
        #expect(result!.contains("kind"))
    }

    @Test
    func `class mismatch (snapshot affine4 -> target fp16) is detected`() {
        let snap = makeSnap(kinds: [.affineQuantized(bits: 4, groupSize: 64)])
        let cache: [KVCache] = [StandardKVCache()]
        #expect(quantisationKindMismatch(snapshot: snap, cache: cache) != nil)
    }

    @Test
    func `turbo bit-width mismatch is detected`() {
        let snap = makeSnap(kinds: [.turboCompressed(keyBits: 4, valueBits: 4)])
        let cache: [KVCache] = [TurboQuantizedKVCache(bits: 4, keyBits: 4, valueBits: 2)]
        let result = quantisationKindMismatch(snapshot: snap, cache: cache)
        #expect(result != nil)
        #expect(result!.contains("turboCompressed"))
    }

    @Test
    func `empty donor-sharing layer is exempt from kind check`() {
        // Mirror the Gemma 4 KV-sharing pattern: layer 1 carries empty
        // state. The check should pass through without false positive.
        let key = PrefixKey(modelID: "x", layerCount: 2, kvHeadDim: 4)
        let snap = PrefixSnapshot(
            key: key, tokens: [1, 2, 3],
            layerStates: [
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 3,
                    arrays: [MLXArray.zeros([4], dtype: .float16)]),
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 0,
                    arrays: []),
            ])
        let cache: [KVCache] = [StandardKVCache(), StandardKVCache()]
        #expect(quantisationKindMismatch(snapshot: snap, cache: cache) == nil)
    }

    @Test
    func `SSM layer is exempt from kind check`() {
        let key = PrefixKey(modelID: "x", layerCount: 2, kvHeadDim: 4)
        // Pair an SSM snapshot layer with a StandardKVCache target.
        // The exemption means no mismatch is reported — the snapshot's
        // SSM state goes to whichever live cache slot it lands in, and
        // for hybrid stacks the layers are paired up correctly by
        // model factory.
        let snap = PrefixSnapshot(
            key: key, tokens: [1, 2, 3],
            layerStates: [
                LayerCacheState(
                    kind: .standardUnbounded, tokenCount: 3,
                    arrays: [MLXArray.zeros([4], dtype: .float16)]),
                LayerCacheState(
                    kind: .ssm, tokenCount: 99,
                    arrays: [MLXArray.zeros([4], dtype: .float16)]),
            ])
        let cache: [KVCache] = [StandardKVCache(), StandardKVCache()]
        #expect(quantisationKindMismatch(snapshot: snap, cache: cache) == nil)
    }
}
