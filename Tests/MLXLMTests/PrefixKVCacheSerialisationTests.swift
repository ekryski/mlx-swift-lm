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
        #expect(fresh.isCompressed == false)
    }

    // MARK: - Issue #197 / #185: compressed-mode snapshot dequants to raw

    /// `serialiseTurbo` dequants compressed K/V back to raw FP16 at
    /// snapshot time. The hydrate path then always loads into
    /// `rawKeys` / `rawValues`, and warm-turn suffix prefill's
    /// `update(...)` correctly appends to the hydrated raw buffer
    /// (instead of overwriting it with a fresh zero buffer). Issue
    /// #185 root-cause fix.
    ///
    /// Drives the raw→compressed transition via the internal
    /// `compressRawCache()` so the cache enters the same state it
    /// would after the first decode step in production, then verifies
    /// the snapshot is emitted as a 2-array raw payload.
    @Test
    func `compressed-mode snapshot dequants to raw (issue #185 fix)`() throws {
        // headDim must be a power-of-2 in the WHT encode-kernel
        // instantiation set: {64, 128, 256, 512}. Smaller dims fail at
        // dispatch with "Unable to load kernel turbo_fused_encode_wht_*".
        let dim = 64
        let cache = TurboQuantizedKVCache(bits: 4)
        let (k, v) = makeRawKV(tokens: 8, dim: dim)
        cache.loadRawKV(keys: k, values: v)
        eval(cache.state)
        cache.compressRawCache()
        eval(cache.state)
        #expect(cache.isCompressed == true)
        #expect(cache.state.count == 4)  // pre-snapshot: 4-array compressed

        // After serialise: dequanted to 2-array raw.
        let captured = try serialiseLayerCacheState(cache)
        #expect(captured.arrays.count == 2)  // raw mode signature
        #expect(captured.tokenCount == 8)
        #expect(captured.metaState.isEmpty)  // raw mode → no metaState
        // The dequanted K/V shape matches the original `[B, H, T, D]`.
        #expect(captured.arrays[0].dim(2) == 8)
        #expect(captured.arrays[0].dim(3) == dim)

        let fresh = TurboQuantizedKVCache(bits: 4)
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 8)
        // Crucially, the hydrated cache is in **raw** mode — so
        // subsequent `update(...)` calls during warm-turn suffix
        // prefill will see `rawKeys` populated and append correctly
        // instead of overwriting with a zero buffer.
        #expect(fresh.isCompressed == false)
    }

    /// `rawKeyMode` compressed cache (Qwen 3.5 default — keyBits=0)
    /// snapshot dequants V from compressed packed-MSE back to raw.
    /// K was already raw in this mode, so it just round-trips.
    @Test
    func `rawKeyMode compressed snapshot dequants V to raw`() throws {
        let dim = 64
        let cache = TurboQuantizedKVCache(bits: 4, keyBits: 0, valueBits: 4)
        let (k, v) = makeRawKV(tokens: 8, dim: dim)
        cache.loadRawKV(keys: k, values: v)
        eval(cache.state)
        cache.compressRawCache()
        eval(cache.state)
        #expect(cache.isCompressed == true)
        #expect(cache.rawKeyMode == true)
        #expect(cache.state.count == 3)  // pre-snapshot: rawK + packedV + vNorms

        // After serialise: dequanted to 2-array raw [rawKeys, rawValues].
        let captured = try serialiseLayerCacheState(cache)
        #expect(captured.arrays.count == 2)
        #expect(captured.tokenCount == 8)

        let fresh = TurboQuantizedKVCache(bits: 4, keyBits: 0, valueBits: 4)
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 8)
        #expect(fresh.isCompressed == false)  // hydrated as raw
        #expect(fresh.rawKeyMode == true)  // constructor flag preserved
    }

    /// `dequantToRaw()` round-trip preserves shape and produces
    /// finite-valued output. Quantisation is lossy so we don't expect
    /// exact equality with the original input — just that the dequanted
    /// arrays are valid and have the right `[B, H, T, D]` shape.
    @Test
    func `dequantToRaw produces correctly-shaped finite output`() throws {
        let dim = 64
        let cache = TurboQuantizedKVCache(bits: 4)
        let (k, v) = makeRawKV(tokens: 8, dim: dim)
        cache.loadRawKV(keys: k, values: v)
        eval(cache.state)
        cache.compressRawCache()
        eval(cache.state)

        let (dequantK, dequantV) = try cache.dequantToRaw()
        eval(dequantK, dequantV)
        #expect(dequantK.shape == k.shape)
        #expect(dequantV.shape == v.shape)
        let kFinite = MLX.isNaN(dequantK).any().item(Bool.self)
        let vFinite = MLX.isNaN(dequantV).any().item(Bool.self)
        #expect(kFinite == false)
        #expect(vFinite == false)
    }

    /// metaState bit-width / seed / step validation triggers a precondition
    /// failure on hydrate when the target cache was constructed with
    /// non-matching parameters. (Tested via the per-class serialise/hydrate
    /// guard rather than the precondition itself.)
    @Test
    func `bit-width mismatch on hydrate throws`() throws {
        let dim = 64
        let cache = TurboQuantizedKVCache(bits: 4)
        let (k, v) = makeRawKV(tokens: 8, dim: dim)
        cache.loadRawKV(keys: k, values: v)
        eval(cache.state)
        cache.compressRawCache()
        eval(cache.state)
        let captured = try serialiseLayerCacheState(cache)

        // Different keyBits than snapshot — caught by the
        // `(snapKB, snapVB) != (cache.keyBits, cache.valueBits)` guard.
        let fresh = TurboQuantizedKVCache(bits: 2, keyBits: 2, valueBits: 2)
        #expect(throws: PrefixKVCacheError.self) {
            try hydrateLayerCache(captured, into: fresh)
        }
    }

    /// Rotating-window compressed cache snapshot dequants to raw — same
    /// behaviour as the unbounded case (issue #185 fix). The 2-array raw
    /// snapshot doesn't carry rotation metadata because raw mode doesn't
    /// have any: the hydrated cache uses `concat-then-trim` against the
    /// raw buffer on subsequent updates.
    @Test
    func `windowed compressed cache snapshot dequants to raw`() throws {
        let dim = 64
        let maxSize = 16
        let cache = TurboQuantizedKVCache(bits: 4, maxSize: maxSize)
        let (k, v) = makeRawKV(tokens: 8, dim: dim)
        cache.loadRawKV(keys: k, values: v)
        eval(cache.state)
        cache.compressRawCache()
        eval(cache.state)
        #expect(cache.isCompressed == true)
        #expect(cache.maxSize == maxSize)

        let captured = try serialiseLayerCacheState(cache)
        #expect(captured.arrays.count == 2)  // dequanted to raw
        #expect(captured.tokenCount == 8)
        #expect(captured.metaState.isEmpty)

        let fresh = TurboQuantizedKVCache(bits: 4, maxSize: maxSize)
        try hydrateLayerCache(captured, into: fresh)
        #expect(fresh.offset == 8)
        #expect(fresh.isCompressed == false)
        #expect(fresh.maxSize == maxSize)  // constructor param preserved
    }

    /// Wrapped rotating-window cache (offset > maxSize) must refuse to
    /// serialise — the circular buffer is no longer a faithful prefix.
    @Test
    func `wrapped windowed turbo cache refuses to serialise`() throws {
        let dim = 64
        let maxSize = 4
        let cache = TurboQuantizedKVCache(bits: 4, maxSize: maxSize)
        // Load 6 tokens into a maxSize=4 buffer. loadRawKV doesn't itself
        // enforce wrap semantics, but we then mark the cache as having
        // seen `offset = 8` total (post-wrap state).
        let (k, v) = makeRawKV(tokens: 4, dim: dim)
        cache.loadRawKV(keys: k, values: v, originalOffset: 8)
        eval(cache.state)
        // Offset (8) exceeds maxSize (4) → snapshot should refuse.
        #expect(cache.offset > maxSize)
        #expect(throws: PrefixKVCacheError.self) {
            _ = try serialiseLayerCacheState(cache)
        }
    }

    /// Determinism guarantee: two `MSECodec(dim, bits, seed)` instances
    /// with the same parameters must produce byte-identical rotation
    /// matrices. Locks in the cross-process / cross-run guarantee that
    /// the compressed-mode hydrate path relies on (the encoded packed
    /// indices were produced under one codec; the next request's
    /// `compressedAttention(...)` rebuilds the codec from
    /// `getOrCreateCodec(dim:bits:seed:)` — we depend on it producing
    /// the same matrices so dequant decodes the same vectors).
    @Test
    func `MSECodec rotation matrices are deterministic from (dim,bits,seed)`() throws {
        let codecA = MSECodec(dim: 64, bits: 4, seed: 42)
        let codecB = MSECodec(dim: 64, bits: 4, seed: 42)
        eval(codecA.rotation, codecB.rotation)
        let diff = MLX.abs(codecA.rotation - codecB.rotation).max().item(Float.self)
        #expect(diff == 0.0)
        // Different seed → different rotation.
        let codecC = MSECodec(dim: 64, bits: 4, seed: 43)
        eval(codecC.rotation)
        let diffSeed = MLX.abs(codecA.rotation - codecC.rotation).max().item(Float.self)
        #expect(diffSeed > 0.0)
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
