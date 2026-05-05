import Foundation
import MLX
import MLXLMCommon
import Testing

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { StandardKVCache() },
    { StandardKVCache(maxSize: 32) },
    { AffineQuantizedKVCache() },
    { ArraysCache(size: 2) },
    { SSMStateCache() },
]

@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as AffineQuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as AffineQuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    let originalOffset = original.offset
    let originalState = original.state
    eval(originalState)
    let originalMeta = original.metaState

    // copy
    let copied = original.copy()

    // same type
    #expect(type(of: original) == type(of: copied))

    // same offset and metadata
    #expect(copied.offset == originalOffset)
    #expect(copied.metaState == originalMeta)

    // same state values
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (origArr, copyArr) in zip(originalState, copiedState) {
        #expect(origArr.shape == copyArr.shape)
        #expect(allClose(origArr, copyArr).item(Bool.self))
    }

    // mutate the copy — push more tokens through it
    let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

    switch copied {
    case let arrays as ArraysCache:
        // overwrite slot 0 with a different array
        arrays[0] = moreKeys
    case let quantized as AffineQuantizedKVCache:
        _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
    default:
        _ = copied.update(keys: moreKeys, values: moreValues)
    }

    // original must be unchanged
    #expect(original.offset == originalOffset)
    #expect(original.metaState == originalMeta)
    let currentState = original.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (origArr, savedArr) in zip(currentState, originalState) {
        #expect(origArr.shape == savedArr.shape)
        #expect(allClose(origArr, savedArr).item(Bool.self))
    }
}

/// copy() on an empty (unpopulated) cache must not crash.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
    let empty = creator()
    let copied = empty.copy()

    #expect(type(of: empty) == type(of: copied))
    #expect(copied.offset == 0)
    #expect(copied.state.count == empty.state.count)
}

// MARK: - StandardKVCache.reserve regression coverage

@Test("StandardKVCache.reserve preallocates the buffer to the hinted size")
func testRotatingKVCacheReservePreallocates() async throws {
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    cache.reserve(800)

    let keys = MLXArray.ones([1, 8, 64, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    // Buffer should be 800 (the hint, capped above step), not 256 (default step).
    #expect(cache.innerState()[0].shape == [1, 8, 800, 128])
}

@Test("StandardKVCache.reserve is no-op after first write")
func testRotatingKVCacheReserveIsIdempotentAfterWrite() async throws {
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    let keys = MLXArray.ones([1, 8, 64, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    // Cache already populated with step-sized buffer; reserve should not retroactively grow it.
    let originalSize = cache.innerState()[0].dim(2)
    cache.reserve(2048)
    #expect(cache.innerState()[0].dim(2) == originalSize)
}

@Test("StandardKVCache.reserve clamps to maxCacheSize")
func testRotatingKVCacheReserveClampsToMax() async throws {
    let cache = StandardKVCache(maxSize: 256, step: 64)
    cache.reserve(10000)  // way over maxCacheSize

    let keys = MLXArray.ones([1, 8, 32, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    #expect(cache.innerState()[0].dim(2) == 256)
}

@Test("StandardKVCache.reserve floor: at least step")
func testRotatingKVCacheReserveFloorIsStep() async throws {
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    cache.reserve(100)  // smaller than step

    let keys = MLXArray.ones([1, 8, 32, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    // Buffer should be at least step (256) even though hint was 100.
    #expect(cache.innerState()[0].dim(2) >= 256)
}

@Test("StandardKVCache.reserve buffer fits multi-token writes within hint")
func testRotatingKVCacheReserveFitsPrefillChunks() async throws {
    // Simulates the typical iterator wiring: hint = prompt + maxTokens, then
    // multi-token prefill writes the prompt in chunks. Each chunk must land
    // in the pre-allocated buffer without triggering a grow.
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    let promptLen = 1024
    let maxTokens = 128
    cache.reserve(promptLen + maxTokens)

    // Two prefill chunks of 512 each.
    let chunk1 = MLXArray.ones([1, 8, 512, 128], dtype: .bfloat16)
    let chunk2 = MLXArray.zeros([1, 8, 512, 128], dtype: .bfloat16)

    _ = cache.update(keys: chunk1, values: chunk1)
    _ = cache.update(keys: chunk2, values: chunk2)

    // After both writes, the buffer should still be the originally allocated size
    // (no growth needed) and the offset should reflect both writes.
    #expect(cache.offset == 1024)
    #expect(cache.innerState()[0].dim(2) == promptLen + maxTokens)
}

@Test("StandardKVCache.reserve grows past hint when workload exceeds it")
func testRotatingKVCacheReserveGrowsOnOverflow() async throws {
    // If the actual workload exceeds the hinted size, the cache should fall
    // back to step-based growth — the hint isn't a hard cap, just a
    // first-allocation size.
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    cache.reserve(512)

    // First chunk fits in the hint.
    let chunk1 = MLXArray.ones([1, 8, 256, 128], dtype: .bfloat16)
    _ = cache.update(keys: chunk1, values: chunk1)
    #expect(cache.innerState()[0].dim(2) == 512)

    // Second chunk overflows the 512-token hint and must grow the buffer.
    let chunk2 = MLXArray.zeros([1, 8, 384, 128], dtype: .bfloat16)
    _ = cache.update(keys: chunk2, values: chunk2)
    #expect(cache.offset == 256 + 384)
    #expect(cache.innerState()[0].dim(2) >= 256 + 384)
}

// MARK: - StandardKVCache under-utilisation + rotation regression coverage
//
// Independent of `reserve(_:)` — covers the path the existing 11 tests in
// KVCacheTests.swift never exercised: step-incremental growth without
// rotation, and the rotation/wrap behaviour past `maxCacheSize`. Locks the
// existing semantics so refactors (incl. the spec 006 consolidation into
// `StandardKVCache`) can prove byte-identical behaviour.

@Test("StandardKVCache step-incremental growth (under-util, no reserve)")
func testRotatingKVCacheUnderUtilisationStepGrowth() async throws {
    // Push 1024 single-token writes. With default step=256 and maxSize=4096
    // the buffer should grow 256 → 512 → 768 → 1024, never reaching 4096.
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< 1024 {
        _ = cache.update(keys: token, values: token)
    }
    #expect(cache.offset == 1024)
    // Buffer should be 1024 (last step boundary), not full maxCacheSize.
    #expect(cache.innerState()[0].dim(2) == 1024)
    #expect(cache.innerState()[0].dim(2) < 4096)
}

@Test("StandardKVCache stays trimmable while under maxCacheSize (no eviction)")
func testRotatingKVCacheUnderUtilisationNoEviction() async throws {
    // After 1024 writes (< maxSize=4096), the cache should still be in the
    // pre-rotation phase: isTrimmable=true and offset advances normally.
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< 1024 {
        _ = cache.update(keys: token, values: token)
    }
    #expect(cache.isTrimmable == true)
    // metaState layout: [keep, maxCacheSize, step, offset, idx]
    #expect(cache.metaState[3] == "1024")
    let idx = Int(cache.metaState[4])!
    #expect(idx == 1024)
    #expect(idx < 4096)
}

@Test("StandardKVCache dispatcher unchanged when reserve() is unused")
func testRotatingKVCacheNoReserveDispatcherUnchanged() async throws {
    // Locks the back-compat guarantee: when initialAllocSize == nil, the
    // dispatcher should route S=1 → updateInPlace (step-sized buffer) and
    // S>1 → updateConcat (buffer == S). Pre-PR behaviour, byte-identical.
    let single = StandardKVCache(maxSize: 4096, step: 256)
    let oneToken = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    _ = single.update(keys: oneToken, values: oneToken)
    // S=1 path: in-place, step-sized buffer.
    #expect(single.innerState()[0].dim(2) == 256)

    let multi = StandardKVCache(maxSize: 4096, step: 256)
    let sixteenTokens = MLXArray.ones([1, 8, 16, 128], dtype: .bfloat16)
    _ = multi.update(keys: sixteenTokens, values: sixteenTokens)
    // S>1 path with no reserve: concat — buffer matches the chunk size, not step.
    #expect(multi.innerState()[0].dim(2) == 16)
}

@Test("StandardKVCache rotation kicks in past maxCacheSize")
func testRotatingKVCacheRotationKicksInPastMax() async throws {
    // Push 96 single-token writes through a maxSize=64 cache. Rotation
    // should engage: buffer stays at 64, offset keeps counting upward, peek()
    // returns the live window in temporal order.
    let cache = StandardKVCache(maxSize: 64, step: 16)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< 96 {
        _ = cache.update(keys: token, values: token)
    }
    #expect(cache.offset == 96)
    #expect(cache.innerState()[0].dim(2) == 64)
    #expect(cache.isTrimmable == false)
    let peeked = cache.peek()
    #expect(peeked != nil)
    #expect(peeked!.0.dim(2) == 64)
}

@Test("StandardKVCache rotation works correctly with reserve()")
func testRotatingKVCacheRotationWithReserve() async throws {
    // reserve(maxCacheSize) → pre-allocate full window; then write 2× max
    // tokens. Buffer stays at maxCacheSize; rotation engages exactly once.
    let maxSize = 64
    let cache = StandardKVCache(maxSize: maxSize, step: 16)
    cache.reserve(maxSize)

    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< (2 * maxSize) {
        _ = cache.update(keys: token, values: token)
    }

    #expect(cache.offset == 2 * maxSize)
    #expect(cache.innerState()[0].dim(2) == maxSize)
    // idx should have wrapped (idx != offset means rotation occurred).
    let idx = Int(cache.metaState[4])!
    #expect(idx != cache.offset)
    #expect(idx <= maxSize)
}

@Test("StandardKVCache reserve respects keep across rotation")
func testRotatingKVCacheReserveWithKeepRotation() async throws {
    // With keep=4, the first 4 slots must be preserved across rotation.
    // Write distinguishable sentinels into the keep region by going through
    // ones-only and then zeros-only writes; after rotation the keep region
    // should still hold the original ones.
    let cache = StandardKVCache(maxSize: 32, keep: 4, step: 8)
    cache.reserve(64)  // clamped to 32

    // First 4 writes: ones (these populate the keep region).
    let onesToken = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< 4 {
        _ = cache.update(keys: onesToken, values: onesToken)
    }
    // Next 60 writes: zeros (force rotation by exceeding maxCacheSize).
    let zerosToken = MLXArray.zeros([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< 60 {
        _ = cache.update(keys: zerosToken, values: zerosToken)
    }

    #expect(cache.offset == 64)
    #expect(cache.innerState()[0].dim(2) == 32)
    // Keep region: positions 0..<4 of the buffer should still be the ones
    // we wrote — rotation overwrites slots in [keep, maxSize) only.
    let buffer = cache.innerState()[0]
    let keepSlice = buffer[.ellipsis, ..<4, 0...]
    let expected = MLXArray.ones([1, 8, 4, 128], dtype: .bfloat16)
    eval(keepSlice, expected)
    #expect(allClose(keepSlice, expected).item(Bool.self))
}

@Test("StandardKVCache reserve(maxCacheSize) — exact boundary")
func testRotatingKVCacheReserveExactMaxBoundary() async throws {
    // reserve at exactly maxCacheSize: first allocation should be the full
    // buffer; subsequent writes up to maxCacheSize should not trigger any
    // re-allocation.
    let maxSize = 256
    let cache = StandardKVCache(maxSize: maxSize, step: 64)
    cache.reserve(maxSize)

    let chunk = MLXArray.ones([1, 8, 64, 128], dtype: .bfloat16)
    _ = cache.update(keys: chunk, values: chunk)
    #expect(cache.innerState()[0].dim(2) == maxSize)

    // Three more 64-token chunks should fit without growth.
    for _ in 0 ..< 3 {
        _ = cache.update(keys: chunk, values: chunk)
    }
    #expect(cache.innerState()[0].dim(2) == maxSize)
    #expect(cache.offset == maxSize)
}

@Test("StandardKVCache reserve(0) and negative are no-ops")
func testRotatingKVCacheReserveZeroAndNegativeNoOp() async throws {
    // reserve(0) and reserve(-N) should be silent no-ops. First write should
    // produce a step-sized buffer (the legacy default), proving
    // initialAllocSize stayed nil.
    let zeroCache = StandardKVCache(maxSize: 4096, step: 256)
    zeroCache.reserve(0)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    _ = zeroCache.update(keys: token, values: token)
    #expect(zeroCache.innerState()[0].dim(2) == 256)

    let negativeCache = StandardKVCache(maxSize: 4096, step: 256)
    negativeCache.reserve(-5)
    _ = negativeCache.update(keys: token, values: token)
    #expect(negativeCache.innerState()[0].dim(2) == 256)
}

@Test("StandardKVCache reserve scales with batch dim B>1")
func testRotatingKVCacheReserveBatchedB2() async throws {
    // reserve(_:) controls dimension 2 (token dim); batch dim should be
    // inferred from the first write. Run B=2 and B=4 to confirm.
    let cacheB2 = StandardKVCache(maxSize: 4096, step: 256)
    cacheB2.reserve(800)
    let chunkB2 = MLXArray.ones([2, 8, 64, 128], dtype: .bfloat16)
    _ = cacheB2.update(keys: chunkB2, values: chunkB2)
    #expect(cacheB2.innerState()[0].shape == [2, 8, 800, 128])

    let cacheB4 = StandardKVCache(maxSize: 4096, step: 256)
    cacheB4.reserve(800)
    let chunkB4 = MLXArray.ones([4, 8, 64, 128], dtype: .bfloat16)
    _ = cacheB4.update(keys: chunkB4, values: chunkB4)
    #expect(cacheB4.innerState()[0].shape == [4, 8, 800, 128])
}

@Test("StandardKVCache reserve allocates once for chunked writes within hint")
func testRotatingKVCacheReserveAllocationOnceOnly() async throws {
    // reserve(2048) + three 256-token chunks (768 total) → buffer should
    // remain at 2048 after every write. Proxy for "no concat happened".
    let cache = StandardKVCache(maxSize: 4096, step: 256)
    cache.reserve(2048)

    let chunk = MLXArray.ones([1, 8, 256, 128], dtype: .bfloat16)
    for _ in 0 ..< 3 {
        _ = cache.update(keys: chunk, values: chunk)
        #expect(cache.innerState()[0].dim(2) == 2048)
    }
    #expect(cache.offset == 768)
}

@Test("StandardKVCache no-reserve back-compat: full grow + rotate cycle")
func testRotatingKVCacheBackcompatExistingPath() async throws {
    // Smoke-test the legacy growth + rotation path with default settings.
    // Push 33 single-token writes through maxSize=32, default step=256.
    // Step gets clamped against maxSize-prev so the buffer grows to 32 then
    // rotation engages on the 33rd write.
    let cache = StandardKVCache(maxSize: 32)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)

    for _ in 0 ..< 32 {
        _ = cache.update(keys: token, values: token)
    }
    // Pre-rotation: buffer at maxSize, offset == maxSize.
    #expect(cache.offset == 32)
    #expect(cache.innerState()[0].dim(2) == 32)

    // 33rd write triggers rotation.
    _ = cache.update(keys: token, values: token)
    #expect(cache.offset == 33)
    #expect(cache.innerState()[0].dim(2) == 32)
    #expect(cache.isTrimmable == false)
    // peek() should still return the maxSize window in temporal order.
    let peeked = cache.peek()
    #expect(peeked != nil)
    #expect(peeked!.0.dim(2) == 32)
}

/// CacheList.copy() produces independent sub-caches.
@Test
func testCacheListCopyIsIndependent() async throws {
    let sub1 = StandardKVCache()
    let sub2 = StandardKVCache(maxSize: 32)
    let composite = CacheList(sub1, sub2)

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = sub1.update(keys: keys, values: values)
    _ = sub2.update(keys: keys, values: values)

    // snapshot original state — eval to materialize before copy
    let originalState = composite.state
    eval(originalState)
    let originalOffset0 = sub1.offset
    let originalOffset1 = sub2.offset

    let copied = composite.copy()

    #expect(copied is CacheList)
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (orig, copy) in zip(originalState, copiedState) {
        #expect(orig.shape == copy.shape)
        #expect(allClose(orig, copy).item(Bool.self))
    }

    // mutate inside the copy
    let copiedList = copied as! CacheList
    _ = copiedList[0].update(
        keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    )

    // originals unchanged
    #expect(sub1.offset == originalOffset0)
    #expect(sub2.offset == originalOffset1)
    let currentState = composite.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (orig, saved) in zip(currentState, originalState) {
        #expect(orig.shape == saved.shape)
        #expect(allClose(orig, saved).item(Bool.self))
    }
}

// MARK: - Spec 006 PR 1: typed surface coverage
//
// Locks in the new type system introduced by spec 006 PR 1:
//   * StandardKVCache (consolidates StandardKVCache + StandardKVCache)
//   * AffineQuantizedKVCache (rename of AffineQuantizedKVCache)
//   * TurboQuantizedKVCache (rename of TurboQuantizedKVCache)
//   * SSMStateCache (rename of SSMStateCache)
//   * KVStorage / KVEviction / KVStorageKind / KVCache.CompressionAlgorithm
//   * makeKVCache(scheme:eviction:) factory
//
// Typealiases keep the old names alive — these tests verify that the typealias
// chain resolves correctly and that the new typed surface produces the same
// behavior as the legacy classes.

@Test("StandardKVCache default init matches legacy StandardKVCache shape (typealias identity)")
func testStandardKVCacheUnboundedTypealiasIdentity() async throws {
    // StandardKVCache should now be a typealias of StandardKVCache, so they're
    // literally the same type at runtime. This locks the typealias direction
    // (PR 1 flipped: StandardKVCache is primary, StandardKVCache aliases it).
    let standard: KVCache = StandardKVCache()
    let legacy: KVCache = StandardKVCache()
    #expect(type(of: standard) == type(of: legacy))
    #expect(String(describing: type(of: standard)) == "StandardKVCache")
}

@Test("StandardKVCache windowed convenience init matches StandardKVCache typealias")
func testStandardKVCacheWindowedConvenienceInitTypealias() async throws {
    // StandardKVCache should now alias StandardKVCache. The convenience init
    // `init(maxSize:keep:step:)` produces an instance with `eviction == .window(...)`.
    let rotating: any KVCache = StandardKVCache(maxSize: 32, keep: 4, step: 16)
    #expect(type(of: rotating) == StandardKVCache.self)
    let std = rotating as! StandardKVCache
    if case .window(let size, let keep) = std.eviction {
        #expect(size == 32)
        #expect(keep == 4)
    } else {
        Issue.record("Expected .window eviction; got \(std.eviction)")
    }
    #expect(std.step == 16)
}

@Test("StandardKVCache unbounded grows step-incrementally and stays trimmable")
func testStandardKVCacheUnboundedStepGrowth() async throws {
    // Locks the legacy StandardKVCache growth shape: buffer grows in step-multiples
    // (default step=256) and isTrimmable is always true.
    let cache = StandardKVCache(eviction: .unbounded)
    let token = MLXArray.ones([1, 8, 1, 64], dtype: .bfloat16)
    for _ in 0 ..< 600 {
        _ = cache.update(keys: token, values: token)
    }
    // After 600 writes with step=256, buffer should be 768 (3 × 256).
    #expect(cache.innerState()[0].dim(2) == 768)
    #expect(cache.offset == 600)
    #expect(cache.isTrimmable == true)
    #expect(cache.metaState == [""])
    #expect(cache.storageKind == .raw)
}

@Test("StandardKVCache windowed rotates correctly and exposes legacy 5-element metaState")
func testStandardKVCacheWindowedRotationAndMetaState() async throws {
    // Locks the legacy StandardKVCache rotation shape + metaState format.
    let cache = StandardKVCache(eviction: .window(size: 16, keep: 4), step: 4)
    let token = MLXArray.ones([1, 8, 1, 64], dtype: .bfloat16)

    // 24 writes through a maxSize=16 cache → rotation engages.
    for _ in 0 ..< 24 {
        _ = cache.update(keys: token, values: token)
    }
    #expect(cache.offset == 24)
    #expect(cache.innerState()[0].dim(2) == 16)
    #expect(cache.isTrimmable == false)
    #expect(cache.storageKind == .raw)

    // metaState shape: [keep, maxCacheSize, step, offset, idx].
    let meta = cache.metaState
    #expect(meta.count == 5)
    #expect(meta[0] == "4")
    #expect(meta[1] == "16")
    #expect(meta[3] == "24")
}

@Test("makeKVCache factory produces the right concrete class for every scheme")
func testMakeKVCacheFactoryAllSchemes() async throws {
    // .none → StandardKVCache (raw)
    let none = makeKVCache(scheme: .none, eviction: .unbounded)
    #expect(type(of: none) == StandardKVCache.self)
    #expect(none.storageKind == .raw)

    // .none + window → StandardKVCache (raw, windowed)
    let noneWindow = makeKVCache(scheme: .none, eviction: .window(size: 64, keep: 0))
    #expect(type(of: noneWindow) == StandardKVCache.self)
    let stdWindow = noneWindow as! StandardKVCache
    if case .window(let size, _) = stdWindow.eviction {
        #expect(size == 64)
    } else {
        Issue.record("Expected .window eviction")
    }

    // .affine(...) → AffineQuantizedKVCache
    let affine = makeKVCache(scheme: .affine(bits: 4, groupSize: 64))
    #expect(type(of: affine) == AffineQuantizedKVCache.self)
    if case .affineQuantized(let bits, let groupSize) = affine.storageKind {
        #expect(bits == 4)
        #expect(groupSize == 64)
    } else {
        Issue.record("Expected .affineQuantized storageKind")
    }

    // .turbo(...) → TurboQuantizedKVCache
    let turbo = makeKVCache(scheme: .turbo(keyBits: 4, valueBits: 2))
    #expect(type(of: turbo) == TurboQuantizedKVCache.self)
    if case .turboCompressed(let kb, let vb) = turbo.storageKind {
        #expect(kb == 4)
        #expect(vb == 2)
    } else {
        Issue.record("Expected .turboCompressed storageKind")
    }
}

@Test("KVCache.CompressionAlgorithm parser round-trips every supported string format")
func testCompressionAlgorithmStringParseRoundTrip() async throws {
    typealias Algo = KVCache.CompressionAlgorithm

    // None / empty. Use `.some(Algo.none)` to disambiguate from `Optional.none`.
    #expect(Algo("none") == .some(Algo.none))
    #expect(Algo("") == .some(Algo.none))
    #expect(Algo("NONE") == .some(Algo.none))
    #expect(Algo("none")?.description == "none")

    // Symmetric turbo.
    #expect(Algo("turbo4") == .turbo(keyBits: 4, valueBits: 4))
    #expect(Algo("turbo4")?.description == "turbo4")

    // Asymmetric turbo.
    #expect(Algo("turbo4v2") == .turbo(keyBits: 4, valueBits: 2))
    #expect(Algo("turbo4v2")?.description == "turbo4v2")

    // Raw-key turbo.
    #expect(Algo("turbo0v4") == .turbo(keyBits: 0, valueBits: 4))
    #expect(Algo("turbo0v4")?.description == "turbo0v4")

    // Affine, default group size.
    #expect(Algo("affine4") == .affine(bits: 4, groupSize: 64))
    #expect(Algo("affine4")?.description == "affine4")

    // Affine, custom group size.
    #expect(Algo("affine4g32") == .affine(bits: 4, groupSize: 32))
    #expect(Algo("affine4g32")?.description == "affine4g32")

    // Whitespace + case insensitivity.
    #expect(Algo("  Turbo4V2  ") == .turbo(keyBits: 4, valueBits: 2))

    // Reject malformed.
    #expect(Algo("bogus") == nil)
    #expect(Algo("turbo") == nil)  // No digit suffix.
    #expect(Algo("turboabc") == nil)
    #expect(Algo("affine") == nil)
}

@Test("storageKind reflects the concrete cache class for every type")
func testStorageKindOnEveryCacheType() async throws {
    let standard: any KVCache = StandardKVCache()
    #expect(standard.storageKind == .raw)

    let rotating: any KVCache = StandardKVCache(maxSize: 64)
    #expect(rotating.storageKind == .raw)  // Both eviction shapes hold raw K/V.

    let affine: any KVCache = AffineQuantizedKVCache(groupSize: 64, bits: 4)
    if case .affineQuantized(let bits, let groupSize) = affine.storageKind {
        #expect(bits == 4)
        #expect(groupSize == 64)
    } else {
        Issue.record("AffineQuantizedKVCache should expose .affineQuantized")
    }

    let turbo: any KVCache = TurboQuantizedKVCache(bits: 4, keyBits: 4, valueBits: 2)
    if case .turboCompressed(let kb, let vb) = turbo.storageKind {
        #expect(kb == 4)
        #expect(vb == 2)
    } else {
        Issue.record("TurboQuantizedKVCache should expose .turboCompressed")
    }

    let ssm: any KVCache = SSMStateCache()
    #expect(ssm.storageKind == .ssm)

    let composite: any KVCache = CacheList(StandardKVCache(), StandardKVCache())
    #expect(composite.storageKind == .composite)
}

@Test("Old class names are typealiases of the new consolidated classes")
func testTypealiasIdentities() async throws {
    // StandardKVCache == StandardKVCache (post-flip).
    let _: StandardKVCache.Type = StandardKVCache.self

    // StandardKVCache == StandardKVCache.
    let _: StandardKVCache.Type = StandardKVCache.self

    // AffineQuantizedKVCache == AffineQuantizedKVCache.
    let _: AffineQuantizedKVCache.Type = AffineQuantizedKVCache.self

    // TurboQuantizedKVCache == TurboQuantizedKVCache.
    let _: TurboQuantizedKVCache.Type = TurboQuantizedKVCache.self

    // SSMStateCache == SSMStateCache.
    let _: SSMStateCache.Type = SSMStateCache.self

    // Constructor-via-typealias should produce an instance of the new class.
    let viaOldName: any KVCache = SSMStateCache()
    #expect(type(of: viaOldName) == SSMStateCache.self)
}

@Test("Persistence emits 'KVCache' for unbounded and 'StandardKVCache' for windowed StandardKVCache")
func testPersistenceClassNameDispatchByEviction() async throws {
    // Save + load a heterogeneous cache list. Verify that the saver picks
    // the right class name based on eviction (since both unbounded and
    // windowed are now StandardKVCache class-identity).
    let unbounded = StandardKVCache(eviction: .unbounded)
    let windowed = StandardKVCache(eviction: .window(size: 16, keep: 0), step: 4)
    let token = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = unbounded.update(keys: token, values: token)
    _ = windowed.update(keys: token, values: token)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: [unbounded, windowed], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 2)
    // Loaded[0] should round-trip as unbounded; loaded[1] as windowed.
    let std0 = loaded[0] as? StandardKVCache
    let std1 = loaded[1] as? StandardKVCache
    #expect(std0 != nil)
    #expect(std1 != nil)
    if let std0 {
        #expect(std0.eviction == .unbounded)
    }
    if let std1, case .window(let size, _) = std1.eviction {
        #expect(size == 16)
    } else {
        Issue.record("Expected windowed eviction on second loaded cache")
    }
}

@Test("StandardKVCache.reserve(_:) on unbounded eviction is a silent no-op")
func testReserveOnUnboundedIsNoOp() async throws {
    // Reserve is window-only (it controls the rotating buffer's first
    // allocation). On unbounded, calling it should not change behaviour:
    // first write produces a step-sized buffer (the legacy StandardKVCache shape).
    let cache = StandardKVCache(eviction: .unbounded)
    cache.reserve(2048)  // No-op on unbounded.

    let token = MLXArray.ones([1, 8, 1, 64], dtype: .bfloat16)
    _ = cache.update(keys: token, values: token)

    // Buffer should be step (256), not the reserve hint (2048).
    #expect(cache.innerState()[0].dim(2) == 256)
}

@Test("StandardKVCache.reserve(_:) on windowed eviction matches the PR #152 behaviour")
func testReserveOnWindowedMatchesPR152() async throws {
    // Locks back-compat: the existing reserve behavior we shipped on the
    // legacy StandardKVCache via PR #152 must work identically through the
    // typealias and through direct StandardKVCache construction.
    let cache = StandardKVCache(eviction: .window(size: 4096, keep: 0), step: 256)
    cache.reserve(800)

    let token = MLXArray.ones([1, 8, 64, 128], dtype: .bfloat16)
    _ = cache.update(keys: token, values: token)
    // Buffer should be exactly the hint (800).
    #expect(cache.innerState()[0].dim(2) == 800)
}

@Test("StandardKVCache toQuantized works for both eviction shapes")
func testToQuantizedDispatchesOnEviction() async throws {
    // Unbounded: simple linear quantization.
    let unbounded = StandardKVCache(eviction: .unbounded)
    let token = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = unbounded.update(keys: token, values: token)
    let unboundedQuant = unbounded.toQuantized(groupSize: 64, bits: 4)
    #expect(unboundedQuant.offset == 4)
    #expect(unboundedQuant.storageKind == .affineQuantized(bits: 4, groupSize: 64))

    // Windowed: must reorder into temporal sequence first (else group
    // boundaries don't align with token order). 8 writes through a 16-token
    // window — pre-rotation, so just need the offset-trim path to work.
    let windowed = StandardKVCache(eviction: .window(size: 16, keep: 0), step: 4)
    for _ in 0 ..< 8 {
        _ = windowed.update(keys: token, values: token)
    }
    let windowedQuant = windowed.toQuantized(groupSize: 64, bits: 4)
    #expect(windowedQuant.offset == windowed.offset)
    #expect(windowedQuant.storageKind == .affineQuantized(bits: 4, groupSize: 64))
}
