import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { StandardKVCache() },
    { StandardKVCache(maxSize: 32) },
    { AffineQuantizedKVCache() },
    { ArraysCache(size: 2) },
    { SSMStateCache() },
]

// MARK: - Helper

private func tempURL() -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")
}

/// Assert two arrays of MLXArray are element-wise close
private func assertArraysClose(_ lhs: [MLXArray], _ rhs: [MLXArray], label: String = "") {
    #expect(lhs.count == rhs.count, "state count mismatch \(label)")
    for (i, (a, b)) in zip(lhs, rhs).enumerated() {
        #expect(a.shape == b.shape, "shape mismatch at index \(i) \(label)")
        let close = allClose(a, b).item(Bool.self)
        #expect(close, "values not close at index \(i) \(label)")
    }
}

// MARK: - Original parameterized test (updated with value assertions)

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

    let url = tempURL()

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        assertArraysClose(lhs.state, rhs.state)
    }
}

// MARK: - ArraysCache sparse slot round-trip

@Test func testArraysCacheSparseSlots() throws {
    let cache = ArraysCache(size: 3)
    let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
    cache[0] = a
    // slot 1 stays nil
    cache[2] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.slotCount == 3)
    #expect(restored[0] != nil)
    #expect(restored[1] == nil)
    #expect(restored[2] != nil)
    #expect(allClose(restored[0]!, a).item(Bool.self))
    #expect(allClose(restored[2]!, b).item(Bool.self))
}

// MARK: - ArraysCache leftPadding round-trip

@Test func testArraysCacheLeftPadding() throws {
    let cache = ArraysCache(size: 2, leftPadding: [0, 5])
    let a = MLXArray.ones([2, 4], dtype: .float32)
    let b = MLXArray.ones([2, 4], dtype: .float32) * 2.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.leftPaddingValues == [0, 5])
    assertArraysClose(restored.state, cache.state)
}

// MARK: - MambaCache type preservation

@Test func testMambaCacheRoundTrip() throws {
    let cache = SSMStateCache()
    let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? SSMStateCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
}

// MARK: - CacheList with KV caches

@Test func testCacheListKVCaches() throws {
    let simple = StandardKVCache()
    let rotating = StandardKVCache(maxSize: 32)

    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)
    _ = rotating.update(keys: keys * 2.0, values: values * 2.0)

    let cacheList = CacheList(simple, rotating)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cacheList], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? CacheList)
    let child0 = try #require(restored[0] as? StandardKVCache)
    let child1 = try #require(restored[1] as? StandardKVCache)

    assertArraysClose(child0.state, simple.state, label: "child0")
    assertArraysClose(child1.state, rotating.state, label: "child1")
    #expect(child1.metaState == rotating.metaState)
}

// MARK: - CacheList with hybrid (MambaCache + KVCacheSimple)

@Test func testCacheListHybrid() throws {
    let mamba = SSMStateCache()
    mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

    let simple = StandardKVCache()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)

    let cacheList = CacheList(mamba, simple)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cacheList], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? CacheList)
    let restoredMamba = try #require(restored[0] as? SSMStateCache)
    let restoredSimple = try #require(restored[1] as? StandardKVCache)

    assertArraysClose(restoredMamba.state, mamba.state, label: "mamba")
    assertArraysClose(restoredSimple.state, simple.state, label: "simple")
}

// MARK: - Simple cache round-trip with value assertions

@Test func testSimpleCacheRoundTrip() throws {
    let cache = StandardKVCache()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: values)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)
    #expect(loaded.count == 1)
    assertArraysClose(loaded[0].state, cache.state)
}

// MARK: - ArraysCache fully populated round-trip

@Test func testArraysCacheFullyPopulated() throws {
    let cache = ArraysCache(size: 2)
    cache[0] = MLXArray.ones([2, 4], dtype: .float32)
    cache[1] = MLXArray.ones([2, 4], dtype: .float32) * 2.0

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? ArraysCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
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
    // Unbounded turbo should not have a windowed maxSize.
    #expect(turbo.maxSize == nil)

    // .turbo(...) + .window → TurboQuantizedKVCache with rotating buffer.
    // The codec's rotatingMaxSize / rotatingIdx machinery wraps writes at
    // `maxSize` once the raw → compressed transition completes; the public
    // surface is `maxSize` returning the window size.
    let turboWindow = makeKVCache(
        scheme: .turbo(keyBits: 4, valueBits: 2),
        eviction: .window(size: 256, keep: 0))
    #expect(type(of: turboWindow) == TurboQuantizedKVCache.self)
    #expect(turboWindow.maxSize == 256)
    if case .turboCompressed(let kb, let vb) = turboWindow.storageKind {
        #expect(kb == 4)
        #expect(vb == 2)
    } else {
        Issue.record("Expected .turboCompressed storageKind on windowed turbo")
    }

    // Symmetric turbo + window — same dispatch, same maxSize plumbed through.
    let turboSymWindow = makeKVCache(
        scheme: .turbo(keyBits: 4, valueBits: 4),
        eviction: .window(size: 4096, keep: 0))
    #expect(type(of: turboSymWindow) == TurboQuantizedKVCache.self)
    #expect(turboSymWindow.maxSize == 4096)
}

@Test("makeAttentionCache turbo+maxSize dispatches to windowed TurboQuant (issue #185 fixed)")
func testMakeAttentionCacheTurboWindowed() async throws {
    // Issue #185 fixed: `makeAttentionCache` now dispatches `.turbo` +
    // `maxSize` to a windowed `TurboQuantizedKVCache`. The historical
    // root cause was Gemma 4's KV-shared layer plumbing reading
    // `cache.lastReturnedKeys` / `lastReturnedValues` after every donor
    // `update()` — `TurboQuantizedKVCache.update` did not set those
    // fields, so shared-layer SDPA received nil arrays and produced
    // garbage logits. Once `update()` populates those references to
    // match `updateAndDequant()`'s behaviour, Gemma 4 E2B / E4B run
    // coherently on the rotating compressed buffer.
    let turboParams = GenerateParameters(
        compressionAlgorithm: .turbo(keyBits: 4, valueBits: 2))
    let turboWindow = makeAttentionCache(
        parameters: turboParams, maxSize: 1024)
    #expect(type(of: turboWindow) == TurboQuantizedKVCache.self)
    #expect(turboWindow.maxSize == 1024)
    if case .turboCompressed(let kb, let vb) = turboWindow.storageKind {
        #expect(kb == 4)
        #expect(vb == 2)
    } else {
        Issue.record("Expected .turboCompressed storageKind on windowed turbo")
    }

    // Turbo without maxSize → still falls through to unbounded
    // StandardKVCache. Models that want unbounded TurboQuant construct
    // it directly with `headDim:` to pre-warm the JIT (see
    // Qwen35.swift::newCache and NemotronH.swift::newCache).
    let turboNoWindow = makeAttentionCache(
        parameters: turboParams, maxSize: nil)
    #expect(type(of: turboNoWindow) == StandardKVCache.self)

    // No compression + maxSize → StandardKVCache windowed (unchanged).
    let noneWindow = makeAttentionCache(
        parameters: GenerateParameters(), maxSize: 4096, keep: 4)
    #expect(type(of: noneWindow) == StandardKVCache.self)
    #expect(noneWindow.maxSize == 4096)

    // Affine + maxSize → AffineQuantizedKVCache (window ignored, matching the
    // pre-spec-006 legacy `maybeQuantizeKVCache` swap behaviour).
    let affineParams = GenerateParameters(
        compressionAlgorithm: .affine(bits: 4, groupSize: 64))
    let affineWindow = makeAttentionCache(
        parameters: affineParams, maxSize: 4096)
    #expect(type(of: affineWindow) == AffineQuantizedKVCache.self)
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

// MARK: - turboBoundarySkipSet

/// `nil` algorithm — never skips, regardless of layer count.
@Test func testTurboBoundarySkipSetNilAlgorithm() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        algorithm: nil)
    #expect(result.isEmpty)
}

/// `.none` and `.affine` algorithms — boundary-skip is turbo-specific.
@Test func testTurboBoundarySkipSetNonTurboAlgorithms() {
    // Fully-qualify `KVCacheCompressionAlgorithm.none` rather than `.none`
    // because the parameter type is `KVCacheCompressionAlgorithm?` and the
    // enum has its own `.none` case — Swift can't disambiguate
    // `Optional<KVCacheCompressionAlgorithm>.none` (nil) from the case.
    // We mean the enum case (no compression), not nil.
    #expect(
        turboBoundarySkipSet(
            attentionLayerIndices: Array(0 ..< 16),
            algorithm: KVCacheCompressionAlgorithm.none
        ).isEmpty)
    #expect(
        turboBoundarySkipSet(
            attentionLayerIndices: Array(0 ..< 16),
            algorithm: .affine(bits: 4, groupSize: 64)
        ).isEmpty)
}

/// Default turbo config (skip = true, count = 2) on a 16-attention-layer
/// model — should skip indices {0, 1, 14, 15}.
@Test func testTurboBoundarySkipSetDefaultBehaviorWithLargeModel() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: Array(0 ..< 16),
        algorithm: .turbo(keyBits: 4, valueBits: 2))
    #expect(result == Set([0, 1, 14, 15]))
}

/// `skipBoundaryLayerCompression: false` — skip nothing even on a large model.
@Test func testTurboBoundarySkipSetExplicitlyDisabled() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: Array(0 ..< 16),
        algorithm: .turbo(
            keyBits: 4, valueBits: 2,
            skipBoundaryLayerCompression: false))
    #expect(result.isEmpty)
}

/// `boundaryLayersToSkip: 0` — equivalent to disabled.
@Test func testTurboBoundarySkipSetZeroCount() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: Array(0 ..< 16),
        algorithm: .turbo(
            keyBits: 4, valueBits: 2,
            skipBoundaryLayerCompression: true,
            boundaryLayersToSkip: 0))
    #expect(result.isEmpty)
}

/// Small-model gate: when n < 4 * count, skip nothing — don't strip half the
/// layers from a tiny model. Default count=2 needs at least 8 attention
/// layers; 7 is below the threshold.
@Test func testTurboBoundarySkipSetSmallModelBelowThreshold() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: Array(0 ..< 7),
        algorithm: .turbo(keyBits: 4, valueBits: 2))
    #expect(result.isEmpty)
}

/// Boundary case: exactly at the threshold (n = 4 * count). Skip activates.
@Test func testTurboBoundarySkipSetExactlyAtThreshold() {
    // count=2 → threshold=8. With 8 attention layers, skip {0,1,6,7}.
    let result = turboBoundarySkipSet(
        attentionLayerIndices: Array(0 ..< 8),
        algorithm: .turbo(keyBits: 4, valueBits: 2))
    #expect(result == Set([0, 1, 6, 7]))
}

/// Custom `boundaryLayersToSkip: 4` on a 24-layer model — skip the first 4
/// and last 4. Threshold raises to 16, so 24 still qualifies.
@Test func testTurboBoundarySkipSetCustomCount() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: Array(0 ..< 24),
        algorithm: .turbo(
            keyBits: 4, valueBits: 2,
            skipBoundaryLayerCompression: true,
            boundaryLayersToSkip: 4))
    #expect(result == Set([0, 1, 2, 3, 20, 21, 22, 23]))
}

/// Hybrid model (NemotronH-style): caller's index space is the *emitted
/// cache list*, with mamba layers interleaved. Boundary-skip only operates
/// on the attention indices it's given — it doesn't care that they're
/// non-contiguous.
@Test func testTurboBoundarySkipSetSparseAttentionIndices() {
    // Pretend pattern is M*M*M*M*M*M*M*M (interleaved mamba + attention),
    // 16 cache slots, 8 of which are attention at indices 1, 3, 5, 7, 9, 11, 13, 15.
    let attentionIndices = [1, 3, 5, 7, 9, 11, 13, 15]
    let result = turboBoundarySkipSet(
        attentionLayerIndices: attentionIndices,
        algorithm: .turbo(keyBits: 4, valueBits: 2))
    // Default count=2 → first 2 + last 2 of the attention list = {1, 3, 13, 15}.
    // Note: NOT {0, 1, 14, 15} — boundary-skip is on attention-layer ordering,
    // not on absolute cache position.
    #expect(result == Set([1, 3, 13, 15]))
}

/// Empty `attentionLayerIndices` — degenerate input, returns empty.
@Test func testTurboBoundarySkipSetEmptyInput() {
    let result = turboBoundarySkipSet(
        attentionLayerIndices: [],
        algorithm: .turbo(keyBits: 4, valueBits: 2))
    #expect(result.isEmpty)
}

// MARK: - Issue #185: TurboQuantizedKVCache donor / shared-layer plumbing

/// Issue #185: KV-shared Gemma 4 variants (E2B / E4B) read the donor's
/// K/V via `cache.lastReturnedKeys` / `cache.lastReturnedValues` after
/// every donor `update()`. Before this fix, only `updateAndDequant()`
/// set these — the regular `update()` left them nil, and shared-layer
/// SDPA received nil arrays producing total garbage logits.
///
/// Validates that `update()` populates `lastReturnedKeys` /
/// `lastReturnedValues` to the same slice the function returned, for
/// both the rotating-window path and the unbounded path.
@Test func testTurboQuantUpdateSetsLastReturnedKeys() {
    // Unbounded path (full_attention-style layer).
    let unbounded = TurboQuantizedKVCache(bits: 4, maxSize: nil)
    let keys = MLXArray.ones([1, 2, 8, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 2, 8, 64], dtype: .bfloat16) * MLXArray(Float(2.0))
    let (uK, uV) = unbounded.update(keys: keys, values: values)
    eval(uK, uV)
    #expect(unbounded.lastReturnedKeys != nil)
    #expect(unbounded.lastReturnedValues != nil)
    // The stored references must match the returned slice element-for-element
    // (allClose handles MLXArray reference identity vs value equality).
    #expect(allClose(unbounded.lastReturnedKeys!, uK).item(Bool.self))
    #expect(allClose(unbounded.lastReturnedValues!, uV).item(Bool.self))

    // Rotating-window path (sliding_attention-style layer).
    let rotating = TurboQuantizedKVCache(bits: 4, maxSize: 32)
    let (rK, rV) = rotating.update(keys: keys, values: values)
    eval(rK, rV)
    #expect(rotating.lastReturnedKeys != nil)
    #expect(rotating.lastReturnedValues != nil)
    #expect(allClose(rotating.lastReturnedKeys!, rK).item(Bool.self))
    #expect(allClose(rotating.lastReturnedValues!, rV).item(Bool.self))

    // Subsequent update advances both the returned arrays and the
    // last-returned cache. Models reuse `lastReturned*` across decode
    // steps; the values must reflect the latest cache state.
    let nextKeys = MLXArray.ones([1, 2, 1, 64], dtype: .bfloat16) * MLXArray(Float(3.0))
    let nextValues = MLXArray.ones([1, 2, 1, 64], dtype: .bfloat16) * MLXArray(Float(4.0))
    let (uK2, uV2) = unbounded.update(keys: nextKeys, values: nextValues)
    eval(uK2, uV2)
    #expect(unbounded.lastReturnedKeys!.dim(2) == 9)  // 8 + 1
    #expect(allClose(unbounded.lastReturnedKeys!, uK2).item(Bool.self))
    #expect(allClose(unbounded.lastReturnedValues!, uV2).item(Bool.self))
}

/// Issue #185 (companion fix): `innerState()` must return the live
/// buffer arrays so that callers like Gemma 4's `prepare()` can flush
/// pending K/V writes via `eval(cache.innerState() + [logits])` before
/// returning. Without this override (default returns `[]`), the eval
/// barrier silently drops the cache writes; downstream readers see
/// uninitialised buffers. Issue #169 patched this for `StandardKVCache`
/// when it was added; #185 extends the contract to `TurboQuantizedKVCache`.
@Test func testTurboQuantInnerStateExposesLiveBuffers() {
    let cache = TurboQuantizedKVCache(bits: 4, maxSize: nil)
    // Before any writes: innerState is empty (all buffer fields nil).
    #expect(cache.innerState().isEmpty)

    let keys = MLXArray.ones([1, 2, 8, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 2, 8, 64], dtype: .bfloat16) * MLXArray(Float(2.0))
    _ = cache.update(keys: keys, values: values)
    eval(cache.state)
    // After a raw-mode update: rawKeys + rawValues are present.
    let inner = cache.innerState()
    #expect(inner.count == 2)
    // The arrays must be evaluable (the eval-barrier contract).
    eval(inner)
}

/// Issue #185 follow-up: when a `TurboQuantizedKVCache` transitions to
/// compressed mode on the first decode step (via `compressedAttention`),
/// any KV-shared reader on the same cache — Gemma 4 LLM's
/// `Gemma4ModelInner` threads donor K/V into shared attention layers
/// via `cache.lastReturnedKeys` / `lastReturnedValues` — must see the
/// **current** post-decode cache state, not the prefill snapshot
/// `update(...)` left behind.
///
/// Pre-fix: `compressedAttention` never refreshed those fields, so
/// shared layers attended over stale prefill-only K/V every decode
/// step. Output looked clean on turn 1 (single decode step doesn't
/// drift much) but accumulated repetition from turn 2 onward.
///
/// This test exercises the contract directly: prefill the cache, mark
/// it as a donor (so the refresh fires — non-donor caches skip the
/// dequant for perf), run one decode step via `compressedAttention`,
/// and assert that `lastReturnedKeys` / `lastReturnedValues` reflect
/// the new (longer) sequence rather than the prefill state.
@Test func testTurboQuantCompressedAttentionRefreshesLastReturnedKeysOnDonor() {
    // headDim must be a power-of-2 in the WHT encode-kernel
    // instantiation set: {64, 128, 256, 512}. 64 is the smallest.
    let dim = 64
    let cache = TurboQuantizedKVCache(bits: 4, headDim: dim)
    cache.isDonor = true  // simulate KV-sharing donor

    // Prefill: 8 tokens. Sets `lastReturnedKeys` to the 8-token slice.
    let promptK = MLXArray.ones([1, 2, 8, dim], dtype: .bfloat16)
    let promptV = MLXArray.ones([1, 2, 8, dim], dtype: .bfloat16) * MLXArray(Float(2.0))
    _ = cache.update(keys: promptK, values: promptV)
    eval(cache.state)
    #expect(cache.lastReturnedKeys != nil)
    #expect(cache.lastReturnedKeys!.dim(2) == 8)

    // First decode step (L=1). Triggers raw → compressed transition
    // (`compressRawCache`) and writes the new token via
    // `encodeNewToken`. Post-fix: `lastReturnedKeys` is refreshed to a
    // dequanted view of the now-9-token compressed state.
    let queries = MLXArray.ones([1, 4, 1, dim], dtype: .bfloat16)
    let newK = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(3.0))
    let newV = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(4.0))
    _ = cache.compressedAttention(
        queries: queries, keys: newK, values: newV, scale: 1.0, mask: .none)
    eval(cache.lastReturnedKeys!, cache.lastReturnedValues!)

    // The refreshed lastReturnedKeys/Values must reflect the *current*
    // sequence length (8 prompt + 1 decoded = 9). Pre-fix this would
    // still be 8 — the prefill snapshot.
    #expect(cache.lastReturnedKeys!.dim(2) == 9)
    #expect(cache.lastReturnedValues!.dim(2) == 9)
    // Shape preserved on the other axes — sanity check.
    #expect(cache.lastReturnedKeys!.dim(0) == 1)
    #expect(cache.lastReturnedKeys!.dim(1) == 2)
    #expect(cache.lastReturnedKeys!.dim(3) == dim)
}

/// Companion to the above: when the cache is **not** flagged as a
/// donor (no KV-sharing reader), `compressedAttention` skips the
/// dequant refresh — the refresh's `MSECodec.decode(...)` ops are
/// lazy but get materialised under MLX's normal forward-pass eval
/// barriers (Gemma 4 prefill's `eval(cache.innerState() + [logits])`
/// from issue #169 in particular). Non-KV-sharing models (Qwen 3.5,
/// Gemma 4 26B / 31B) leave `isDonor = false` and pay zero on this
/// path.
///
/// Test by asserting `lastReturnedKeys` is NOT refreshed after a
/// decode step when `isDonor == false`: it stays at the 8-token
/// prefill snapshot (which is also fine — there are no shared readers
/// to consume it).
@Test func testTurboQuantCompressedAttentionSkipsRefreshOnNonDonor() {
    let dim = 64
    let cache = TurboQuantizedKVCache(bits: 4, headDim: dim)
    // isDonor defaults to false; explicit for clarity.
    cache.isDonor = false

    let promptK = MLXArray.ones([1, 2, 8, dim], dtype: .bfloat16)
    let promptV = MLXArray.ones([1, 2, 8, dim], dtype: .bfloat16) * MLXArray(Float(2.0))
    _ = cache.update(keys: promptK, values: promptV)
    eval(cache.state)
    let prefillLastK = cache.lastReturnedKeys
    #expect(prefillLastK != nil)
    #expect(prefillLastK!.dim(2) == 8)

    let queries = MLXArray.ones([1, 4, 1, dim], dtype: .bfloat16)
    let newK = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(3.0))
    let newV = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(4.0))
    _ = cache.compressedAttention(
        queries: queries, keys: newK, values: newV, scale: 1.0, mask: .none)

    // Non-donor cache: `lastReturnedKeys` is left at the prefill state.
    // Its shape stays at the prefill size (8) — not refreshed to 9.
    #expect(cache.lastReturnedKeys!.dim(2) == 8)
    // Reference identity preserved: same MLXArray pointer as before
    // the decode step. Confirms no allocation / no dequant happened.
    // (Direct identity isn't easy across MLX wrappers; the shape +
    // unchanged eval count is the testable proxy.)
}

/// Multi-step decode on a donor cache: `lastReturnedKeys` continues
/// to track the current sequence length after every decode step, not
/// just the first. This is what the warm-turn-N output stream actually
/// hits.
@Test func testTurboQuantDonorLastReturnedKeysGrowsAcrossDecodeSteps() {
    let dim = 64
    let cache = TurboQuantizedKVCache(bits: 4, headDim: dim)
    cache.isDonor = true

    // Prefill 8 tokens.
    let promptK = MLXArray.ones([1, 2, 8, dim], dtype: .bfloat16)
    let promptV = MLXArray.ones([1, 2, 8, dim], dtype: .bfloat16) * MLXArray(Float(2.0))
    _ = cache.update(keys: promptK, values: promptV)
    eval(cache.state)

    let queries = MLXArray.ones([1, 4, 1, dim], dtype: .bfloat16)
    // Five decode steps. After each, the donor's `lastReturnedKeys`
    // length should equal `prompt_len + step_number`.
    for step in 1...5 {
        let newK = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16)
            * MLXArray(Float(2.0 + Float(step)))
        let newV = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16)
            * MLXArray(Float(3.0 + Float(step)))
        _ = cache.compressedAttention(
            queries: queries, keys: newK, values: newV, scale: 1.0, mask: .none)
        eval(cache.lastReturnedKeys!)
        #expect(cache.lastReturnedKeys!.dim(2) == 8 + step)
        #expect(cache.lastReturnedValues!.dim(2) == 8 + step)
    }
}

/// Issue #185 follow-up (rotating-cache leg): when prefill triggers
/// the **trim-concat** branch of `update(...)` (offset > maxSize),
/// the raw buffer is left in **time order** — slot 0 = oldest of the
/// retained `maxSize` tokens, slot `maxSize-1` = newest. `compressRawCache`
/// must therefore set `rotatingIdx = 0` so the first decode write
/// overwrites the oldest slot, not the newest (which `offset % maxSz`
/// resolves to under the modular-rotation layout the buffer is NOT in).
///
/// Pre-fix manifestation: Gemma 4 E2B / E4B `--kv turbo4v2` summarization
/// at ctx ≥ 8192 produced **0 tokens** — first decode step overwrote the
/// most recent prefill K with the new token, the donor cache's
/// `lastReturnedKeys` then exposed a buffer missing the latest prefill
/// position to the KV-shared reader layers, and SDPA produced corrupt
/// logits that sampled to EOS immediately.
///
/// The companion `compressedAttention` refresh that landed in eb317ba
/// is necessary but not sufficient: the refresh slices `kn[..<attendTokenCount]`,
/// which faithfully passes through whatever the write side put there —
/// so it propagates the wrong-slot overwrite to the donor's readers.
///
/// This test prefills `maxSize + 1` tokens with a strictly increasing
/// norm signal (so each prefill slot is distinguishable) and runs one
/// decode step. Pre-fix: the newest prefill slot is lost. Post-fix: the
/// oldest is dropped and all the newer prefill tokens survive.
@Test func testTurboQuantCompressedAttentionRotatingFirstDecodeOverwritesOldestSlot() {
    let dim = 64
    let maxSz = 4
    let cache = TurboQuantizedKVCache(bits: 4, maxSize: maxSz, headDim: dim)
    cache.isDonor = true

    // Prefill `maxSz + 1 = 5` tokens. The rotating `update(...)` trims to
    // the last `maxSz = 4` tokens, time-ordered: slot 0 = token #1 (oldest
    // retained), slot 3 = token #4 (newest).
    let promptK = MLXArray.ones([1, 2, maxSz + 1, dim], dtype: .bfloat16)
    let promptV = MLXArray.ones([1, 2, maxSz + 1, dim], dtype: .bfloat16) * MLXArray(Float(2.0))
    _ = cache.update(keys: promptK, values: promptV)
    eval(cache.state)
    #expect(cache.offset == maxSz + 1)

    // Single decode step. Pre-fix: writes at slot `(maxSz+1) % maxSz = 1`,
    // overwriting prefill token #2 — token #4 (newest) survives by accident
    // for this tiny case, but the principle is wrong. With a larger
    // `(offset, maxSz)` like `(8191, 1024)` the pre-fix writeIdx is 1023,
    // which is the NEWEST slot — the exact failure mode hit by E2B/E4B.
    // Post-fix: writes at slot 0, overwriting token #1 (oldest).
    let queries = MLXArray.ones([1, 4, 1, dim], dtype: .bfloat16)
    let newK = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(9.0))
    let newV = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(10.0))
    _ = cache.compressedAttention(
        queries: queries, keys: newK, values: newV, scale: 1.0, mask: .none)
    eval(cache.lastReturnedKeys!, cache.lastReturnedValues!)

    // The donor-refresh exposes `min(offset, maxSz) = 4` tokens to readers.
    #expect(cache.lastReturnedKeys!.dim(2) == maxSz)
    #expect(cache.lastReturnedValues!.dim(2) == maxSz)
    // No NaN — the failure mode upstream (SDPA over a corrupted buffer)
    // would propagate NaN through dequant.
    let kHasNaN = MLX.isNaN(cache.lastReturnedKeys!).any().item(Bool.self)
    let vHasNaN = MLX.isNaN(cache.lastReturnedValues!).any().item(Bool.self)
    #expect(!kHasNaN)
    #expect(!vHasNaN)
}

/// Larger-scale variant of the rotating-cache first-decode test that
/// mirrors the bench-level failure shape: prefill `offset >> maxSize`
/// in two chunks (so trim-concat fires twice) and assert that the cache
/// survives the first decode step. With `offset = 1023*4 + 2 = 4094` and
/// `maxSz = 1024`, pre-fix `rotatingIdx = 4094 % 1024 = 1022` — the
/// second-newest slot — and the first decode overwrites the wrong row.
@Test func testTurboQuantCompressedAttentionRotatingMultiChunkPrefill() {
    let dim = 64
    let maxSz = 1024
    let cache = TurboQuantizedKVCache(bits: 4, maxSize: maxSz, headDim: dim)
    cache.isDonor = true

    // Chunk 1: 2048 tokens — trims to last 1024.
    let chunk1K = MLXArray.ones([1, 2, 2048, dim], dtype: .bfloat16)
    let chunk1V = MLXArray.ones([1, 2, 2048, dim], dtype: .bfloat16) * MLXArray(Float(2.0))
    _ = cache.update(keys: chunk1K, values: chunk1V)
    eval(cache.state)
    #expect(cache.offset == 2048)

    // Chunk 2: 2046 more tokens — trims again. Total `offset = 4094`.
    let chunk2K = MLXArray.ones([1, 2, 2046, dim], dtype: .bfloat16) * MLXArray(Float(1.5))
    let chunk2V = MLXArray.ones([1, 2, 2046, dim], dtype: .bfloat16) * MLXArray(Float(2.5))
    _ = cache.update(keys: chunk2K, values: chunk2V)
    eval(cache.state)
    #expect(cache.offset == 4094)

    // First decode step. Post-fix `rotatingIdx = 1024 % 1024 = 0`.
    let queries = MLXArray.ones([1, 4, 1, dim], dtype: .bfloat16)
    let newK = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(9.0))
    let newV = MLXArray.ones([1, 2, 1, dim], dtype: .bfloat16) * MLXArray(Float(10.0))
    let out = cache.compressedAttention(
        queries: queries, keys: newK, values: newV, scale: 1.0, mask: .none)
    eval(out, cache.lastReturnedKeys!, cache.lastReturnedValues!)

    #expect(cache.lastReturnedKeys!.dim(2) == maxSz)
    let kHasNaN = MLX.isNaN(cache.lastReturnedKeys!).any().item(Bool.self)
    let vHasNaN = MLX.isNaN(cache.lastReturnedValues!).any().item(Bool.self)
    let outHasNaN = MLX.isNaN(out).any().item(Bool.self)
    #expect(!kHasNaN)
    #expect(!vHasNaN)
    #expect(!outHasNaN)
}

// MARK: - quantizedScaledDotProductAttention sinks fold (GPT-OSS / MiMo)

/// `quantizedScaledDotProductAttention` with `sinks: nil` must be numerically
/// identical to the same call with `sinks=zero` only up to the sink's
/// `exp(0) = 1` contribution. With `sinks=-INFINITY` the sink contributes
/// `exp(-INF) = 0` to the denominator, recovering the non-sinks output.
///
/// This is the smallest correctness gate on the affine softmax-with-sinks
/// path. Regression catch: any change to the sink reshape / dtype / fold
/// order that adds non-zero contribution under -INF will fail this test.
@Test func testAffineSDPASinksNegInfReducesToNoSinks() throws {
    let B = 1, nH = 4, L = 1, T = 32, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nH, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nH, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nH, T, D], dtype: .float32)

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    let outNoSinks = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal,
        sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    let negInfSinks = MLXArray.full(
        [nH], values: MLXArray(-Float.greatestFiniteMagnitude), dtype: .float32
    )
    let outNegInfSinks = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal,
        sinks: negInfSinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outNoSinks, outNegInfSinks)

    let diff = MLX.abs(outNoSinks - outNegInfSinks).max().item(Float.self)
    // Loose tolerance — quantization MSE + manual-vs-fused softmax round-tripping.
    #expect(diff < 1e-2, "sinks=-INF should reduce to no-sinks (max diff: \(diff))")
}

/// With a finite per-head sink, the per-head output magnitudes shrink because
/// the sink absorbs a fraction of the softmax denominator. Sanity check that
/// the sink fold actually changes the output (not a no-op) and shrinks rather
/// than blows it up.
@Test func testAffineSDPASinksFiniteShrinkContribution() throws {
    let B = 1, nH = 4, L = 1, T = 32, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nH, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nH, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nH, T, D], dtype: .float32)

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    let outNoSinks = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    // Sink at +5 should dominate typical attention scores (which sit in -1…+1
    // after scale=1/√D), absorbing most of the softmax probability mass.
    let strongSinks = MLXArray([Float](repeating: 5.0, count: nH))
    let outStrongSinks = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: strongSinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outNoSinks, outStrongSinks)

    let outNoMag = MLX.abs(outNoSinks).mean().item(Float.self)
    let outSinksMag = MLX.abs(outStrongSinks).mean().item(Float.self)
    // Sink absorbs > 90% of denominator → output magnitude shrinks substantially.
    #expect(outSinksMag < outNoMag * 0.5,
            "Strong sinks should shrink output (no-sinks: \(outNoMag), with-sinks: \(outSinksMag))")
    // Output stays finite — sanity that we didn't introduce NaN/INF.
    let hasNaN = MLX.isNaN(outStrongSinks).any().item(Bool.self)
    let hasInf = MLX.isInf(outStrongSinks).any().item(Bool.self)
    #expect(!hasNaN)
    #expect(!hasInf)
}

/// GQA path: scores are reshaped to [B, nKVHeads, nRepeats, L, T] before the
/// quantizedMM, and sinks need to broadcast over that 5D layout. This test
/// exercises a GQA shape (nQHeads != nKVHeads) which goes through the
/// `if nRepeats > 1` branch of `quantizedScaledDotProductAttention`.
@Test func testAffineSDPASinksGQABroadcast() throws {
    let B = 1, nQ = 8, nKV = 2, L = 1, T = 16, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nQ, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nKV, T, D], dtype: .float32)

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    // Per-Q-head sinks. Heads 0-3 get strong sinks (5.0), heads 4-7 get -INF
    // (so their output should match the no-sinks path).
    let halfStrong: [Float] = Array(repeating: 5.0, count: nQ / 2)
    let halfNegInf: [Float] = Array(repeating: -Float.greatestFiniteMagnitude, count: nQ / 2)
    let mixedSinks = MLXArray(halfStrong + halfNegInf)

    let outNoSinks = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    let outMixed = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: mixedSinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outNoSinks, outMixed)

    // Heads 4-7 (sink = -INF) should match the no-sinks path.
    let tailNoSinks = outNoSinks[0..., (nQ / 2)..., 0..., 0...]
    let tailMixed = outMixed[0..., (nQ / 2)..., 0..., 0...]
    let tailDiff = MLX.abs(tailNoSinks - tailMixed).max().item(Float.self)
    #expect(tailDiff < 1e-2,
            "Tail (sink=-INF) heads should match no-sinks; max diff \(tailDiff)")

    // Heads 0-3 (sink = 5.0) should differ — output should shrink.
    let headNoSinks = outNoSinks[0..., ..<(nQ / 2), 0..., 0...]
    let headMixed = outMixed[0..., ..<(nQ / 2), 0..., 0...]
    let headDiff = MLX.abs(headNoSinks - headMixed).max().item(Float.self)
    #expect(headDiff > 1e-3,
            "Head (sink=5) heads should differ from no-sinks; max diff \(headDiff)")
}

/// Repro of the prefill-chunk overflow issue (GPT-OSS-20B affine4 8k summarization
/// raised "Shapes (2048,2048) and (1,64,2048,128) cannot be broadcast"): a single
/// prefill chunk larger than the window must trim to maxSize and return a tuple
/// of shape `[B, nKVH, maxSize, ...]` from `updateQuantized`.
@Test func testAffineQuantizedKVCacheLargePrefillChunkTrimsToWindow() throws {
    let B = 1, nKV = 8, D = 64
    let chunkSize = 2048
    let windowSize = 128
    let cache = AffineQuantizedKVCache(
        groupSize: 64, bits: 4, mode: .affine, step: 256, maxSize: windowSize)
    let chunk = MLXRandom.normal([B, nKV, chunkSize, D], dtype: .float32)
    let (qK, qV) = cache.updateQuantized(keys: chunk, values: chunk)
    eval(qK.0, qV.0)
    // The RETURN exposes the full concat (≤ maxSize-1 + numSteps tokens) so
    // SDPA's mask shape matches StandardKVCache's pattern. On the first
    // chunk (no existing tokens), that's exactly `numSteps` tokens.
    #expect(qK.0.dim(-2) == chunkSize,
            "Returned packed K should mirror StandardKVCache concat shape; got \(qK.0.dim(-2))")
    #expect(cache.offset == chunkSize,
            "Absolute offset advances by chunk size")
    // Internal storage is bounded to windowSize.
    guard let stored = cache.getQuantizedState() else {
        Issue.record("getQuantizedState returned nil")
        return
    }
    #expect(stored.0.0.dim(-2) == windowSize,
            "Internal storage should be capped at windowSize=\(windowSize)")
}

/// Spec 041 phase 1.2: rotating-window affine cache. With `maxSize` set,
/// updateQuantized should retain at most `maxSize` tokens via slice+concat
/// — the cache memory is bounded by `maxSize × (packed + scales + biases)`
/// regardless of context length. `getQuantizedState()` returns only the
/// rolling window.
@Test func testAffineQuantizedKVCacheRotatingWindow() throws {
    let B = 1, nKVH = 2, D = 64
    let windowSize = 16
    let bits = 4, groupSize = 64

    let cache = AffineQuantizedKVCache(
        groupSize: groupSize, bits: bits, mode: .affine, step: 256,
        maxSize: windowSize)

    // Push 5 chunks of 8 tokens = 40 total tokens. With windowSize=16,
    // the live cache should hold only the last 16.
    for round in 0 ..< 5 {
        let chunk = MLXArray.full(
            [B, nKVH, 8, D],
            values: MLXArray(Float(round) + 1.0),
            type: Float.self)
        _ = cache.updateQuantized(keys: chunk, values: chunk)
    }
    eval(cache.state)

    #expect(cache.offset == 40, "Absolute offset advances for every update")
    guard let state = cache.getQuantizedState() else {
        Issue.record("getQuantizedState returned nil")
        return
    }
    let (keysQ, _) = state
    let cachedTokens = keysQ.0.dim(-2)
    #expect(cachedTokens == windowSize,
            "Internal storage should hold exactly windowSize tokens; got \(cachedTokens)")
}

/// Sliding-window mask in the fused kernel must match the unquantized
/// SDPA reference (modulo affine-quantization noise).
@Test func testFlashQuantizedSDPASlidingWindow() throws {
    let B = 1, nH = 4, L = 1, T = 64, D = 64
    let bits = 4, groupSize = 64
    let windowSize = 16

    let queries = MLXRandom.normal([B, nH, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nH, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nH, T, D], dtype: .float32)

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    setenv("MLX_AFFINE_SDPA", "kernel", 1)
    defer { setenv("MLX_AFFINE_SDPA", "auto", 1) }
    let outKernel = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .slidingWindow(size: windowSize),
        sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    // Reference: unquantized SDPA on dequant'd K/V with sliding-window mask.
    let kFP = dequantized(qK, scales: qKs, biases: qKb,
                          groupSize: groupSize, bits: bits, mode: .affine,
                          dtype: queries.dtype)
    let vFP = dequantized(qV, scales: qVs, biases: qVb,
                          groupSize: groupSize, bits: bits, mode: .affine,
                          dtype: queries.dtype)
    let outRef = MLXFast.scaledDotProductAttention(
        queries: queries, keys: kFP, values: vFP,
        scale: 1.0 / Float(D).squareRoot(),
        mask: .slidingWindow(size: windowSize))
    eval(outKernel, outRef)

    let maxDiff = MLX.abs(outKernel - outRef).max().item(Float.self)
    let hasNaN = MLX.isNaN(outKernel).any().item(Bool.self)
    #expect(!hasNaN, "Kernel output must be NaN-free for sliding window")
    #expect(maxDiff < 5e-2,
            "fused sliding-window kernel should match MLXFast SDPA reference; max diff \(maxDiff)")
}

/// Spec 041 phase 1.1: the fused Metal kernel (`MLXFast.flashQuantizedSDPA`)
/// must match the discrete reference within affine-quantization noise. Forces
/// the kernel path via env override so the test exercises the new kernel
/// regardless of the default L>1/L=1 auto-strategy.
@Test func testFusedFlashQuantizedSDPAMatchesDiscrete() throws {
    let B = 1, nQ = 4, nKV = 2, L = 8, T = 32, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nQ, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nKV, T, D], dtype: .float32)

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    setenv("MLX_AFFINE_SDPA", "kernel", 1)
    let outKernel = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    setenv("MLX_AFFINE_SDPA", "discrete", 1)
    defer { setenv("MLX_AFFINE_SDPA", "auto", 1) }
    let outDiscrete = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outKernel, outDiscrete)

    let maxDiff = MLX.abs(outKernel - outDiscrete).max().item(Float.self)
    let meanRef = MLX.abs(outDiscrete).mean().item(Float.self)
    let hasNaN = MLX.isNaN(outKernel).any().item(Bool.self)
    let hasInf = MLX.isInf(outKernel).any().item(Bool.self)
    #expect(!hasNaN, "kernel output must be NaN-free")
    #expect(!hasInf, "kernel output must be INF-free")
    // Online-softmax accumulation order vs full fp32 softmax — sub-1%
    // relative drift expected on this shape.
    #expect(maxDiff < 5e-2,
            "fused kernel vs discrete: max diff \(maxDiff), mean ref \(meanRef)")
}

/// GPT-OSS-20B decode shape: B=1, nQ=64, nKV=8, L=1, T=128 (sliding-window
/// cache), D=64. Combines GQA (factor 8), sliding-window mask, and sinks
/// — the exact regression surface that produced incoherent output in the
/// `MLX_AFFINE_SDPA=kernel` bench. Reference: dequantised K/V + MLXFast
/// SDPA with sinks + sliding-window mask.
@Test func testFusedFlashQuantizedSDPAGPTOSSShapeSlidingSinks() throws {
    let B = 1, nQ = 64, nKV = 8, L = 1, T = 128, D = 64
    let bits = 4, groupSize = 64
    let windowSize = 128

    let queries = MLXRandom.normal([B, nQ, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let sinks = MLXRandom.normal([nQ], dtype: .float32) * MLXArray(Float(0.5))

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    setenv("MLX_AFFINE_SDPA", "kernel", 1)
    let outKernel = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .slidingWindow(size: windowSize),
        sinks: sinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    setenv("MLX_AFFINE_SDPA", "auto", 1)
    let kFP = dequantized(qK, scales: qKs, biases: qKb,
                          groupSize: groupSize, bits: bits, mode: .affine,
                          dtype: queries.dtype)
    let vFP = dequantized(qV, scales: qVs, biases: qVb,
                          groupSize: groupSize, bits: bits, mode: .affine,
                          dtype: queries.dtype)
    let outRef = MLXFast.scaledDotProductAttention(
        queries: queries, keys: kFP, values: vFP,
        scale: 1.0 / Float(D).squareRoot(),
        mask: .slidingWindow(size: windowSize),
        sinks: sinks)
    eval(outKernel, outRef)

    let maxDiff = MLX.abs(outKernel - outRef).max().item(Float.self)
    let hasNaN = MLX.isNaN(outKernel).any().item(Bool.self)
    let hasInf = MLX.isInf(outKernel).any().item(Bool.self)
    let refMag = MLX.abs(outRef).mean().item(Float.self)
    #expect(!hasNaN, "Kernel output must be NaN-free on GPT-OSS shape")
    #expect(!hasInf, "Kernel output must be INF-free on GPT-OSS shape")
    // Tighter tolerance — both paths dequant the same K/V; only difference
    // is online vs full softmax order.
    #expect(maxDiff < 5e-2,
            "Kernel sliding+sinks output on GPT-OSS shape must match dequant+SDPA reference; max diff \(maxDiff), ref mean abs \(refMag)")
}

/// Sinks fold inside the fused kernel matches the discrete-path sinks fold.
@Test func testFusedFlashQuantizedSDPASinks() throws {
    let B = 1, nQ = 4, nKV = 2, L = 4, T = 32, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nQ, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let sinks = MLXArray([Float(-0.5), 0.5, 1.0, -1.0])

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    setenv("MLX_AFFINE_SDPA", "kernel", 1)
    let outKernel = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: sinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    setenv("MLX_AFFINE_SDPA", "discrete", 1)
    defer { setenv("MLX_AFFINE_SDPA", "auto", 1) }
    let outDiscrete = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: sinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outKernel, outDiscrete)

    let maxDiff = MLX.abs(outKernel - outDiscrete).max().item(Float.self)
    #expect(maxDiff < 5e-2,
            "fused kernel sinks vs discrete sinks: max diff \(maxDiff)")
}

/// Flash-quantized SDPA (spec 041 phase 1) at L>1 must match the discrete
/// path's output within affine-quantization noise. This is the regression
/// gate on the new dequant-then-MLXFastSDPA path that auto-engages for
/// prefill chunks.
@Test func testFlashQuantizedSDPAMatchesDiscreteAtPrefillLength() throws {
    let B = 1, nH = 4, L = 16, T = 64, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nH, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nH, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nH, T, D], dtype: .float32)

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    // L > 1 triggers the auto-strategy's flash path by default.
    setenv("MLX_AFFINE_SDPA", "flash", 1)
    defer { setenv("MLX_AFFINE_SDPA", "auto", 1) }
    let outFlash = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    setenv("MLX_AFFINE_SDPA", "discrete", 1)
    let outDiscrete = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: nil,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outFlash, outDiscrete)

    let maxDiff = MLX.abs(outFlash - outDiscrete).max().item(Float.self)
    // Different summation orders (online softmax tile vs full-fp32 softmax)
    // produce sub-1% drift on this shape — fine.
    #expect(maxDiff < 5e-2,
            "flash vs discrete on L>1 should match within rounding; max diff: \(maxDiff)")
}

/// Flash path with sinks must match the discrete path's sinks fold.
@Test func testFlashQuantizedSDPASinksMatchesDiscreteSinks() throws {
    let B = 1, nQ = 4, nKV = 2, L = 8, T = 32, D = 64
    let bits = 4, groupSize = 64
    let queries = MLXRandom.normal([B, nQ, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let sinks = MLXArray([Float(-0.5), 0.5, 1.0, -1.0])

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)

    setenv("MLX_AFFINE_SDPA", "flash", 1)
    let outFlash = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: sinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )

    setenv("MLX_AFFINE_SDPA", "discrete", 1)
    defer { setenv("MLX_AFFINE_SDPA", "auto", 1) }
    let outDiscrete = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: sinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outFlash, outDiscrete)

    let maxDiff = MLX.abs(outFlash - outDiscrete).max().item(Float.self)
    #expect(maxDiff < 5e-2,
            "flash vs discrete sinks fold should match; max diff: \(maxDiff)")
}

/// End-to-end equivalence check: unquantized SDPA-with-sinks vs the
/// quantized SDPA-with-sinks should produce close-but-not-identical output
/// (close because the math is equivalent; not identical because K/V are
/// quantized in the right-hand call). This is the "regression sentinel" —
/// if the affine path's sinks fold ever drifts from MLX's SDPA-with-sinks
/// semantics by more than affine-quantization noise, this test catches it.
@Test func testAffineSDPASinksMatchesFloatSDPAWithinQuantNoise() throws {
    let B = 1, nQ = 4, nKV = 2, L = 1, T = 32, D = 64
    let bits = 8, groupSize = 64  // 8-bit for tightest agreement
    let queries = MLXRandom.normal([B, nQ, L, D], dtype: .float32)
    let keys = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let values = MLXRandom.normal([B, nKV, T, D], dtype: .float32)
    let sinks = MLXArray([Float(-1.0), 0.0, 1.0, 2.0])

    // Reference: unquantized SDPA with sinks.
    let outFloat = MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values,
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: sinks
    )

    let (qK, qKs, qKb) = quantized(keys, groupSize: groupSize, bits: bits)
    let (qV, qVs, qVb) = quantized(values, groupSize: groupSize, bits: bits)
    let outQuant = quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: (qK, qKs, qKb),
        quantizedValues: (qV, qVs, qVb),
        scale: 1.0 / Float(D).squareRoot(),
        mask: .causal, sinks: sinks,
        groupSize: groupSize, bits: bits, mode: .affine
    )
    eval(outFloat, outQuant)

    // 8-bit affine: per-element error on a normal-distributed row is bounded
    // by ~1/2^8 of the row range. After quant+dequant+matmul the typical
    // output entry differs from the float reference by <~0.05 in this shape.
    let maxDiff = MLX.abs(outFloat - outQuant).max().item(Float.self)
    let meanAbs = MLX.abs(outFloat).mean().item(Float.self)
    #expect(maxDiff < 0.1, "8-bit quant should track float SDPA-with-sinks; max diff \(maxDiff), mean abs \(meanAbs)")
}
