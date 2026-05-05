import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
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
        case let quantized as QuantizedKVCache:
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
    let cache = MambaCache()
    let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
    cache[0] = a
    cache[1] = b

    let url = tempURL()
    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? MambaCache)
    #expect(restored.slotCount == 2)
    assertArraysClose(restored.state, cache.state)
}

// MARK: - CacheList with KV caches

@Test func testCacheListKVCaches() throws {
    let simple = KVCacheSimple()
    let rotating = RotatingKVCache(maxSize: 32)

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
    let child0 = try #require(restored[0] as? KVCacheSimple)
    let child1 = try #require(restored[1] as? RotatingKVCache)

    assertArraysClose(child0.state, simple.state, label: "child0")
    assertArraysClose(child1.state, rotating.state, label: "child1")
    #expect(child1.metaState == rotating.metaState)
}

// MARK: - CacheList with hybrid (MambaCache + KVCacheSimple)

@Test func testCacheListHybrid() throws {
    let mamba = MambaCache()
    mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

    let simple = KVCacheSimple()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)

    let cacheList = CacheList(mamba, simple)

    let url = tempURL()
    try savePromptCache(url: url, cache: [cacheList], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    #expect(loaded.count == 1)
    let restored = try #require(loaded[0] as? CacheList)
    let restoredMamba = try #require(restored[0] as? MambaCache)
    let restoredSimple = try #require(restored[1] as? KVCacheSimple)

    assertArraysClose(restoredMamba.state, mamba.state, label: "mamba")
    assertArraysClose(restoredSimple.state, simple.state, label: "simple")
}

// MARK: - Simple cache round-trip with value assertions

@Test func testSimpleCacheRoundTrip() throws {
    let cache = KVCacheSimple()
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
    case let quantized as QuantizedKVCache:
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
    case let quantized as QuantizedKVCache:
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

// MARK: - RotatingKVCache.reserve regression coverage

@Test("RotatingKVCache.reserve preallocates the buffer to the hinted size")
func testRotatingKVCacheReservePreallocates() async throws {
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
    cache.reserve(800)

    let keys = MLXArray.ones([1, 8, 64, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    // Buffer should be 800 (the hint, capped above step), not 256 (default step).
    #expect(cache.innerState()[0].shape == [1, 8, 800, 128])
}

@Test("RotatingKVCache.reserve is no-op after first write")
func testRotatingKVCacheReserveIsIdempotentAfterWrite() async throws {
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
    let keys = MLXArray.ones([1, 8, 64, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    // Cache already populated with step-sized buffer; reserve should not retroactively grow it.
    let originalSize = cache.innerState()[0].dim(2)
    cache.reserve(2048)
    #expect(cache.innerState()[0].dim(2) == originalSize)
}

@Test("RotatingKVCache.reserve clamps to maxCacheSize")
func testRotatingKVCacheReserveClampsToMax() async throws {
    let cache = RotatingKVCache(maxSize: 256, step: 64)
    cache.reserve(10000)  // way over maxCacheSize

    let keys = MLXArray.ones([1, 8, 32, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    #expect(cache.innerState()[0].dim(2) == 256)
}

@Test("RotatingKVCache.reserve floor: at least step")
func testRotatingKVCacheReserveFloorIsStep() async throws {
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
    cache.reserve(100)  // smaller than step

    let keys = MLXArray.ones([1, 8, 32, 128], dtype: .bfloat16)
    _ = cache.update(keys: keys, values: keys)

    // Buffer should be at least step (256) even though hint was 100.
    #expect(cache.innerState()[0].dim(2) >= 256)
}

@Test("RotatingKVCache.reserve buffer fits multi-token writes within hint")
func testRotatingKVCacheReserveFitsPrefillChunks() async throws {
    // Simulates the typical iterator wiring: hint = prompt + maxTokens, then
    // multi-token prefill writes the prompt in chunks. Each chunk must land
    // in the pre-allocated buffer without triggering a grow.
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
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

@Test("RotatingKVCache.reserve grows past hint when workload exceeds it")
func testRotatingKVCacheReserveGrowsOnOverflow() async throws {
    // If the actual workload exceeds the hinted size, the cache should fall
    // back to step-based growth — the hint isn't a hard cap, just a
    // first-allocation size.
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
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

// MARK: - RotatingKVCache under-utilisation + rotation regression coverage
//
// Independent of `reserve(_:)` — covers the path the existing 11 tests in
// KVCacheTests.swift never exercised: step-incremental growth without
// rotation, and the rotation/wrap behaviour past `maxCacheSize`. Locks the
// existing semantics so refactors (incl. the spec 006 consolidation into
// `StandardKVCache`) can prove byte-identical behaviour.

@Test("RotatingKVCache step-incremental growth (under-util, no reserve)")
func testRotatingKVCacheUnderUtilisationStepGrowth() async throws {
    // Push 1024 single-token writes. With default step=256 and maxSize=4096
    // the buffer should grow 256 → 512 → 768 → 1024, never reaching 4096.
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    for _ in 0 ..< 1024 {
        _ = cache.update(keys: token, values: token)
    }
    #expect(cache.offset == 1024)
    // Buffer should be 1024 (last step boundary), not full maxCacheSize.
    #expect(cache.innerState()[0].dim(2) == 1024)
    #expect(cache.innerState()[0].dim(2) < 4096)
}

@Test("RotatingKVCache stays trimmable while under maxCacheSize (no eviction)")
func testRotatingKVCacheUnderUtilisationNoEviction() async throws {
    // After 1024 writes (< maxSize=4096), the cache should still be in the
    // pre-rotation phase: isTrimmable=true and offset advances normally.
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
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

@Test("RotatingKVCache dispatcher unchanged when reserve() is unused")
func testRotatingKVCacheNoReserveDispatcherUnchanged() async throws {
    // Locks the back-compat guarantee: when initialAllocSize == nil, the
    // dispatcher should route S=1 → updateInPlace (step-sized buffer) and
    // S>1 → updateConcat (buffer == S). Pre-PR behaviour, byte-identical.
    let single = RotatingKVCache(maxSize: 4096, step: 256)
    let oneToken = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    _ = single.update(keys: oneToken, values: oneToken)
    // S=1 path: in-place, step-sized buffer.
    #expect(single.innerState()[0].dim(2) == 256)

    let multi = RotatingKVCache(maxSize: 4096, step: 256)
    let sixteenTokens = MLXArray.ones([1, 8, 16, 128], dtype: .bfloat16)
    _ = multi.update(keys: sixteenTokens, values: sixteenTokens)
    // S>1 path with no reserve: concat — buffer matches the chunk size, not step.
    #expect(multi.innerState()[0].dim(2) == 16)
}

@Test("RotatingKVCache rotation kicks in past maxCacheSize")
func testRotatingKVCacheRotationKicksInPastMax() async throws {
    // Push 96 single-token writes through a maxSize=64 cache. Rotation
    // should engage: buffer stays at 64, offset keeps counting upward, peek()
    // returns the live window in temporal order.
    let cache = RotatingKVCache(maxSize: 64, step: 16)
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

@Test("RotatingKVCache rotation works correctly with reserve()")
func testRotatingKVCacheRotationWithReserve() async throws {
    // reserve(maxCacheSize) → pre-allocate full window; then write 2× max
    // tokens. Buffer stays at maxCacheSize; rotation engages exactly once.
    let maxSize = 64
    let cache = RotatingKVCache(maxSize: maxSize, step: 16)
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

@Test("RotatingKVCache reserve respects keep across rotation")
func testRotatingKVCacheReserveWithKeepRotation() async throws {
    // With keep=4, the first 4 slots must be preserved across rotation.
    // Write distinguishable sentinels into the keep region by going through
    // ones-only and then zeros-only writes; after rotation the keep region
    // should still hold the original ones.
    let cache = RotatingKVCache(maxSize: 32, keep: 4, step: 8)
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

@Test("RotatingKVCache reserve(maxCacheSize) — exact boundary")
func testRotatingKVCacheReserveExactMaxBoundary() async throws {
    // reserve at exactly maxCacheSize: first allocation should be the full
    // buffer; subsequent writes up to maxCacheSize should not trigger any
    // re-allocation.
    let maxSize = 256
    let cache = RotatingKVCache(maxSize: maxSize, step: 64)
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

@Test("RotatingKVCache reserve(0) and negative are no-ops")
func testRotatingKVCacheReserveZeroAndNegativeNoOp() async throws {
    // reserve(0) and reserve(-N) should be silent no-ops. First write should
    // produce a step-sized buffer (the legacy default), proving
    // initialAllocSize stayed nil.
    let zeroCache = RotatingKVCache(maxSize: 4096, step: 256)
    zeroCache.reserve(0)
    let token = MLXArray.ones([1, 8, 1, 128], dtype: .bfloat16)
    _ = zeroCache.update(keys: token, values: token)
    #expect(zeroCache.innerState()[0].dim(2) == 256)

    let negativeCache = RotatingKVCache(maxSize: 4096, step: 256)
    negativeCache.reserve(-5)
    _ = negativeCache.update(keys: token, values: token)
    #expect(negativeCache.innerState()[0].dim(2) == 256)
}

@Test("RotatingKVCache reserve scales with batch dim B>1")
func testRotatingKVCacheReserveBatchedB2() async throws {
    // reserve(_:) controls dimension 2 (token dim); batch dim should be
    // inferred from the first write. Run B=2 and B=4 to confirm.
    let cacheB2 = RotatingKVCache(maxSize: 4096, step: 256)
    cacheB2.reserve(800)
    let chunkB2 = MLXArray.ones([2, 8, 64, 128], dtype: .bfloat16)
    _ = cacheB2.update(keys: chunkB2, values: chunkB2)
    #expect(cacheB2.innerState()[0].shape == [2, 8, 800, 128])

    let cacheB4 = RotatingKVCache(maxSize: 4096, step: 256)
    cacheB4.reserve(800)
    let chunkB4 = MLXArray.ones([4, 8, 64, 128], dtype: .bfloat16)
    _ = cacheB4.update(keys: chunkB4, values: chunkB4)
    #expect(cacheB4.innerState()[0].shape == [4, 8, 800, 128])
}

@Test("RotatingKVCache reserve allocates once for chunked writes within hint")
func testRotatingKVCacheReserveAllocationOnceOnly() async throws {
    // reserve(2048) + three 256-token chunks (768 total) → buffer should
    // remain at 2048 after every write. Proxy for "no concat happened".
    let cache = RotatingKVCache(maxSize: 4096, step: 256)
    cache.reserve(2048)

    let chunk = MLXArray.ones([1, 8, 256, 128], dtype: .bfloat16)
    for _ in 0 ..< 3 {
        _ = cache.update(keys: chunk, values: chunk)
        #expect(cache.innerState()[0].dim(2) == 2048)
    }
    #expect(cache.offset == 768)
}

@Test("RotatingKVCache no-reserve back-compat: full grow + rotate cycle")
func testRotatingKVCacheBackcompatExistingPath() async throws {
    // Smoke-test the legacy growth + rotation path with default settings.
    // Push 33 single-token writes through maxSize=32, default step=256.
    // Step gets clamped against maxSize-prev so the buffer grows to 32 then
    // rotation engages on the 33rd write.
    let cache = RotatingKVCache(maxSize: 32)
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
    let sub1 = KVCacheSimple()
    let sub2 = RotatingKVCache(maxSize: 32)
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
