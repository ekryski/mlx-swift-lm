import Foundation
import MLX
import MLXLMCommon
import Testing

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
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
        case let quantized as QuantizedKVCache:
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
