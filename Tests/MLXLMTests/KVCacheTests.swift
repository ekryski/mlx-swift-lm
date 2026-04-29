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

// MARK: - RotatingKVCache prefill regression coverage
//
// These tests exercise the path that previously triggered Metal OOMs at
// B>1 long-context prefill. The legacy `updateConcat` grew the underlying
// buffer in `step`-sized chunks via `concatenated`, which holds the old
// + new buffer simultaneously and doubles memory transiently per resize.
// The fix routes all writes through `updateInPlace`, which now allocates
// the full `maxCacheSize` buffer up front. The tests below check the
// behaviours that the fix has to preserve while removing the surge:
// shape correctness, content correctness, multi-chunk accumulation,
// post-`copy()` mutation, and the B>1 batch case.

/// Build a constant-valued `[1, 8, S, 64]` bfloat16 tensor for cache tests.
private func constantTensor(value: Float, S: Int) -> MLXArray {
    MLXArray.zeros([1, 8, S, 64], dtype: .bfloat16) + value.asMLXArray(dtype: .bfloat16)
}

@Test("RotatingKVCache: multi-token write returns correct shape and content")
func testRotatingKVCachePrefillSingleChunk() async throws {
    let cache = RotatingKVCache(maxSize: 32)
    let keys = constantTensor(value: 7, S: 16)
    let values = constantTensor(value: 11, S: 16)

    let (k, v) = cache.update(keys: keys, values: values)

    #expect(k.shape == [1, 8, 16, 64])
    #expect(v.shape == [1, 8, 16, 64])
    #expect(cache.offset == 16)
    #expect(allClose(k, keys).item(Bool.self))
    #expect(allClose(v, values).item(Bool.self))
}

@Test("RotatingKVCache: chunked prefill accumulates correctly")
func testRotatingKVCachePrefillMultipleChunks() async throws {
    // Simulates the prefill loop: model passes successive `[B, kvHeads, chunk, headDim]`
    // chunks to the cache. Each chunk must be appended without losing prior content.
    let cache = RotatingKVCache(maxSize: 32)
    let chunkSize = 8
    var allKeys: [MLXArray] = []

    for chunkIdx in 0..<3 {
        let chunkValue = Float(chunkIdx + 1)  // 1, 2, 3 — distinct per chunk
        let keys = constantTensor(value: chunkValue, S: chunkSize)
        allKeys.append(keys)

        let (k, _) = cache.update(keys: keys, values: keys)
        let expectedLen = (chunkIdx + 1) * chunkSize
        #expect(k.shape == [1, 8, expectedLen, 64])
    }

    let final = cache.update(
        keys: MLXArray.zeros([1, 8, 1, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 1, 64], dtype: .bfloat16)
    )

    let expectedFinalLen = 3 * chunkSize + 1
    #expect(final.0.shape == [1, 8, expectedFinalLen, 64])
    // Verify each prior chunk's distinct value lives at its expected position.
    for chunkIdx in 0..<3 {
        let start = chunkIdx * chunkSize
        let end = start + chunkSize
        let slice = final.0[.ellipsis, start..<end, 0...]
        #expect(allClose(slice, allKeys[chunkIdx]).item(Bool.self))
    }
}

@Test("RotatingKVCache: B>1 multi-token prefill produces independent per-batch slices")
func testRotatingKVCachePrefillBatched() async throws {
    // The headline regression test: B=4 multi-token write must produce a
    // [B, kvHeads, S, headDim] return where each batch member's content
    // matches its input. Previously this path triggered the
    // concatenation-driven memory surge that OOMed Qwen3.5-9B at long ctx.
    let cache = RotatingKVCache(maxSize: 64)

    var perBatchKeys: [MLXArray] = []
    for b in 0..<4 {
        // Distinct value per batch member so we can verify slice independence.
        let k = constantTensor(value: Float(b + 1), S: 16)
        perBatchKeys.append(k)
    }
    let stackedK = concatenated(perBatchKeys, axis: 0)  // [4, 8, 16, 64]
    #expect(stackedK.shape == [4, 8, 16, 64])

    let (k, v) = cache.update(keys: stackedK, values: stackedK)

    #expect(k.shape == [4, 8, 16, 64])
    #expect(v.shape == [4, 8, 16, 64])
    #expect(cache.offset == 16)

    for b in 0..<4 {
        let slice = k[b ..< (b + 1), .ellipsis]
        #expect(allClose(slice, perBatchKeys[b]).item(Bool.self))
    }
}

@Test("RotatingKVCache: write into copied cache produces correct shape")
func testRotatingKVCacheWriteAfterCopy() async throws {
    // Regression test for the `copy()` interaction: the state setter
    // installs sliced views (e.g. `[1, 8, offset, 64]`), and the next
    // write must grow the buffer back to `maxCacheSize` rather than try
    // to slice past the view's bounds.
    let original = RotatingKVCache(maxSize: 32)
    let initial = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = original.update(keys: initial, values: initial)

    let copied = original.copy() as! RotatingKVCache
    #expect(copied.offset == 4)

    // This previously broadcast-failed because the copy's `keys` was a
    // [1,8,4,64] sliced view, so writing into [4..6] was out of bounds.
    let appended = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let (k, v) = copied.update(keys: appended, values: appended)

    #expect(k.shape == [1, 8, 6, 64])
    #expect(v.shape == [1, 8, 6, 64])
    #expect(copied.offset == 6)
    // Original must remain unchanged.
    #expect(original.offset == 4)
}

@Test("RotatingKVCache: pre-allocated buffer is maxCacheSize after first write")
func testRotatingKVCachePreallocatesFullBuffer() async throws {
    // Verifies the fix's central claim: after the first write, the
    // underlying buffer is sized to `maxCacheSize`, not `step`. This is
    // what eliminates the per-chunk concatenation surge during prefill.
    let cache = RotatingKVCache(maxSize: 256, step: 32)
    let firstWrite = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = cache.update(keys: firstWrite, values: firstWrite)

    // The full state — when offset (4) < maxCacheSize (256) — returns the
    // sliced view, but the underlying inner state must be the full buffer.
    let inner = cache.innerState()
    #expect(inner.count == 2)
    #expect(inner[0].shape == [1, 8, 256, 64])
    #expect(inner[1].shape == [1, 8, 256, 64])
}

@Test("RotatingKVCache: rotation past maxCacheSize still works after fix")
func testRotatingKVCacheRotation() async throws {
    // The rotation path was preserved; verify it still triggers correctly
    // when the cache fills up. With keep=2, after writing maxCacheSize
    // tokens, the next write should land at idx=2 (overwriting the
    // post-keep prefix).
    let maxSize = 16
    let keep = 2
    let cache = RotatingKVCache(maxSize: maxSize, keep: keep)

    // Fill the cache exactly to maxCacheSize.
    let initial = MLXArray.ones([1, 4, maxSize, 32], dtype: .bfloat16)
    _ = cache.update(keys: initial, values: initial)
    #expect(cache.offset == maxSize)

    // Single-token write past maxCacheSize triggers rotation.
    let extra = MLXArray.zeros([1, 4, 1, 32], dtype: .bfloat16)
    let (k, _) = cache.update(keys: extra, values: extra)

    // Cache should have rotated and returned the full buffer.
    #expect(k.shape == [1, 4, maxSize, 32])
    #expect(cache.offset == maxSize + 1)
}
