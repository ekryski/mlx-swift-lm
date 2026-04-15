import Foundation
import MLX
import MLXLMCommon
import Testing

private let cacheCreators: [() -> any KVCache] = [
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

// MARK: - Prefill → Decode transition bug (caddf7a regression)

/// When updateMultiToken() stores fewer tokens than maxCacheSize, the buffer
/// must still be maxCacheSize wide. Otherwise the subsequent updateSingleToken()
/// writes at idx=totalLen which is out of bounds on a totalLen-sized buffer.
/// This caused decode incoherence for ALL Gemma4 models at short contexts.
@Test
func testRotatingCachePrefillThenDecodeDoesNotCorrupt() async throws {
    let maxSize = 1024
    let cache = RotatingKVCache(maxSize: maxSize)

    // Prefill with 39 tokens (simulating a 40-token prompt, last token reserved)
    let prefillK = MLXRandom.normal([1, 2, 39, 4]).asType(.bfloat16)
    let prefillV = MLXRandom.normal([1, 2, 39, 4]).asType(.bfloat16)
    let (_, _) = cache.update(keys: prefillK, values: prefillV)

    #expect(cache.offset == 39)
    let meta = cache.metaState
    let idxAfterPrefill = Int(meta[4])!
    #expect(idxAfterPrefill == 39)

    // First decode token — with the old bug (undersized buffer), writing at
    // idx=39 on a 39-element buffer silently corrupts → incoherent decode.
    let decodeK = MLXRandom.normal([1, 2, 1, 4]).asType(.bfloat16)
    let decodeV = MLXRandom.normal([1, 2, 1, 4]).asType(.bfloat16)
    let (retK, retV) = cache.update(keys: decodeK, values: decodeV)
    eval(retK, retV)

    #expect(cache.offset == 40)
    #expect(retK.dim(2) == 40, "Should return 40 tokens after prefill(39) + decode(1)")
    #expect(retV.dim(2) == 40)

    // Continue decoding — should not corrupt
    for _ in 0..<10 {
        let k = MLXRandom.normal([1, 2, 1, 4]).asType(.bfloat16)
        let v = MLXRandom.normal([1, 2, 1, 4]).asType(.bfloat16)
        let (rk, rv) = cache.update(keys: k, values: v)
        eval(rk, rv)
        #expect(rk.dim(2) > 0)
    }
    #expect(cache.offset == 50)
}

/// Same test but with prefill that exactly fills the cache.
@Test
func testRotatingCachePrefillExactFillThenDecode() async throws {
    let maxSize = 8
    let cache = RotatingKVCache(maxSize: maxSize)

    // Prefill exactly maxSize tokens
    let prefillK = MLXRandom.normal([1, 2, 8, 4]).asType(.bfloat16)
    let prefillV = MLXRandom.normal([1, 2, 8, 4]).asType(.bfloat16)
    let (_, _) = cache.update(keys: prefillK, values: prefillV)

    #expect(cache.offset == 8)
    let meta = cache.metaState
    let idx = Int(meta[4])!
    #expect(idx == 0, "After filling exactly maxSize, idx should wrap to 0")

    // Decode should work fine
    let decodeK = MLXRandom.normal([1, 2, 1, 4]).asType(.bfloat16)
    let decodeV = MLXRandom.normal([1, 2, 1, 4]).asType(.bfloat16)
    let (retK, _) = cache.update(keys: decodeK, values: decodeV)
    eval(retK)
    #expect(retK.dim(2) == maxSize, "Full buffer should be returned after wrap")
}

// MARK: - RotatingKVCache trim/circularWrite bug tests

/// M1: RotatingKVCache.trim() must not make idx negative.
/// Scenario: cache wraps around (offset > maxCacheSize), then trim is called
/// (e.g. from speculative decoding rejection in Evaluate.swift:1346).
/// If idx < trimmed, idx becomes negative, corrupting subsequent circularWrite.
@Test
func testRotatingCacheTrimDoesNotProduceNegativeIdx() async throws {
    let maxSize = 8
    let cache = RotatingKVCache(maxSize: maxSize)

    // Fill cache past the wrap point: 10 single-token updates → offset=10, idx=2
    for _ in 0..<10 {
        let k = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        _ = cache.update(keys: k, values: v)
    }

    #expect(cache.offset == 10)
    // idx should be 10 % 8 = 2 (internal state, verify via metaState)
    let meta = cache.metaState
    let idxBefore = Int(meta[4])!
    #expect(idxBefore == 2)

    // Trim 3 tokens: trimmed = min(10, 3) = 3, offset = 7, idx should NOT go to -1
    cache.trim(3)

    let metaAfter = cache.metaState
    let idxAfter = Int(metaAfter[4])!
    #expect(idxAfter >= 0, "idx became negative after trim: \(idxAfter)")
    #expect(cache.offset == 7)
}

/// M1 boundary: trim exactly idx amount → idx should be 0, not negative.
@Test
func testRotatingCacheTrimExactlyIdxIsZero() async throws {
    let cache = RotatingKVCache(maxSize: 8)

    // 10 tokens → offset=10, idx=2
    for _ in 0..<10 {
        let k = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        _ = cache.update(keys: k, values: v)
    }

    // Trim exactly 2 (== idx)
    cache.trim(2)

    let meta = cache.metaState
    let idxAfter = Int(meta[4])!
    #expect(idxAfter >= 0, "idx became negative after trim: \(idxAfter)")
}

/// M2: After trim corrupts idx, subsequent updateSingleToken uses invalid idx for
/// circularWrite, causing out-of-bounds write.
@Test
func testRotatingCacheUpdateAfterTrimDoesNotCrash() async throws {
    let cache = RotatingKVCache(maxSize: 8)

    // 10 tokens → offset=10, idx=2
    for _ in 0..<10 {
        let k = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        _ = cache.update(keys: k, values: v)
    }

    // Trim 3 → idx was 2, would become -1 without fix
    cache.trim(3)

    // This should not crash or produce garbage
    let k = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
    let v = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
    let (retK, retV) = cache.update(keys: k, values: v)
    eval(retK, retV)

    // Verify shapes are valid
    #expect(retK.dim(2) > 0)
    #expect(retV.dim(2) > 0)
}

/// Trim on a pre-wrap cache (offset < maxCacheSize) should work correctly.
@Test
func testRotatingCacheTrimPreWrap() async throws {
    let cache = RotatingKVCache(maxSize: 8)

    // 5 tokens → offset=5, idx=5
    for _ in 0..<5 {
        let k = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        _ = cache.update(keys: k, values: v)
    }

    // Trim 3 → offset=2, idx=2
    let trimmed = cache.trim(3)
    #expect(trimmed == 3)
    #expect(cache.offset == 2)

    let meta = cache.metaState
    let idx = Int(meta[4])!
    #expect(idx == 2)
}

/// Trim full offset on wrapped cache should reset to valid state.
@Test
func testRotatingCacheTrimFullOffsetAfterWrap() async throws {
    let cache = RotatingKVCache(maxSize: 4)

    // 6 tokens → offset=6, idx=2
    for _ in 0..<6 {
        let k = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        let v = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
        _ = cache.update(keys: k, values: v)
    }

    let meta = cache.metaState
    let idxBefore = Int(meta[4])!
    #expect(idxBefore == 2)

    // Trim all 6: trimmed=6, offset=0, idx should be >= 0
    cache.trim(6)
    #expect(cache.offset == 0)

    let metaAfter = cache.metaState
    let idxAfter = Int(metaAfter[4])!
    #expect(idxAfter >= 0, "idx became negative: \(idxAfter)")
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
