import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// Tests for spec 034 phase 1 — BlockSummaryCache protocol + StandardKVCache /
// TurboQuantizedKVCache conformances. Verifies summary correctness against
// brute-force reference computations and lifecycle invalidation.

// MARK: - helpers

private func bruteForceBlockSummaries(
    keys: MLXArray, blockSize: Int
) -> (means: MLXArray, maxAbs: MLXArray) {
    let B = keys.dim(0)
    let H = keys.dim(1)
    let T = keys.dim(2)
    let D = keys.dim(3)
    let blockCount = T / blockSize

    let promoted = keys.asType(.float32)
    var means = MLXArray.zeros([B, H, blockCount, D], dtype: .float32)
    var maxAbs = MLXArray.zeros([B, H, blockCount, D], dtype: .float32)

    for i in 0..<blockCount {
        let lo = i * blockSize
        let hi = lo + blockSize
        let block = promoted[.ellipsis, lo..<hi, 0...]
        means[.ellipsis, i..<(i + 1), 0...] =
            block.mean(axis: 2, keepDims: true)
        maxAbs[.ellipsis, i..<(i + 1), 0...] =
            abs(block).max(axis: 2, keepDims: true)
    }

    return (means.asType(keys.dtype), maxAbs.asType(keys.dtype))
}

private func assertSummariesClose(
    _ actual: BlockKSummaries, _ expected: (means: MLXArray, maxAbs: MLXArray),
    tol: Float = 1e-2,
    label: String = ""
) {
    #expect(actual.means.shape == expected.means.shape, "means shape \(label)")
    #expect(actual.maxAbs.shape == expected.maxAbs.shape, "maxAbs shape \(label)")
    let meansClose = allClose(
        actual.means.asType(.float32), expected.means.asType(.float32),
        rtol: Double(tol), atol: Double(tol)
    ).item(Bool.self)
    let maxAbsClose = allClose(
        actual.maxAbs.asType(.float32), expected.maxAbs.asType(.float32),
        rtol: Double(tol), atol: Double(tol)
    ).item(Bool.self)
    #expect(meansClose, "means not close \(label)")
    #expect(maxAbsClose, "maxAbs not close \(label)")
}

// MARK: - computeKBlockSummaries

@Test func testComputeKBlockSummariesMatchesBruteForce() {
    let keys = MLXRandom.normal([1, 4, 256, 32], dtype: .float32)
    let result = computeKBlockSummaries(keys: keys, blockSize: 64)
    let summaries = try! #require(result)
    #expect(summaries.blockSize == 64)
    #expect(summaries.blockCount == 4)
    #expect(summaries.means.shape == [1, 4, 4, 32])

    let reference = bruteForceBlockSummaries(keys: keys, blockSize: 64)
    assertSummariesClose(summaries, reference, tol: 1e-5)
}

@Test func testComputeKBlockSummariesDropsPartialTail() {
    // T=200 with blockSize=64 → 3 full blocks (192 tokens), tail of 8 dropped.
    let keys = MLXRandom.normal([1, 2, 200, 16], dtype: .float32)
    let result = computeKBlockSummaries(keys: keys, blockSize: 64)
    let summaries = try! #require(result)
    #expect(summaries.blockCount == 3)

    // Reference computed on the truncated [..192] slice should match.
    let truncated = keys[.ellipsis, ..<192, 0...]
    let reference = bruteForceBlockSummaries(keys: truncated, blockSize: 64)
    assertSummariesClose(summaries, reference, tol: 1e-5)
}

@Test func testComputeKBlockSummariesShortInputReturnsNil() {
    let keys = MLXRandom.normal([1, 2, 16, 8], dtype: .float32)
    let result = computeKBlockSummaries(keys: keys, blockSize: 64)
    #expect(result == nil)
}

@Test func testComputeKBlockSummariesBlockSizeOne() {
    // Edge case: blockSize=1 means each token is its own block.
    let keys = MLXRandom.normal([1, 2, 8, 4], dtype: .float32)
    let result = computeKBlockSummaries(keys: keys, blockSize: 1)
    let summaries = try! #require(result)
    #expect(summaries.blockCount == 8)
    // Mean over a single-token block is the token itself.
    let close = allClose(
        summaries.means, keys, rtol: 1e-5, atol: 1e-5
    ).item(Bool.self)
    #expect(close)
}

// MARK: - StandardKVCache conformance

@Test func testStandardKVCacheBlockSummaries() {
    let cache = StandardKVCache()
    #expect(cache.blockSummaries == nil, "summaries should be nil before compute")

    let keys = MLXRandom.normal([1, 4, 256, 32], dtype: .float32)
    let values = MLXRandom.normal([1, 4, 256, 32], dtype: .float32)
    _ = cache.update(keys: keys, values: values)

    cache.computeBlockSummaries(blockSize: 64)
    let summaries = try! #require(cache.blockSummaries)
    #expect(summaries.blockSize == 64)
    #expect(summaries.blockCount == 4)

    let reference = bruteForceBlockSummaries(keys: keys, blockSize: 64)
    assertSummariesClose(summaries, reference, tol: 1e-5, label: "StandardKVCache")
}

@Test func testStandardKVCacheBlockSummariesAfterMultipleUpdates() {
    let cache = StandardKVCache()
    let chunk1 = MLXRandom.normal([1, 2, 128, 16], dtype: .float32)
    let chunk2 = MLXRandom.normal([1, 2, 128, 16], dtype: .float32)
    _ = cache.update(keys: chunk1, values: chunk1)
    _ = cache.update(keys: chunk2, values: chunk2)

    cache.computeBlockSummaries(blockSize: 64)
    let summaries = try! #require(cache.blockSummaries)
    #expect(summaries.blockCount == 4) // 256 / 64

    // Reference computed on the concatenated K.
    let concat = concatenated([chunk1, chunk2], axis: 2)
    let reference = bruteForceBlockSummaries(keys: concat, blockSize: 64)
    assertSummariesClose(summaries, reference, tol: 1e-5)
}

@Test func testStandardKVCacheTrimInvalidatesSummaries() {
    let cache = StandardKVCache()
    let keys = MLXRandom.normal([1, 2, 128, 16], dtype: .float32)
    _ = cache.update(keys: keys, values: keys)

    cache.computeBlockSummaries(blockSize: 64)
    #expect(cache.blockSummaries != nil)

    let trimmed = cache.trim(32)
    #expect(trimmed == 32)
    #expect(cache.blockSummaries == nil, "trim should invalidate summaries")
}

@Test func testStandardKVCacheExplicitInvalidate() {
    let cache = StandardKVCache()
    let keys = MLXRandom.normal([1, 2, 128, 16], dtype: .float32)
    _ = cache.update(keys: keys, values: keys)

    cache.computeBlockSummaries(blockSize: 64)
    #expect(cache.blockSummaries != nil)
    cache.invalidateBlockSummaries()
    #expect(cache.blockSummaries == nil)
}

@Test func testStandardKVCacheEmptyCacheNoSummary() {
    let cache = StandardKVCache()
    cache.computeBlockSummaries(blockSize: 64)
    #expect(cache.blockSummaries == nil)
}

@Test func testStandardKVCacheTooFewTokensNoSummary() {
    let cache = StandardKVCache()
    let keys = MLXRandom.normal([1, 2, 16, 8], dtype: .float32)
    _ = cache.update(keys: keys, values: keys)

    cache.computeBlockSummaries(blockSize: 64)
    #expect(cache.blockSummaries == nil, "T<blockSize should yield no summary")
}

@Test func testStandardKVCacheWindowedSummary() {
    let cache = StandardKVCache(maxSize: 256)
    let keys = MLXRandom.normal([1, 2, 256, 16], dtype: .float32)
    _ = cache.update(keys: keys, values: keys)

    cache.computeBlockSummaries(blockSize: 64)
    let summaries = try! #require(cache.blockSummaries)
    #expect(summaries.blockCount == 4)
}

// MARK: - TurboQuantizedKVCache conformance (raw-K, pre-compress)

@Test func testTurboQuantKVCachePreCompressSummaries() {
    let cache = TurboQuantizedKVCache(bits: 4, headDim: 32)
    let keys = MLXRandom.normal([1, 2, 128, 32], dtype: .float32)
    let values = MLXRandom.normal([1, 2, 128, 32], dtype: .float32)
    _ = cache.update(keys: keys, values: values)

    // Cache is in raw mode pre-compress; summaries computed directly on raw K.
    cache.computeBlockSummaries(blockSize: 64)
    let summaries = try! #require(cache.blockSummaries)
    #expect(summaries.blockCount == 2)

    let reference = bruteForceBlockSummaries(keys: keys, blockSize: 64)
    assertSummariesClose(summaries, reference, tol: 1e-5, label: "TurboQuant pre-compress")
}

@Test func testTurboQuantKVCacheEmptyNoSummary() {
    let cache = TurboQuantizedKVCache(bits: 4, headDim: 32)
    cache.computeBlockSummaries(blockSize: 64)
    #expect(cache.blockSummaries == nil)
}

@Test func testTurboQuantKVCacheTrimInvalidates() {
    let cache = TurboQuantizedKVCache(bits: 4, headDim: 32)
    let keys = MLXRandom.normal([1, 2, 128, 32], dtype: .float32)
    _ = cache.update(keys: keys, values: keys)

    cache.computeBlockSummaries(blockSize: 64)
    #expect(cache.blockSummaries != nil)
    _ = cache.trim(64)
    #expect(cache.blockSummaries == nil)
}
