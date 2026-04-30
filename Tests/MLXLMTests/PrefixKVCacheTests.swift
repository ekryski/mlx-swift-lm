// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - Cross-request prefix KV cache tests (spec 017 phase 1)
//
// All pure-Swift / pure-array — no model forward, no tokenizer, no chat
// template. Phase 1 tests cover:
//
//   1. `StablePrefixPolicy`: Identity + FixedTrim semantics.
//   2. `PrefixKey`: equality + hash invariance.
//   3. `PrefixKVCache`:
//      - lookup miss / hit / longest-prefix selection
//      - cross-key isolation (different model ID rejects)
//      - LRU bump on hit
//      - byte-budget eviction
//      - entry-count cap eviction
//      - stat counters (hits / misses / partial / insertions / evictions)
//      - clear / resetStats
//
// The tests build snapshots with placeholder MLXArrays sized to fixed
// known byte counts so the budget math is exact.

// MARK: - Helpers

private func makeKey(model: String = "test", layers: Int = 4) -> PrefixKey {
    PrefixKey(modelID: model, layerCount: layers, kvHeadDim: 64, kvBits: nil)
}

private func makeSnapshot(
    model: String = "test",
    tokens: [Int],
    layers: Int = 4,
    bytesPerLayer: Int = 1024
) -> PrefixSnapshot {
    let key = makeKey(model: model, layers: layers)
    // Use fp16 (Float16 = 2 bytes) — `bytesPerLayer / 2` Float16 values
    // gives an array of exactly `bytesPerLayer` bytes.
    let elementCount = bytesPerLayer / 2  // Float16 is 2 bytes
    let arr: () -> MLXArray = { MLXArray.zeros([elementCount], dtype: .float16) }
    let layerStates = (0 ..< layers).map { _ in
        LayerCacheState(tokenCount: tokens.count, arrays: [arr()])
    }
    return PrefixSnapshot(key: key, tokens: tokens, layerStates: layerStates)
}

// MARK: - Stable-prefix policy

@Suite
struct StablePrefixPolicyTests {

    @Test
    func `IdentityPolicy returns full token count`() {
        let p = IdentityPolicy()
        #expect(p.stablePrefixLen([1, 2, 3, 4, 5]) == 5)
        #expect(p.stablePrefixLen([]) == 0)
    }

    @Test
    func `FixedTrimPolicy trims fixed suffix`() {
        let p = FixedTrimPolicy(trimSuffix: 3)
        #expect(p.stablePrefixLen([1, 2, 3, 4, 5, 6, 7]) == 4)
    }

    @Test
    func `FixedTrimPolicy floors at zero on short prompts`() {
        let p = FixedTrimPolicy(trimSuffix: 10)
        #expect(p.stablePrefixLen([1, 2, 3]) == 0)
    }

    @Test
    func `FixedTrimPolicy with zero trim is identity-equivalent`() {
        let p = FixedTrimPolicy(trimSuffix: 0)
        #expect(p.stablePrefixLen([1, 2, 3]) == 3)
    }
}

// MARK: - PrefixKey

@Suite
struct PrefixKeyTests {

    @Test
    func `Equality requires all fields match`() {
        let a = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        let b = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        #expect(a == b)

        let differentLayers = PrefixKey(modelID: "x", layerCount: 16, kvHeadDim: 64, kvBits: nil)
        #expect(a != differentLayers)

        let differentBits = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: 4)
        #expect(a != differentBits)

        let differentModel = PrefixKey(modelID: "y", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        #expect(a != differentModel)
    }

    @Test
    func `Hashable produces consistent buckets`() {
        let a = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        let b = PrefixKey(modelID: "x", layerCount: 32, kvHeadDim: 64, kvBits: nil)
        var set = Set<PrefixKey>()
        set.insert(a)
        set.insert(b)
        #expect(set.count == 1)
    }
}

// MARK: - PrefixKVCache lookup / insert / eviction

@Suite
struct PrefixKVCacheTests {

    @Test
    func `Empty cache reports miss`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let r = cache.lookup(prefix: [1, 2, 3], key: makeKey())
        #expect(r == nil)
        #expect(cache.stats.misses == 1)
        #expect(cache.stats.hits == 0)
    }

    @Test
    func `Exact prefix match returns full snapshot`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1, 2, 3])
        cache.insert(snap)

        let r = cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        #expect(r != nil)
        #expect(r?.matchedLength == 3)
        #expect(cache.stats.hits == 1)
        #expect(cache.stats.partialHits == 1)  // 3 < 5 → partial
    }

    @Test
    func `Full-cover hit (matchedLen == request length) is not a partial hit`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1, 2, 3])
        cache.insert(snap)
        _ = cache.lookup(prefix: [1, 2, 3], key: makeKey())
        #expect(cache.stats.hits == 1)
        #expect(cache.stats.partialHits == 0)
    }

    @Test
    func `Longest-prefix selection picks the longest matching snapshot`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2]))
        cache.insert(makeSnapshot(tokens: [1, 2, 3, 4]))
        cache.insert(makeSnapshot(tokens: [1, 2, 3]))

        let r = cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        #expect(r?.matchedLength == 4)
        #expect(r?.snapshot.tokens == [1, 2, 3, 4])
    }

    @Test
    func `Cross-key isolation rejects mismatched model ID`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(model: "modelA", tokens: [1, 2, 3]))

        let r = cache.lookup(
            prefix: [1, 2, 3, 4],
            key: PrefixKey(modelID: "modelB", layerCount: 4, kvHeadDim: 64))
        #expect(r == nil)
        #expect(cache.stats.misses == 1)
    }

    @Test
    func `Mismatching prefix returns nil`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2, 3]))

        let r = cache.lookup(prefix: [9, 8, 7, 6], key: makeKey())
        #expect(r == nil)
    }

    @Test
    func `Snapshot longer than request returns nil (won't truncate)`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2, 3, 4, 5]))

        let r = cache.lookup(prefix: [1, 2], key: makeKey())
        #expect(r == nil)
    }

    @Test
    func `Re-insert with same key replaces in place (no double-count)`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let snap = makeSnapshot(tokens: [1, 2, 3])
        cache.insert(snap)
        let bytesAfterFirst = cache.stats.bytesUsed
        cache.insert(snap)
        #expect(cache.count == 1)
        #expect(cache.stats.bytesUsed == bytesAfterFirst)
    }

    @Test
    func `LRU bump moves matched entry to MRU position`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        let s1 = makeSnapshot(tokens: [1, 2])
        let s2 = makeSnapshot(tokens: [3, 4])
        let s3 = makeSnapshot(tokens: [5, 6])
        cache.insert(s1)
        cache.insert(s2)
        cache.insert(s3)
        // Order: s1 (oldest), s2, s3 (newest).

        // Hit s1: should move it to the back.
        _ = cache.lookup(prefix: [1, 2, 99], key: makeKey())

        let order = cache.entrySnapshot.map { $0.tokens }
        #expect(order == [[3, 4], [5, 6], [1, 2]])
    }

    @Test
    func `Entry-count cap evicts oldest first`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 2)
        cache.insert(makeSnapshot(tokens: [1]))
        cache.insert(makeSnapshot(tokens: [2]))
        // Inserting third should evict tokens=[1].
        cache.insert(makeSnapshot(tokens: [3]))
        #expect(cache.count == 2)
        let surviving = Set(cache.entrySnapshot.map { $0.tokens.first! })
        #expect(surviving == [2, 3])
        #expect(cache.stats.evictions == 1)
    }

    @Test
    func `Byte-budget eviction frees space for large insert`() {
        // 4 layers × 1024 bytes = 4 KiB per snapshot. Cache holds 8 KiB
        // → exactly two snapshots before eviction.
        let cache = PrefixKVCache(maxBytes: 8 * 1024, maxEntries: 100)
        cache.insert(makeSnapshot(tokens: [1], bytesPerLayer: 1024))
        cache.insert(makeSnapshot(tokens: [2], bytesPerLayer: 1024))
        #expect(cache.count == 2)

        // Third insert (4 KiB) → must evict to fit (8 + 4 > 8).
        cache.insert(makeSnapshot(tokens: [3], bytesPerLayer: 1024))
        #expect(cache.count == 2)
        #expect(cache.stats.evictions == 1)
    }

    @Test
    func `clear empties entries and resets bytes`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2]))
        cache.insert(makeSnapshot(tokens: [3, 4]))
        cache.clear()
        #expect(cache.count == 0)
        #expect(cache.stats.bytesUsed == 0)
        #expect(cache.stats.entryCount == 0)
        // Stats counters preserved.
        #expect(cache.stats.insertions == 2)
    }

    @Test
    func `resetStats keeps occupancy counts, zeros activity counters`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2]))
        _ = cache.lookup(prefix: [1, 2], key: makeKey())
        _ = cache.lookup(prefix: [9], key: makeKey())
        #expect(cache.stats.hits == 1)
        #expect(cache.stats.misses == 1)

        cache.resetStats()
        #expect(cache.stats.hits == 0)
        #expect(cache.stats.misses == 0)
        #expect(cache.stats.insertions == 0)
        // Occupancy preserved.
        #expect(cache.stats.bytesUsed > 0)
        #expect(cache.stats.entryCount == 1)
    }

    @Test
    func `hitRate computation is correct`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2]))
        _ = cache.lookup(prefix: [1, 2], key: makeKey())
        _ = cache.lookup(prefix: [1, 2], key: makeKey())
        _ = cache.lookup(prefix: [9], key: makeKey())
        // 2 hits, 1 miss → 2/3.
        #expect(abs(cache.stats.hitRate - 2.0/3.0) < 1e-9)
    }

    @Test
    func `meanMatchedLength averages over hits only`() {
        let cache = PrefixKVCache(maxBytes: 1_000_000, maxEntries: 4)
        cache.insert(makeSnapshot(tokens: [1, 2]))
        cache.insert(makeSnapshot(tokens: [1, 2, 3, 4]))
        // Hit on [1, 2] (matched 2)
        _ = cache.lookup(prefix: [1, 2, 9], key: makeKey())
        // Hit on [1, 2, 3, 4] (matched 4)
        _ = cache.lookup(prefix: [1, 2, 3, 4, 5], key: makeKey())
        // Miss
        _ = cache.lookup(prefix: [9, 8], key: makeKey())

        #expect(cache.stats.hits == 2)
        #expect(cache.stats.totalMatchedTokens == 6)
        #expect(abs(cache.stats.meanMatchedLength - 3.0) < 1e-9)
    }
}
